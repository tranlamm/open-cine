"""Where the subject is, every 0.5 s of a clip (CPU, OpenCV only): ``index/subjects/<clip>.json``.

Per sample, the first that answers: **face** (YuNet, ~230 KB ONNX fetched once into
``models_dir``, sha256-checked), **motion** (largest blob moving against the camera) or
**saliency** (frequency-tuned, Achanta 2009). A row is ``{f, box: [x, y, w, h] (fractions), kind,
conf}``; empty samples are left out. ``plan.reframe`` turns the rows into a 9:16 crop path.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
from pathlib import Path

import cv2
import httpx
import numpy as np

from opencine.core.config import get_settings
from opencine.core.time import Rate, frames_for, to_frame

log = logging.getLogger(__name__)

SAMPLE_FPS = 2
GAP_S = 0.25
FACE_EDGE = 640  # long edge the face detector sees
FACE_SCORE = 0.7
MOTION_EDGE = 160
MOTION_MIN_AREA, MOTION_MAX_AREA = 0.004, 0.5  # of the frame
SAL_EDGE = 64
YUNET = ("face_detection_yunet_2023mar.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
         "face_detection_yunet_2023mar.onnx",
         "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
VERSION = 1


def face_model() -> Path | None:
    """The YuNet model file (downloaded and verified on first use), or None when face
    detection is off or the model cannot be fetched."""
    s = get_settings()
    if not s.face_detect:
        return None
    name, url, sha = YUNET
    path = Path(s.models_dir) / name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    try:
        data = httpx.get(url, follow_redirects=True, timeout=30).raise_for_status().content
    except httpx.HTTPError as e:
        log.warning("face model download failed (%s); tracking without faces", e)
        return None
    if hashlib.sha256(data).hexdigest() != sha:
        log.warning("face model checksum mismatch; tracking without faces")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def params() -> dict:
    """Cache key part: the method and whether faces were available."""
    return {"v": VERSION, "fps": SAMPLE_FPS, "gap": GAP_S, "face": [FACE_EDGE, FACE_SCORE],
            "motion": [MOTION_EDGE, MOTION_MIN_AREA, MOTION_MAX_AREA], "sal": SAL_EDGE,
            "faces": face_model() is not None}


class FaceFinder:
    def __init__(self, model: Path | None):
        self.model = model
        self._det = None
        self._size: tuple[int, int] | None = None

    def __call__(self, img: np.ndarray) -> tuple[list[float], float] | None:
        """Largest confident face: ``([x, y, w, h] fractions, score)``."""
        if self.model is None:
            return None
        h, w = img.shape[:2]
        k = min(1.0, FACE_EDGE / max(h, w))
        small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
        sh, sw = small.shape[:2]
        if self._det is None:
            try:  # OpenCV 5 warns once per net that it ignores the target; nothing to act on
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                pass
            self._det = cv2.FaceDetectorYN.create(str(self.model), "", (sw, sh), FACE_SCORE)
        if self._size != (sw, sh):
            self._det.setInputSize((sw, sh))
            self._size = (sw, sh)
        _, faces = self._det.detect(small)
        if faces is None or not len(faces):
            return None
        best = max(faces, key=lambda f: f[2] * f[3] * f[-1])
        x, y, fw, fh = best[:4]
        return _clip_box([x / sw, y / sh, fw / sw, fh / sh]), float(best[-1])


def _clip_box(b: list[float]) -> list[float]:
    """Inside the frame, as plain floats (detectors return numpy float32, which JSON refuses)."""
    x, y, w, h = (float(v) for v in b)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(1.0, x + w), min(1.0, y + h)
    return [round(x0, 3), round(y0, 3), round(max(0.0, x1 - x0), 3), round(max(0.0, y1 - y0), 3)]


def _small(img: np.ndarray, edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    k = edge / max(h, w)
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def _grey(img: np.ndarray, edge: int) -> np.ndarray:
    return cv2.cvtColor(_small(img, edge), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _largest_blob(mask: np.ndarray) -> tuple[list[float], float] | None:
    """Box (fractions) and area share of the largest component of a boolean mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    H, W = mask.shape
    return _clip_box([x / W, y / H, w / W, h / H]), float(area) / (W * H)


def motion_box(a: np.ndarray, b: np.ndarray) -> tuple[list[float], float] | None:
    """Largest blob moving between two small greys (``GAP_S`` apart) once the global shift
    (phase correlation) is taken out."""
    if a.std() < 3 or b.std() < 3:
        return None
    (sx, sy), _ = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64))
    h, w = a.shape
    warped = cv2.warpAffine(b, np.float32([[1, 0, -sx], [0, 1, -sy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    # the strongest shift can be the subject's own (a bright thing on a soft background): keep
    # it only if it aligns most of the picture better than no shift at all
    if np.median(np.abs(a - warped)) >= np.median(np.abs(a - b)):
        warped, sx, sy = b, 0.0, 0.0
    diff = cv2.GaussianBlur(np.abs(a - warped), (5, 5), 0)
    m = int(min(h, w) * 0.05) + int(max(abs(sx), abs(sy))) + 1  # borders carry warp artefacts
    diff[:m], diff[-m:], diff[:, :m], diff[:, -m:] = 0, 0, 0, 0
    thr = max(12.0, 0.5 * float(np.quantile(diff, 0.995)))
    mask = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    found = _largest_blob(mask > 0)
    if found is None or not MOTION_MIN_AREA <= found[1] <= MOTION_MAX_AREA:
        return None
    return found[0], round(min(1.0, 0.3 + found[1] * 10), 3)


def saliency_box(img: np.ndarray) -> tuple[list[float], float] | None:
    """Frequency-tuned saliency on a small Lab image (blurred pixel vs mean colour): the largest
    blob of the most salient 5 %."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if lab[..., 0].std() < 3:
        return None
    blur = cv2.GaussianBlur(lab, (5, 5), 0)
    sal = np.linalg.norm(blur - lab.reshape(-1, 3).mean(axis=0), axis=2)
    found = _largest_blob(sal >= max(1e-6, float(np.quantile(sal, 0.95))))
    return (found[0], 0.2) if found else None


def track(proxy: str | Path, rate: Rate, n_frames: int, cuts: list[int]) -> list[dict]:
    """Subject rows of one clip (see the module doc), decoding the proxy once."""
    picks = sorted({f for k in range(math.ceil(n_frames / rate * SAMPLE_FPS) + 1)
                    if (f := to_frame(k / SAMPLE_FPS, rate, "nearest")) < n_frames})
    gap = max(1, frames_for(GAP_S, rate))

    def shot(f: int) -> int:
        return bisect.bisect_right(cuts, f)

    partner = {}
    for f in picks:
        for p in (f + gap, f - gap):
            if 0 <= p < n_frames and shot(p) == shot(f):
                partner[f] = p
                break
    pick_set = set(picks)
    needed = pick_set | set(partner.values())
    faces = FaceFinder(face_model())
    imgs: dict[int, np.ndarray] = {}
    greys: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(proxy))
    try:
        for i in range(max(needed) + 1 if needed else 0):
            if not cap.grab():
                break
            if i not in needed:
                continue
            ok, img = cap.retrieve()
            if not ok:
                continue
            greys[i] = _grey(img, MOTION_EDGE)
            if i in pick_set:
                imgs[i] = img
    finally:
        cap.release()
    rows = []
    for f in picks:
        img = imgs.get(f)
        if img is None:
            continue
        found, kind = faces(img), "face"
        if found is None and partner.get(f) in greys:
            found, kind = motion_box(greys[f], greys[partner[f]]), "motion"
        if found is None:
            found, kind = saliency_box(_small(img, SAL_EDGE)), "saliency"
        if found is not None:
            rows.append({"f": f, "box": found[0], "kind": kind, "conf": round(float(found[1]), 3)})
    return rows
