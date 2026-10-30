"""Face pass, CPU side: track one face through a finished clip, cut a square crop that follows it
for the Space (``HfSpaceProvider.face``), and paste the redrawn crop back.

Recipe after ``Carasibana/ComfyUI-H3-FaceRefine`` (MIT); face detection is YuNet (OpenCV, MIT).
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from opencine.core.config import get_settings
from opencine.core.ffmpeg import FFmpegError
from opencine.core.proc import NO_WINDOW
from opencine.gen.base import ProviderError

SHORT = 768  # face heights are counted at H3's native short side
LINK, SIZE_RATIO, GAP = 0.8, (0.6, 1.65), 12
SMOOTH_CENTRE, SMOOTH_SIZE, SMOOTH_STRENGTH = 21, 51, 9  # Gaussian windows (frames), ±3σ
FACE_SMALL, FACE_LARGE, FACTOR_LARGE = 30.0, 120.0, 0.35
MIN_PRESENT = 0.3  # a face in fewer of the frames: nothing to refine
FADE = 6
MASK_W, MASK_H, FEATHER = 0.72, 0.82, 0.05  # ellipse half-axes / blur sigma, of the crop's face box
SCORE = 0.6
VERSION = 1


@dataclass
class Face:
    """One face: its box (pixels, top-left) and score."""
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


Detector = Callable[[np.ndarray], list[Face]]


def yunet() -> Detector:
    """YuNet on an RGB frame: at its size, and 2× up when nothing shows (faces under ~20 px)."""
    from opencine.vision import models

    d = models.fetch("yunet")
    if d is None:
        raise ProviderError("no face detector: YuNet could not be fetched (OPENCINE_VISION_MODELS on, "
                            "network up?)")
    net = cv2.FaceDetectorYN.create(str(d / models.MODELS["yunet"]["files"][0]), "", (320, 320),
                                    SCORE, 0.3, 50)

    def run(rgb: np.ndarray, k: float) -> list[Face]:
        img = rgb if k == 1 else cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        net.setInputSize((img.shape[1], img.shape[0]))
        _, found = net.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return [Face(*(float(v) / k for v in f[:4]), float(f[-1])) for f in (found if found is not None else [])]

    def detect(rgb: np.ndarray) -> list[Face]:
        return run(rgb, 1.0) or run(rgb, 2.0)

    return detect


def _link(a: Face, b: Face) -> bool:
    size = max(a.h, b.h)
    return (math.hypot(a.cx - b.cx, a.cy - b.cy) < LINK * size
            and SIZE_RATIO[0] <= b.h / max(a.h, 1e-6) <= SIZE_RATIO[1])


def subject_track(found: list[list[Face]]) -> list[Face | None]:
    """The subject's face per frame from every frame's detections: tracks chained greedily (largest
    face first), the one with the most face kept; gaps ≤ ``GAP`` frames filled linearly."""
    tracks: list[dict[int, Face]] = []
    last: list[int] = []  # each track's latest frame
    for t, faces in enumerate(found):
        taken: set[int] = set()
        for f in sorted(faces, key=lambda f: -f.h):
            near = [i for i, tr in enumerate(tracks)
                    if i not in taken and t - last[i] <= GAP and _link(tr[last[i]], f)]
            if near:
                i = min(near, key=lambda i: math.hypot(tracks[i][last[i]].cx - f.cx,
                                                       tracks[i][last[i]].cy - f.cy))
                tracks[i][t], last[i] = f, t
            else:
                tracks.append({t: f})
                last.append(t)
                i = len(tracks) - 1
            taken.add(i)
    out: list[Face | None] = [None] * len(found)
    if not tracks:
        return out
    track = max(tracks, key=lambda tr: sum(f.h * f.score for f in tr.values()))
    keys = sorted(track)
    for a, b in zip(keys, keys[1:], strict=False):
        out[a] = track[a]
        if 1 < b - a <= GAP + 1:
            fa, fb = track[a], track[b]
            for t in range(a + 1, b):
                u = (t - a) / (b - a)
                out[t] = Face(*(pa + u * (pb - pa) for pa, pb in zip(
                    (fa.x, fa.y, fa.w, fa.h), (fb.x, fb.y, fb.w, fb.h), strict=True)), min(fa.score, fb.score))
    out[keys[-1]] = track[keys[-1]]
    return out


def _segments(present: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of each run of frames with a face."""
    edges = np.flatnonzero(np.diff(np.r_[0, present.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def gauss(values: np.ndarray, window: int) -> np.ndarray:
    """``values`` Gaussian-smoothed over ``window`` frames (σ = window / 6), edges reflected."""
    sigma = max(window / 6, 1e-3)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if len(values) < 2:
        return values.astype(float)
    padded = np.pad(values.astype(float), r, mode="reflect" if len(values) > r else "edge")
    return np.convolve(padded, k, mode="valid")


@dataclass
class Plan:
    """What the face pass does on one clip (``plan``)."""
    width: int
    height: int
    n: int  # frames of the clip
    size: int  # the crop canvas
    present: np.ndarray  # (n,) bool: a face on the frame
    cx: np.ndarray  # smoothed centre, height and width of the face, pixels
    cy: np.ndarray
    h: np.ndarray
    w: np.ndarray
    side: np.ndarray  # crop side, pixels
    strength: np.ndarray  # denoise per frame (0 = kept)
    face_px: float  # median face height at H3's short side
    skip: str | None = None  # why the clip is left as it is
    info: dict[str, Any] = field(default_factory=dict)

    def warp(self, t: int) -> np.ndarray:
        """The 2×3 affine taking frame ``t`` onto its crop."""
        k = self.size / self.side[t]
        return np.array([[k, 0, self.size / 2 - k * self.cx[t]], [0, k, self.size / 2 - k * self.cy[t]]],
                        np.float64)


def strengths(face_px: np.ndarray, base: float) -> np.ndarray:
    """Denoise per frame for face heights ``face_px`` (at H3's short side): ``base`` × a factor
    from 1 (≤ ``FACE_SMALL``) down to ``FACTOR_LARGE`` (≥ ``FACE_LARGE``)."""
    u = np.clip((face_px - FACE_SMALL) / (FACE_LARGE - FACE_SMALL), 0.0, 1.0)
    return base * (1 - (1 - FACTOR_LARGE) * u)


def plan(frames: np.ndarray, detect: Detector, *, size: int, base: float, crop: float,
         max_px: float, force: bool = False) -> Plan:
    """Track the subject's face through ``frames`` (T, H, W, 3) RGB and lay out its crops.

    Centre and size are Gaussian-smoothed per run of faces; the crop is per frame (``crop`` face
    heights wide), so a push-in does not leave the face small in a crop sized for its widest frame.
    """
    n, hgt, wid = frames.shape[:3]
    track = subject_track([detect(f) for f in frames])
    present = np.array([f is not None for f in track])
    zero = np.zeros(n)
    if not present.any():
        return Plan(wid, hgt, n, size, present, zero, zero, zero, zero, zero + 1, zero, 0.0,
                    skip="no face found")
    first = next(f for f in track if f is not None)
    raw = np.array([[f.cx, f.cy, f.h, f.w] if f else [np.nan] * 4 for f in track])
    for t in range(n):  # frames off the track hold the nearest face (they are not pasted)
        if not present[t]:
            near = min(np.flatnonzero(present), key=lambda s: abs(s - t))
            raw[t] = raw[near]
    cx, cy = np.empty(n), np.empty(n)
    h, w = np.empty(n), np.empty(n)
    for a, b in _segments(present) or [(0, n)]:
        cx[a:b], cy[a:b] = gauss(raw[a:b, 0], SMOOTH_CENTRE), gauss(raw[a:b, 1], SMOOTH_CENTRE)
        h[a:b], w[a:b] = gauss(raw[a:b, 2], SMOOTH_SIZE), gauss(raw[a:b, 3], SMOOTH_SIZE)
    for t in np.flatnonzero(~present):
        cx[t], cy[t], h[t], w[t] = raw[t]
    side = np.maximum(crop * h, 32.0)
    norm = h * SHORT / min(hgt, wid)
    face_px = float(np.median(norm[present]))
    st = np.where(present, gauss(strengths(norm, base), SMOOTH_STRENGTH), 0.0)
    st = np.clip(np.where(present, st, 0.0), 0.0, 0.95)
    p = Plan(wid, hgt, n, size, present, cx, cy, h, w, side, st, face_px)
    p.info = {"present": round(float(present.mean()), 3), "face_px": round(face_px, 1),
              "face_px_range": [round(float(norm[present].min()), 1), round(float(norm[present].max()), 1)],
              "magnification": round(float(np.median(size / side[present])), 2),
              "denoise": [round(float(st[present].min()), 3), round(float(st[present].max()), 3)],
              "first_box": [round(float(v), 1) for v in (first.x, first.y, first.w, first.h)]}
    if present.mean() < MIN_PRESENT:
        p.skip = f"a face on only {present.mean():.0%} of the frames"
    elif np.median(size / side[present]) <= 1:
        p.skip = (f"the face is already larger than the {size}² crop (median {face_px:.0f} px): "
                  "nothing to gain")
    elif face_px >= max_px and not force:
        p.skip = (f"median face {face_px:.0f} px tall (≥ {max_px:g}): already sharp enough; force=True "
                  "to refine it anyway")
    return p


def crops(frames: np.ndarray, p: Plan) -> np.ndarray:
    """Every frame warped onto its crop: (T, size, size, 3)."""
    flags = cv2.INTER_LANCZOS4
    return np.stack([cv2.warpAffine(f, p.warp(t), (p.size, p.size), flags=flags,
                                    borderMode=cv2.BORDER_REFLECT_101) for t, f in enumerate(frames)])


def crop_mask(p: Plan) -> np.ndarray:
    """The paste mask in crop space (float 0–1): a feathered ellipse around the face box, sized by
    the track's median width/height ratio (the crop is ``crop`` face heights wide)."""
    face_h = p.size * float(np.median(p.h[p.present] / p.side[p.present]))
    face_w = face_h * float(np.median(p.w[p.present] / np.maximum(p.h[p.present], 1e-6)))
    m = np.zeros((p.size, p.size), np.float32)
    c = (p.size // 2, p.size // 2)
    cv2.ellipse(m, c, (max(1, int(MASK_W * face_w)), max(1, int(MASK_H * face_h))), 0, 0, 360, 1.0, -1)
    sigma = max(1.0, FEATHER * p.size)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0.0, 1.0)


def fade(present: np.ndarray, frames: int = FADE) -> np.ndarray:
    """Paste weight per frame: 1 inside a run of faces, ramping over ``frames`` where the run
    starts or stops inside the clip (a face there from the first or to the last frame: no ramp)."""
    n, w = len(present), np.zeros(len(present))
    for a, b in _segments(present):
        t = np.arange(a, b)
        up = (t - a + 1) / frames if a > 0 else np.ones(b - a)
        down = (b - t) / frames if b < n else np.ones(b - a)
        w[a:b] = np.clip(np.minimum(up, down), 0.0, 1.0)
    return w


GAIN_L, GAIN_AB = (0.9, 1.1), (0.8, 1.25)  # std ratio clamps: lightness barely (detail lives there)


def colour_transform(orig: np.ndarray, new: np.ndarray, mask: np.ndarray, every: int = 3
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(mean_new, std_new, mean_orig, std_target)`` in Lab of the masked face over every
    ``every``-th crop: one transform for the whole clip (std ratio clamped per channel)."""
    sel = mask > 0.5

    def stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab = np.concatenate([cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)[sel] for f in x[::every]])
        return lab.mean(0), lab.std(0) + 1e-3

    mn, sn = stats(new)
    mo, so = stats(orig)
    lo, hi = np.array([GAIN_L[0], *[GAIN_AB[0]] * 2]), np.array([GAIN_L[1], *[GAIN_AB[1]] * 2])
    return mn, sn, mo, np.clip(so / sn, lo, hi) * sn


def _apply_colour(crop: np.ndarray, t: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mn, sn, mo, so = t
    lab = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lab = (lab - mn) / sn * so + mo
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB) * 255, 0, 255)


def paste(frames: np.ndarray, redrawn: np.ndarray, p: Plan, orig_crops: np.ndarray | None = None
          ) -> tuple[np.ndarray, dict[str, Any]]:
    """``frames`` with the redrawn crops pasted back (only frames with a face, faded at the run
    ends), colours of the redrawn face moved onto the original's. → (frames, what was done)."""
    mask = crop_mask(p)
    weight = fade(p.present) * (p.strength > 0)
    orig_crops = crops(frames, p) if orig_crops is None else orig_crops
    tr = colour_transform(orig_crops[p.present], redrawn[: p.n][p.present], mask)
    out = frames.copy()
    for t in np.flatnonzero(weight > 0):
        inv = cv2.invertAffineTransform(p.warp(t))
        new = _apply_colour(redrawn[t], tr)
        back = cv2.warpAffine(new, inv, (p.width, p.height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)
        m = cv2.warpAffine(mask, inv, (p.width, p.height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None] * weight[t]
        out[t] = np.clip(frames[t] * (1 - m) + back * m, 0, 255).astype(np.uint8)
    shift = [round(float(v), 2) for v in (tr[2] - tr[0])]
    return out, {"pasted": int((weight > 0).sum()), "lab_shift": shift}


def self_reference(frames: np.ndarray, p: Plan, dst: Path) -> Path:
    """The clip's own clearest face as the identity picture (no character picture known): the
    largest one, cut 1.8 face heights square, at least 512 px."""
    from PIL import Image

    t = int(np.argmax(np.where(p.present, p.h, -1)))
    side = max(32.0, 1.8 * p.h[t])
    k = max(512, int(side)) / side
    m = np.array([[k, 0, side * k / 2 - k * p.cx[t]], [0, k, side * k / 2 - k * p.cy[t]]])
    out = int(side * k)
    pic = cv2.warpAffine(frames[t], m, (out, out), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT_101)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pic).save(dst)
    return dst


def soundtrack(src: Path, seconds: float, dst: Path, tempo: float = 1.0) -> Path | None:
    """``src``'s first audio stream as 32 kHz stereo 16-bit WAV of ``seconds`` (padded with
    silence), played ``tempo``× as fast (a clip at 30 fps sent as 24 fps frames: 0.8); None when
    it has none."""
    from opencine.gen.qc import video_stream

    if not video_stream(src).get("has_audio"):
        return None
    chain = []
    while tempo < 0.5 - 1e-9 or tempo > 2.0 + 1e-9:  # atempo takes 0.5-2 per stage
        step = 0.5 if tempo < 1 else 2.0
        chain.append(f"atempo={step}")
        tempo /= step
    if abs(tempo - 1) > 1e-6:
        chain.append(f"atempo={tempo:.6f}")
    _ffmpeg(["-y", "-i", str(src), "-map", "0:a:0", "-af", ",".join([*chain, "apad"]), "-t",
             f"{seconds:.4f}", "-ac", "2", "-ar", "32000", "-c:a", "pcm_s16le", str(dst)])
    return dst


def write(frames: np.ndarray, fps: Any, dst: Path, audio_from: Path | None = None, crf: int = 12) -> Path:
    """Frames (RGB uint8) as H.264 at ``fps`` (BT.709, tagged), ``audio_from``'s sound copied."""
    h, w = frames.shape[1:3]
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio = ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"] if audio_from else []
    _ffmpeg(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             *audio, "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags",
             "+faststart", str(dst)], stdin=b"".join(np.ascontiguousarray(f).tobytes() for f in frames))
    return dst


def _ffmpeg(args: list[str], stdin: bytes | None = None) -> None:
    cmd = [get_settings().ffmpeg, "-hide_banner", *(["-nostdin"] if stdin is None else []),
           "-loglevel", "error", *args]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
