"""Mask wipes: something passing in front of the camera at the end of shot A reveals shot B
through its own shape.

The last ``d`` frames of A are rendered as seen (crop, zoom, colour), the passing object is
segmented in each, and the masks become a matte: accumulated (once revealed, B stays), dilated,
feathered and forced to full B over the last ``RAMP`` frames. Segmentation is SAM 2.1
Hiera-tiny (ONNX, ``mask`` extra; pinned zip fetched once into ``OPENCINE_MODELS_DIR``),
re-prompted each frame with the previous mask's centre and box, else frame differencing. If
neither covers ``MIN_COVER`` of the frame, the cut becomes a ``whip`` (noted).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from opencine.core.config import get_settings
from opencine.core.ffmpeg import run

if TYPE_CHECKING:
    from opencine.core.project import Project

VERSION = 1
WIDTH = 512  # analysis width
MIN_COVER = 0.3
RAMP = 3
DIFF_THRESHOLD = 25  # grey levels
MODEL = {"url": "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main/"
                "sam2.1_hiera_tiny_20260221.zip",
         "size": 116507723,
         "sha256": "c602cb3f6fd297312a415f885f0df3f0eef9fbb334b069c9d30e828f2ae7c69a",
         "dir": "sam2.1_hiera_tiny"}
SAM_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---- SAM 2.1 ---------------------------------------------------------------------------
def model_files() -> tuple[Path, Path] | None:
    """``(encoder, decoder)`` ONNX files, fetched (sha-pinned) on first use; None when off,
    ``onnxruntime`` is not installed, or the download fails."""
    s = get_settings()
    if not s.mask_model:
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    d = s.models_dir / MODEL["dir"]
    found = _onnx_pair(d)
    if found:
        return found
    try:
        import urllib.request

        d.mkdir(parents=True, exist_ok=True)
        z = d / Path(MODEL["url"]).name
        tmp = z.with_suffix(".part")
        with urllib.request.urlopen(MODEL["url"]) as r, tmp.open("wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MODEL["sha256"]:
            tmp.unlink()
            return None
        tmp.replace(z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        z.unlink()
    except OSError:
        return None
    return _onnx_pair(d)


def _onnx_pair(d: Path) -> tuple[Path, Path] | None:
    files = sorted(d.glob("**/*.onnx")) if d.exists() else []
    enc = next((f for f in files if "encoder" in f.name), None)
    dec = next((f for f in files if "decoder" in f.name), None)
    return (enc, dec) if enc and dec else None


class Sam:
    """Image SAM 2.1 through onnxruntime: ``segment(rgb, point, box)`` → bool mask. Inputs are
    matched by name, so encoder/decoder exports that differ in extras (``orig_im_size``) work."""

    def __init__(self, encoder: Any, decoder: Any):
        self.enc, self.dec = encoder, decoder

    @classmethod
    def load(cls) -> Sam | None:
        files = model_files()
        if files is None:
            return None
        from opencine.vision.runtime import session

        return cls(session(files[0]), session(files[1]))

    def segment(self, rgb: np.ndarray, point: tuple[float, float],
                box: tuple[float, float, float, float] | None = None) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = SAM_SIZE / max(h, w)
        nh, nw = round(h * s), round(w * s)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        pad = np.zeros((SAM_SIZE, SAM_SIZE, 3), np.float32)
        pad[:nh, :nw] = (x - MEAN) / STD
        feats = self.enc.run(None, {self.enc.get_inputs()[0].name: pad.transpose(2, 0, 1)[None]})
        names = [o.name for o in self.enc.get_outputs()]
        by_name = dict(zip(names, feats, strict=True))
        coords = [[point[0] * s, point[1] * s]]
        labels = [1.0]
        if box is not None:
            coords += [[box[0] * s, box[1] * s], [box[2] * s, box[3] * s]]
            labels += [2.0, 3.0]
        feed: dict[str, np.ndarray] = {}
        for i in self.dec.get_inputs():
            n = i.name
            if n in by_name:
                feed[n] = by_name[n]
            elif "embed" in n:
                feed[n] = next(v for k, v in by_name.items() if "embed" in k)
            elif n == "point_coords":
                feed[n] = np.array([coords], np.float32)
            elif n == "point_labels":
                feed[n] = np.array([labels], np.float32)
            elif n == "mask_input":
                feed[n] = np.zeros((1, 1, SAM_SIZE // 4, SAM_SIZE // 4), np.float32)
            elif n == "has_mask_input":
                feed[n] = np.zeros(1, np.float32)
            elif n == "orig_im_size":
                feed[n] = np.array([h, w], np.int64 if "int64" in i.type else np.int32)
        outs = dict(zip([o.name for o in self.dec.get_outputs()], self.dec.run(None, feed), strict=True))
        masks = next(v for v in outs.values() if v.ndim == 4)[0]
        iou = next((v for k, v in outs.items() if "iou" in k), None)
        best = masks[int(np.argmax(iou[0]))] if iou is not None and len(masks) > 1 else masks[0]
        mh, mw = best.shape
        if (mh, mw) != (h, w):  # model space (padded square): keep the image part, then resize
            if mh == mw:
                best = best[:round(nh * mh / SAM_SIZE), :round(nw * mw / SAM_SIZE)]
            best = cv2.resize(best.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return best > 0


def sam_masks(sam: Sam, frames: list[np.ndarray], point: tuple[float, float]) -> list[np.ndarray]:
    """Track by re-prompting: the previous mask's centre and box prompt the next frame."""
    out: list[np.ndarray] = []
    pt, box = point, None
    for f in frames:
        m = sam.segment(f, pt, box)
        if not m.any() and out:
            m = out[-1]
        out.append(m)
        ys, xs = np.nonzero(m)
        if len(xs):
            pt = (float(xs.mean()), float(ys.mean()))
            pad = 0.1 * max(np.ptp(xs), np.ptp(ys))
            h, w = m.shape
            box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
                   min(w - 1, xs.max() + pad), min(h - 1, ys.max() + pad))
    return out


# ---- motion fallback ---------------------------------------------------------------------
def motion_masks(frames: list[np.ndarray], point: tuple[float, float] | None) -> list[np.ndarray]:
    """What changed since the first frame of the window: the blob at ``point`` (else the largest)."""
    grey = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0) for f in frames]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = []
    for g in grey:
        diff = (cv2.absdiff(g, grey[0]) > DIFF_THRESHOLD).astype(np.uint8)
        diff = cv2.morphologyEx(cv2.morphologyEx(diff, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(diff)
        if n <= 1:
            out.append(np.zeros_like(diff, bool))
            continue
        if point is not None:
            x, y = int(point[0]), int(point[1])
            lab = labels[min(y, diff.shape[0] - 1), min(x, diff.shape[1] - 1)]
            if lab == 0:
                lab = 1 + int(np.argmin(np.hypot(cents[1:, 0] - x, cents[1:, 1] - y)))
        else:
            lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out.append(labels == lab)
    return out


def auto_point(frames: list[np.ndarray]) -> tuple[float, float] | None:
    """Centre of the largest moving blob in the middle of the window."""
    masks = motion_masks(frames, None)
    mid = masks[len(masks) // 2] if masks[len(masks) // 2].any() else masks[-1]
    ys, xs = np.nonzero(mid)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


# ---- the matte -----------------------------------------------------------------------------
def build_matte(masks: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Accumulated, dilated, feathered, ramped to 1 at the end → ``(alpha 0–1 per frame,
    coverage before the ramp)``."""
    acc = np.zeros(masks[0].shape, np.float32)
    h, w = acc.shape
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 100) | 1,) * 2)
    sigma = max(1.0, 0.01 * w)
    out = []
    d = len(masks)
    for t, m in enumerate(masks):
        acc = np.maximum(acc, cv2.dilate(m.astype(np.float32), grow))
        a = cv2.GaussianBlur(acc, (0, 0), sigma)
        if t >= d - RAMP:
            a = np.maximum(a, (t - (d - RAMP - 1)) / RAMP)
        out.append(np.clip(a, 0, 1))
    return out, float(acc.mean())


def frames_of(video: Path, first: int, n: int, width: int = WIDTH) -> list[np.ndarray]:
    """RGB frames ``first … first+n-1`` of a rendered (CFR, BT.709) video."""
    from opencine.gen.qc import read_frames

    return list(read_frames(video, width=width, max_frames=first + n)[first:first + n])


def write_matte(alpha: list[np.ndarray], size: tuple[int, int], rate, out: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        for t, a in enumerate(alpha):
            img = cv2.resize((a * 255).round().astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(Path(tmp) / f"{t:05d}.png"), img)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".tmp.mkv")
        run(["-y", "-loglevel", "error", "-framerate", str(rate), "-i", str(Path(tmp) / "%05d.png"),
             "-pix_fmt", "gray", "-c:v", "ffv1", str(part)])
        part.replace(out)
    return out


def make(video: Path, n_a: int, d: int, point: tuple[float, float] | None, size: tuple[int, int],
         rate, out: Path, sam: Sam | None = None) -> dict[str, Any]:
    """Matte for the last ``d`` frames of ``video`` (A rendered alone, ``n_a`` frames) →
    ``{path | None, by, cover}``. ``point``: the object on A's frame (0–1); None = the largest
    moving blob."""
    frames = frames_of(video, n_a - d, d)
    h, w = frames[0].shape[:2]
    pt = (point[0] * w, point[1] * h) if point is not None else auto_point(frames)
    by, masks = "motion", None
    if sam is not None and pt is not None:
        try:
            masks, by = sam_masks(sam, frames, pt), "sam2.1-tiny"
        except Exception:  # noqa: BLE001 — a model problem falls back, never fails the render
            masks = None
    if masks is None:
        masks = motion_masks(frames, pt)
    alpha, cover = build_matte(masks)
    if cover < MIN_COVER:
        return {"path": None, "by": by, "cover": round(cover, 2)}
    return {"path": write_matte(alpha, size, rate, out), "by": by, "cover": round(cover, 2)}


def prepare(project: Project, plan, decoded: list, size: tuple[int, int], enc, font: str,
            final: bool, luts: list | None, cache) -> tuple[dict[int, Path | None], list[str]]:
    """Mattes for every ``mask`` transition of ``plan`` → ``({item index: matte or None}, notes)``."""
    from opencine.core.cache import cache_key
    from opencine.plan.schema import EditPlan
    from opencine.render.encode import encode_segment
    from opencine.render.segments import build, finalize_graph

    found: dict[int, Path | None] = {}
    notes: list[str] = []
    sam: Sam | None | bool = False  # loaded on first need
    for k, it in enumerate(plan.items):
        if k == 0 or it.transition_in.kind != "mask" or it.is_cut:
            continue
        a = plan.items[k - 1].model_copy(update={"transition_in": type(it.transition_in)(), "text": None})
        solo = EditPlan.model_validate({"rate": plan.rate, "deliverable": plan.deliverable,
                                        "items": [a.model_dump(by_alias=True)]})
        seg = build(solo, [0], [decoded[k - 1]], 0, size, enc.args, font, final=final,
                    luts=[luts[k - 1]] if luts else None)
        video = cache.path("segment", seg.key, ".mp4")
        if cache.hit("segment", seg.key, ".mp4") is None:
            encode_segment(seg.inputs, finalize_graph(seg, None), seg.out, seg.frames, enc, video)
        d, pt = it.transition_in.frames, it.transition_in.point
        key = cache_key("matte", seg.key, d, pt, size, VERSION, get_settings().mask_model)
        out, meta = cache.path("matte", key, ".mkv"), cache.path("matte", key, ".json")
        if meta.exists():
            info = json.loads(meta.read_text("utf-8"))
        else:
            if sam is False:
                sam = Sam.load()
            info = make(video, seg.frames, d, pt, size, plan.fps, out, sam or None)
            info["path"] = str(info["path"]) if info["path"] else None
            meta.write_text(json.dumps(info), "utf-8")
        if info.get("path"):
            found[k] = Path(info["path"])
            notes.append(f"#{k + 1}: mask wipe ({info.get('by')}, cover {info.get('cover')})")
        else:
            found[k] = None
            notes.append(f"#{k + 1}: mask wipe found too little to wipe with "
                         f"(cover {info.get('cover')}); rendered as a whip")
    return found, notes

