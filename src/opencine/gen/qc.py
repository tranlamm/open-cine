"""Cheap CPU checks of generated stills and clips (numpy + OpenCV + ffmpeg; no model).

Each check returns ``{"flags": [...], "metrics": {...}}`` (clips also ``trim_head``, frames to
drop at the start). Flags are heuristic hints on 160-px frames, not verdicts.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from opencine.core.config import get_settings
from opencine.core.ffmpeg import FFmpegError, probe_json
from opencine.core.proc import NO_WINDOW

QC_WIDTH = 160
BLACK_LUMA, BLACK_STD = 12.0, 6.0
FREEZE_DIFF, FREEZE_MIN_S = 0.2, 0.5  # mean |Δ| on 0–255 grey; shortest frozen run flagged
CUT_DIFF, CUT_CORR = 20.0, 0.7
FLICKER_STEP = 4.0  # luma jump that reverses next frame
HEAD_FRAMES, HEAD_DE, HEAD_MAX_TRIM = 6, 4.0, 6
STATIC_FLOW = 0.05  # px/frame at QC_WIDTH
DRIFT_DE, SHIFT_DE, SOFT_RATIO, MISMATCH_SSIM = 12.0, 10.0, 0.5, 0.4
# noise check: pixel correlation GARBAGE_LAG px apart / between consecutive frames; frames
# that do not correlate with the next at all are noise whatever their texture
GARBAGE_LAG, GARBAGE_SPATIAL, GARBAGE_TEMPORAL, GARBAGE_INCOHERENT = 4, 0.3, 0.8, 0.2


# ---- decoding ---------------------------------------------------------------
def video_stream(path: Path) -> dict[str, Any]:
    """The first video stream of ``path`` plus ``has_audio`` and ``duration``."""
    info = probe_json(path)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"{path.name}: no video stream")
    v = dict(v)
    v["has_audio"] = any(s.get("codec_type") == "audio" for s in streams)
    v["duration"] = float(v.get("duration") or info.get("format", {}).get("duration") or 0)
    return v


def matrix_of(stream: dict[str, Any]) -> tuple[str, str]:
    """``(in_color_matrix, in_range)`` for ffmpeg's scale: tagged BT.709 / BT.2020 as such,
    anything else (untagged, bt470bg, smpte170m) as BT.601 — what ffmpeg itself assumes."""
    cs = (stream.get("color_space") or "").lower()
    matrix = "bt709" if cs == "bt709" else "bt2020" if cs.startswith("bt2020") else "bt601"
    full = (stream.get("color_range") == "pc") or (stream.get("pix_fmt") or "").startswith("yuvj")
    return matrix, "full" if full else "tv"


def fps_of(stream: dict[str, Any]) -> Fraction:
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = stream.get(key) or "0/0"
        n, _, d = v.partition("/")
        if d and int(d) and int(n):
            return Fraction(int(n), int(d))
    return Fraction(24)


def read_frames(path: Path, width: int = QC_WIDTH, max_frames: int | None = None,
                stream: dict[str, Any] | None = None) -> np.ndarray:
    """``(N, h, width, 3)`` uint8 RGB frames of ``path``, decoded with its colour matrix."""
    v = stream or video_stream(path)
    w0, h0 = int(v["width"]), int(v["height"])
    height = max(2, round(width * h0 / w0 / 2) * 2)
    matrix, rng = matrix_of(v)
    cmd = [get_settings().ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-vf", f"scale={width}:{height}:in_color_matrix={matrix}:in_range={rng}:flags=area,"
                   "format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-"]
    proc = subprocess.run(cmd, capture_output=True, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
    frame = width * height * 3
    n = len(proc.stdout) // frame
    return np.frombuffer(proc.stdout[: n * frame], np.uint8).reshape(n, height, width, 3)


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """An image as RGB uint8, EXIF rotation applied; ``size`` = cover-fit to ``(w, h)``."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if size is not None:
            im = ImageOps.fit(im, size, method=Image.Resampling.LANCZOS)
        return np.asarray(im)


# ---- measures ---------------------------------------------------------------
def lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (L 0–100) of uint8 RGB pixels, any leading shape."""
    flat = rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_RGB2Lab).reshape(*rgb.shape[:-1], 3)


def lab_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Lab mean and std over all pixels of ``rgb`` (image or frame stack)."""
    px = lab(rgb).reshape(-1, 3)
    return px.mean(axis=0), px.std(axis=0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """CIE76 ΔE between two Lab colours."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def grey(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004, Gaussian 11×11 σ 1.5) of two same-size grey images."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(m.mean())


def sharpness(g: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _resize_long(rgb: np.ndarray, edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = edge / max(h, w)
    if k >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))),
                      interpolation=cv2.INTER_AREA)


def _r(x: float, n: int = 3) -> float:
    return round(float(x), n)


# ---- stills -------------------------------------------------------------------
def qc_still(still: Path, ref: Path | None = None) -> dict[str, Any]:
    """Flags: ``blank`` (flat image), ``garbage`` (noise), ``soft`` (much blurrier than the
    ref), ``colour_shift`` (mean colour far from the ref — expected when the prompt relights or
    recolours)."""
    img = _resize_long(load_rgb(still), 512)
    g = grey(img)
    m: dict[str, Any] = {"std": _r(g.std(), 1), "sharpness": _r(sharpness(g), 1)}
    flags = []
    m["spatial_corr"] = _r(spatial_corr(g))
    if g.std() < 4:
        flags.append("blank")
    if m["spatial_corr"] < GARBAGE_SPATIAL:
        flags.append("garbage")
    if ref is not None:
        r = load_rgb(ref, (img.shape[1], img.shape[0]))
        gr = grey(r)
        m["sharpness_ratio"] = _r(m["sharpness"] / max(sharpness(gr), 1e-6), 2)
        m["colour_de"] = _r(delta_e(lab_stats(img)[0], lab_stats(r)[0]), 1)
        small = lambda x: cv2.resize(x, (256, round(256 * x.shape[0] / x.shape[1])))  # noqa: E731
        m["ssim_ref"] = _r(ssim(small(g), small(gr)))
        if m["sharpness_ratio"] < SOFT_RATIO:
            flags.append("soft")
        if m["colour_de"] > SHIFT_DE:
            flags.append("colour_shift")
    return {"flags": flags, "metrics": m}


# ---- clips --------------------------------------------------------------------
def _runs(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _head_trim(lab_means: np.ndarray) -> tuple[int, list[float]]:
    """Frames to drop at the start: the head frames (≤ ``HEAD_FRAMES``) whose mean colour is
    off the settled clip (median of the next frames) by more than ``HEAD_DE``."""
    n = len(lab_means)
    if n < HEAD_FRAMES + 4:
        return 0, []
    settled = np.median(lab_means[HEAD_FRAMES:min(n, HEAD_FRAMES + 12)], axis=0)
    des = [delta_e(lab_means[i], settled) for i in range(HEAD_FRAMES)]
    off = [i for i, d in enumerate(des) if d > HEAD_DE]
    return (min(off[-1] + 1, HEAD_MAX_TRIM) if off else 0), [_r(d, 1) for d in des]


def spatial_corr(g: np.ndarray, lag: int = GARBAGE_LAG) -> float:
    """Correlation of pixels ``lag`` apart (horizontally and vertically) in a grey frame at
    QC size. Camera footage measures 0.6–0.85 at lag 4 (160 px wide); noise, including the
    blobby noise a VAE decodes from a failed denoise, measures ~0."""
    if min(g.shape) <= lag:
        return 1.0
    a = np.concatenate([g[:, :-lag].ravel(), g[:-lag, :].ravel()])
    b = np.concatenate([g[:, lag:].ravel(), g[lag:, :].ravel()])
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 1.0  # flat: not noise (``blank`` / ``black`` cover it)
    return float(np.corrcoef(a, b)[0, 1])


def frame_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pixelwise correlation of two grey frames (temporal coherence)."""
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 1.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def qc_clip(clip: Path, still: Path | None = None) -> dict[str, Any]:
    """Flags: ``garbage`` (noise, no spatial or temporal structure: a failed generation),
    ``flash_head`` (sets ``trim_head``), then on the frames after that trim: ``black``,
    ``frozen``, ``cut`` (hard cut inside the clip), ``flicker``, ``first_frame_mismatch``
    (the first kept frame is not the still), ``static`` (almost no motion), ``colour_drift``
    (first vs last second)."""
    v = video_stream(clip)
    fps = fps_of(v)
    frames = read_frames(clip, stream=v)
    if len(frames) == 0:
        return {"flags": ["unreadable"], "metrics": {"frames": 0}, "trim_head": 0}
    flags: list[str] = []
    m: dict[str, Any] = {"frames": len(frames), "fps": str(fps)}
    greys = np.stack([grey(f) for f in frames])
    labs = np.stack([lab_stats(f)[0] for f in frames])

    sample = range(0, len(frames), max(1, len(frames) // 8))
    m["spatial_corr"] = _r(np.median([spatial_corr(greys[i]) for i in sample]))
    pairs = [frame_corr(greys[i], greys[i + 1]) for i in sample if i + 1 < len(frames)]
    m["temporal_corr"] = _r(np.median(pairs)) if pairs else 1.0
    if m["temporal_corr"] < GARBAGE_INCOHERENT or (
            m["spatial_corr"] < GARBAGE_SPATIAL and m["temporal_corr"] < GARBAGE_TEMPORAL):
        return {"flags": ["garbage"], "metrics": m, "trim_head": 0}

    # post-process trims the flash head; judge only the frames the edit keeps
    trim, head_de = _head_trim(labs)
    m["head_de"] = head_de
    if trim:
        flags.append("flash_head")
    frames, greys, labs = frames[trim:], greys[trim:], labs[trim:]
    n = len(frames)
    luma = greys.mean(axis=(1, 2))
    std = greys.std(axis=(1, 2))
    diff = np.abs(np.diff(greys, axis=0)).mean(axis=(1, 2)) if n > 1 else np.zeros(0)

    black = int(((luma < BLACK_LUMA) & (std < BLACK_STD)).sum())
    m["black_frames"] = black
    if black:
        flags.append("black")

    frozen = _runs(diff < FREEZE_DIFF)
    m["longest_freeze_s"] = _r(frozen / float(fps), 2)
    if frozen and frozen / float(fps) >= FREEZE_MIN_S:
        flags.append("frozen")

    cuts = []
    if len(diff):
        limit = max(CUT_DIFF, 5 * float(np.median(diff)) + 5)
        for i in np.flatnonzero(diff > limit):
            h1 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            h2 = cv2.calcHist([frames[i + 1]], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            if cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < CUT_CORR:
                cuts.append(_r((trim + i + 1) / float(fps), 2))  # time in the generated file
    m["cuts_s"] = cuts
    if cuts:
        flags.append("cut")

    steps = np.diff(luma)
    flicker = int(sum(1 for a, b in zip(steps, steps[1:], strict=False)
                      if abs(a) > FLICKER_STEP and abs(b) > FLICKER_STEP and a * b < 0))
    m["flicker_events"] = flicker
    if flicker >= 2:
        flags.append("flicker")

    if still is not None:
        h, w = frames.shape[1:3]
        s = load_rgb(still, (w, h))
        m["first_ssim"] = _r(ssim(greys[0], grey(s)))
        m["first_de"] = _r(delta_e(labs[0], lab_stats(s)[0]), 1)
        if m["first_ssim"] < MISMATCH_SSIM:
            flags.append("first_frame_mismatch")

    if n > 2:
        idx = range(0, n - 1, max(1, (n - 1) // 12))
        mags = []
        for i in idx:
            flow = cv2.calcOpticalFlowFarneback(greys[i].astype(np.uint8),
                                                greys[i + 1].astype(np.uint8),
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.linalg.norm(flow, axis=2).mean()))
        m["motion_px"] = _r(np.mean(mags))
        if m["motion_px"] < STATIC_FLOW:
            flags.append("static")

    k = max(1, min(n // 4, round(float(fps))))
    m["drift_de"] = _r(delta_e(labs[:k].mean(axis=0), labs[-k:].mean(axis=0)), 1)
    if m["drift_de"] > DRIFT_DE:
        flags.append("colour_drift")
    return {"flags": flags, "metrics": m, "trim_head": trim}
