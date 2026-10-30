"""Track what a scribble is drawn on through the rendered video (output-frame fractions).

- ``follow``: Lucas–Kanade flow of corners around a point; median shift of corners passing a
  forward–backward check (robust to captions, zooms, outliers); keys ``{t, x, y}`` every ``STEP``.
- ``masks``: EdgeTAM from a box/point on the first frame, ``MASK_FPS`` (``TRAIL_FPS`` for fast
  trails); ``sam_masks``: SAM 2.1 tiny re-prompted per frame when EdgeTAM is missing.
- From masks: ``contours`` (offset outer contour, ``POINTS`` points, aligned key to key so the
  renderer can blend them), ``centroids`` (a trail's path), ``boxes`` (a doodle's anchor/scale).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from opencine.core.config import get_settings
from opencine.core.ffmpeg import FFmpegError
from opencine.core.proc import NO_WINDOW

if TYPE_CHECKING:
    from opencine.render.matte import Sam
    from opencine.vision.tam import Tam

WIDTH = 360  # analysis width (tracking)
SAM_WIDTH = 512
TAM_WIDTH = 576  # EdgeTAM and Florence frames (both squash to a square)
STEP = 2  # frames between tracked keys
AROUND = 0.1  # corner search radius around the point (fraction of the width)
CORNERS, MIN_CORNERS = 40, 6
FB_PX = 1.0  # max forward–backward error of a kept corner (px)
LK = {"winSize": (21, 21), "maxLevel": 3,
      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)}
MASK_FPS = 10
TRAIL_FPS = 15
POINTS = 72
MIN_AREA = 0.002  # share of the frame; smaller blobs are noise
VERSION = 4  # cache version: bump when tracking output changes

Found = list[tuple[float, np.ndarray]]


def frames(video: Path, first: int, n: int, rate: float, width: int = WIDTH) -> np.ndarray:
    """``(n', h, width, 3)`` RGB frames ``first … first+n-1`` of a rendered (CFR, BT.709) video."""
    from opencine.gen.qc import video_stream

    v = video_stream(video)
    h = max(2, round(width * int(v["height"]) / int(v["width"]) / 2) * 2)
    cmd = [get_settings().ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
           "-ss", f"{max(0.0, (first - 0.5) / rate):.6f}", "-i", str(video), "-frames:v", str(n),
           "-vf", f"scale={width}:{h}:in_color_matrix=bt709:in_range=tv:flags=area,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-"]
    proc = subprocess.run(cmd, capture_output=True, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
    size = width * h * 3
    k = len(proc.stdout) // size
    return np.frombuffer(proc.stdout[:k * size], np.uint8).reshape(k, h, width, 3)


def _smooth(vals: list[float]) -> list[float]:
    return [float(np.mean(vals[max(0, i - 1):i + 2])) for i in range(len(vals))]


def _corners(img: np.ndarray, x: float, y: float) -> np.ndarray:
    h, w = img.shape
    r = AROUND * w
    mask = np.zeros_like(img)
    mask[int(max(0, y - r)):int(min(h, y + r)), int(max(0, x - r)):int(min(w, x + r))] = 255
    found = cv2.goodFeaturesToTrack(img, CORNERS, 0.01, 4, mask=mask)
    return found if found is not None else np.empty((0, 1, 2), np.float32)


def follow(rgb: np.ndarray, point: tuple[float, float], rate: float, step: int = STEP) -> list[dict]:
    """Keys ``{t, x, y}`` (seconds from the first frame; fractions) of the point at ``point``."""
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    if not grey:
        return []
    h, w = grey[0].shape
    cx, cy = point[0] * w, point[1] * h
    pts = _corners(grey[0], cx, cy)
    xs, ys = [cx], [cy]
    for prev, img in zip(grey, grey[1:], strict=False):
        if len(pts) < MIN_CORNERS:
            pts = _corners(prev, cx, cy)
        if len(pts):
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, None, **LK)
            back, ok2, _ = cv2.calcOpticalFlowPyrLK(img, prev, nxt, None, **LK)
            good = (ok.ravel() == 1) & (ok2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < FB_PX)
            if good.sum() >= 3:
                dx, dy = np.median((nxt - pts).reshape(-1, 2)[good], axis=0)
                cx, cy = min(w - 1.0, max(0.0, cx + float(dx))), min(h - 1.0, max(0.0, cy + float(dy)))
                pts = nxt[good].reshape(-1, 1, 2)
            else:
                pts = pts[:0]
        xs.append(cx)
        ys.append(cy)
    xs, ys = _smooth(xs), _smooth(ys)
    return [{"t": round(k / rate, 3), "x": round(xs[k] / w, 4), "y": round(ys[k] / h, 4)}
            for k in range(0, len(xs), step)]


# ---- masks ---------------------------------------------------------------------------------
def masks(tam: Tam, rgb: np.ndarray, rate: float, box: tuple[float, float, float, float] | None = None,
          point: tuple[float, float] | None = None, fps: float = MASK_FPS) -> Found:
    """``(t, mask)`` at ``fps`` for the thing at ``box`` / ``point`` on the first frame (empty
    masks where EdgeTAM says it is not visible)."""
    step = max(1, round(rate / fps))
    found = tam.propagate(list(rgb[::step]), box=box, point=point)
    return [(round(k * step / rate, 3), m if s > 0 else np.zeros_like(m)) for k, (m, s) in enumerate(found)]


def sam_masks(sam: Sam, rgb: np.ndarray, rate: float, point: tuple[float, float], fps: float = MASK_FPS) -> Found:
    """Like ``masks`` with SAM 2.1 re-prompted frame to frame from ``point`` (drifts sooner)."""
    from opencine.render.matte import sam_masks as reprompt

    step = max(1, round(rate / fps))
    picked = list(rgb[::step])
    if not picked:
        return []
    h, w = picked[0].shape[:2]
    return [(round(k * step / rate, 3), m) for k, m in enumerate(reprompt(sam, picked, (point[0] * w, point[1] * h)))]


def save_masks(path: Path, found: Found) -> None:
    ts = np.array([t for t, _ in found], np.float64)
    ms = np.stack([m.astype(bool) for _, m in found]) if found else np.zeros((0, 1, 1), bool)
    with path.open("wb") as f:
        np.savez_compressed(f, t=ts, m=np.packbits(ms, axis=-1), shape=np.array(ms.shape))


def load_masks(path: Path) -> Found:
    with np.load(path) as z:
        n, h, w = (int(v) for v in z["shape"])
        ms = np.unpackbits(z["m"], axis=-1, count=w).astype(bool).reshape(n, h, w)
        return [(float(t), m) for t, m in zip(z["t"], ms, strict=True)]


# ---- what the masks give -------------------------------------------------------------------
def contour(mask: np.ndarray, offset: float) -> np.ndarray | None:
    """Outer contour of ``mask`` grown by ``offset`` (fraction of its width): ``(POINTS, 2)``
    evenly spaced pixel points, lightly smoothed, starting at the top."""
    h, w = mask.shape
    m = mask.astype(np.uint8)
    if (grow := round(offset * w)) > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return None
    c = max(found, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    area = cv2.contourArea(c.astype(np.float32), oriented=True)
    if len(c) < 8 or abs(area) < MIN_AREA * w * h:
        return None
    if area < 0:  # same winding for every key, so keys can be aligned
        c = c[::-1]
    seg = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T)
    pos = np.concatenate([[0], np.cumsum(seg)])
    at = np.linspace(0, pos[-1], POINTS, endpoint=False)
    ring = np.vstack([c, c[:1]])
    pts = np.stack([np.interp(at, pos, ring[:, 0]), np.interp(at, pos, ring[:, 1])], axis=1)
    for _ in range(2):  # soften pixel steps and mask noise
        pts = (np.roll(pts, 1, 0) + 2 * pts + np.roll(pts, -1, 0)) / 4
    return np.roll(pts, -int(np.argmin(pts[:, 1])), 0)  # start at the top, as a hand would


def align(prev: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``pts`` cyclically shifted to best match ``prev`` point by point, so blending keys moves
    each point a short way instead of sliding along the line."""
    cost = [float(((np.roll(pts, -s, 0) - prev) ** 2).sum()) for s in range(len(pts))]
    return np.roll(pts, -int(np.argmin(cost)), 0)


def contours(found: Found, offset: float) -> list[dict]:
    """Keys ``{t, pts}`` (fractions) of the masks' contours, aligned and smoothed over time;
    no key where the thing is lost."""
    h, w = found[0][1].shape if found else (1, 1)
    ts, rings = [], []
    for t, m in found:
        pts = contour(m, offset)
        if pts is None:
            continue
        rings.append(align(rings[-1], pts) if rings else pts)
        ts.append(t)
    if len(rings) > 2:  # light [1 2 1] over time; the ends stay
        a = np.stack(rings)
        a[1:-1] = (a[:-2] + 2 * a[1:-1] + a[2:]) / 4
        rings = list(a)
    return [{"t": t, "pts": [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in r]}
            for t, r in zip(ts, rings, strict=True)]


def centroids(found: Found) -> list[dict]:
    """Keys ``{t, x, y}`` of the mask centres (fractions), lightly smoothed; none where lost."""
    keys = []
    for t, m in found:
        ys, xs = np.nonzero(m)
        if len(xs) >= MIN_AREA * m.size:
            keys.append({"t": t, "x": float(xs.mean() + 0.5) / m.shape[1], "y": float(ys.mean() + 0.5) / m.shape[0]})
    for k in ("x", "y"):
        for key, v in zip(keys, _smooth([key[k] for key in keys]), strict=True):
            key[k] = round(v, 4)
    return keys


def boxes(found: Found) -> list[dict]:
    """Keys ``{t, x0, y0, x1, y1}`` (fractions) of the mask boxes, lightly smoothed; a lost frame
    holds the last box (dropped before the first)."""
    keys: list[dict] = []
    for t, m in found:
        ys, xs = np.nonzero(m)
        if len(xs):
            h, w = m.shape
            b = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
        elif keys:
            b = [keys[-1][k] for k in ("x0", "y0", "x1", "y1")]
        else:
            continue
        keys.append({"t": t, **dict(zip(("x0", "y0", "x1", "y1"), map(float, b), strict=True))})
    for k in ("x0", "y0", "x1", "y1"):
        for key, v in zip(keys, _smooth([key[k] for key in keys]), strict=True):
            key[k] = round(v, 4)
    return keys
