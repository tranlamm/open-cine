"""Mattes for text behind a subject (``<Behind matte="me">``): per timeline frame, how much of
each pixel the subject covers (0–255).

The ``from``–``to`` window is split at cuts; per shot Florence-2 boxes ``on`` (default "person")
on the first clear frame or at ``at`` (``x``, ``y`` pick among several), and EdgeTAM propagates a
soft mask forward and backward. A guided filter was rejected (text bled through low-contrast
edges); a 3-frame median removes flicker, and the edge is eroded 1 px at output size. Computed on
the base render at ``WIDTH``, cached per plan version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from opencine.core.cache import Cache, cache_key

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.schema import EditPlan

WIDTH = 576  # EdgeTAM resizes to 1024², so wider frames add nothing
MAX_S = 15.0  # longer windows cost minutes of EdgeTAM; split the text instead
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
KEYS = {"on", "from", "to", "at", "x", "y"}
VERSION = 1


class MatteError(ValueError):
    pass


@dataclass
class Matte:
    name: str
    on: str
    first: int  # timeline frame of m[0]
    m: np.ndarray  # (n, h, WIDTH) uint8 subject coverage
    notes: list[str] = field(default_factory=list)
    key: str = ""

    @property
    def end(self) -> int:
        return self.first + len(self.m)

    def at(self, frame: int) -> np.ndarray | None:
        return self.m[frame - self.first] if self.first <= frame < self.end else None


# ---- the spec ------------------------------------------------------------------------------
def check(behind: dict[str, Any]) -> None:
    for name, spec in behind.items():
        if not NAME_RE.match(name):
            raise MatteError(f"behind: matte name {name!r}: a letter, then letters, digits, - or _")
        if not isinstance(spec, dict) or not {"from", "to"} <= spec.keys():
            raise MatteError(f"behind.{name}: give from and to (seconds or moments), e.g. "
                             '{"on": "person", "from": 0.4, "to": 2.5}')
        if extra := set(spec) - KEYS:
            raise MatteError(f"behind.{name}: unknown {sorted(extra)}; takes {sorted(KEYS)}")


def window(project: Project, plan: EditPlan, name: str, spec: dict[str, Any]) -> tuple[int, int, int | None]:
    """``(first, end, at)`` timeline frames of a spec."""
    from opencine.scribble.anchor import timeline_s

    rate = float(plan.fps)
    try:
        a, b = (round(timeline_s(project, plan, spec[k]) * rate) for k in ("from", "to"))
        at = round(timeline_s(project, plan, spec["at"]) * rate) if "at" in spec else None
    except (ValueError, KeyError) as e:
        raise MatteError(f"behind.{name}: {e}") from None
    a, b = max(0, a), min(plan.total_frames, b)
    if b <= a:
        raise MatteError(f"behind.{name}: to must come after from, inside the video")
    if (b - a) / rate > MAX_S:
        raise MatteError(f"behind.{name}: {(b - a) / rate:.1f} s; at most {MAX_S:.0f} s a matte")
    return a, b, at


def segments(plan: EditPlan, first: int, end: int) -> list[tuple[int, int]]:
    """``[first, end)`` cut at the shots it crosses."""
    from opencine.scribble.service import shot_end

    out, a = [], first
    while a < end:
        b = min(end, shot_end(plan, a))
        out.append((a, b))
        a = b
    return out


# ---- the edge ------------------------------------------------------------------------------
def smooth(ms: np.ndarray) -> np.ndarray:
    """Temporal median of 3 (ends held): removes one-frame flicker without dragging moving edges."""
    if len(ms) < 3:
        return ms
    prev, nxt = np.concatenate([ms[:1], ms[:-1]]), np.concatenate([ms[1:], ms[-1:]])
    return np.maximum(np.minimum(prev, ms), np.minimum(np.maximum(prev, ms), nxt))


# ---- compute -------------------------------------------------------------------------------
def _pick(things: list[list[float]], spec: dict[str, Any]) -> list[float]:
    if "x" in spec and "y" in spec:
        x, y = float(spec["x"]), float(spec["y"])
        return min(things, key=lambda b: ((b[0] + b[2]) / 2 - x) ** 2 + ((b[1] + b[3]) / 2 - y) ** 2)
    return max(things, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _fit(ms: np.ndarray, n: int) -> np.ndarray:
    """Exactly ``n`` frames (a short decode at a shot's end repeats its last frame)."""
    return ms[:n] if len(ms) >= n else np.concatenate([ms, np.repeat(ms[-1:], n - len(ms), axis=0)])


def _follow(tam: Any, rgb: np.ndarray, k: int, box: list[float] | None, point: tuple[float, float]
            ) -> list[np.ndarray]:
    """Soft masks of ``rgb`` from frame ``k``: forward to the end, backward to the start."""
    prompt = {"box": tuple(box) if box else None, "point": None if box else point, "soft": True}
    fwd = tam.propagate(list(rgb[k:]), **prompt)
    bwd = tam.propagate(list(rgb[:k + 1][::-1]), **prompt)[1:][::-1] if k else []
    return [m if s > 0 else np.zeros_like(m) for m, s in [*bwd, *fwd]]


def compute(project: Project, plan: EditPlan, version: int, video: Path, name: str,
            spec: dict[str, Any], vis: Any = None) -> Matte:
    """The matte of ``spec`` over its window, cached per plan version."""
    from opencine.scribble import track
    from opencine.scribble.service import _look, _Vision, shot_clear

    first, end, at = window(project, plan, name, spec)
    on = str(spec.get("on") or "person")
    point = (float(spec.get("x", 0.5)), float(spec.get("y", 0.5)))
    cache = Cache(project.cache_dir / "motion")
    key = cache_key("matte", version, first, end, at, on.lower(), spec.get("x"), spec.get("y"), VERSION)
    notes: list[str] = []
    if hit := cache.hit("matte", key, ".npz"):
        with np.load(hit) as z:
            return Matte(name, on, first, z["m"], [str(n) for n in z["notes"]], key)
    vis = vis or _Vision()
    tam = vis.get("tam")
    if tam is None:
        raise MatteError("text behind a subject needs EdgeTAM (extra `vision`, `opencine models fetch`)")
    rate = float(plan.fps)
    parts, seen = [], 0
    for a, b in segments(plan, first, end):
        look = at if at is not None and a <= at < b else shot_clear(plan, a)
        things, why = _look(project, plan, version, video, on, look, vis)
        box = _pick(things, spec) if things else None
        rgb = track.frames(video, a, b - a, rate, width=WIDTH)
        if box is None and why != "fills the frame (a close-up)":
            notes.append(f"{name}: '{on}' {why} at {look / rate:.2f} s")
            parts.append(np.zeros((b - a, *rgb.shape[1:3]), np.uint8))
            continue
        seen += 1
        soft = _follow(tam, rgb, min(look - a, len(rgb) - 1), box, point)
        parts.append(_fit(smooth(np.stack(soft)), b - a))
    if not seen:
        raise MatteError(f"behind.{name}: {notes[0].split(': ', 1)[1] if notes else 'nothing found'}")
    m = np.concatenate(parts)
    with cache.path("matte", key, ".npz").open("wb") as f:
        np.savez_compressed(f, m=m, notes=np.array(notes, dtype=str))
    return Matte(name, on, first, m, notes, key)


# ---- lay together --------------------------------------------------------------------------
def at_size(matte: np.ndarray, w: int, h: int) -> np.ndarray:
    """A matte frame resized to the output and eroded 1 px (no background fringe on the text);
    float 0–1."""
    up = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.erode(up, np.ones((3, 3), np.uint8)).astype(np.float32) / 255


def _over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Straight-alpha ``top`` over ``bottom`` (float, colour first, alpha last)."""
    ta, ba = top[..., 3:], bottom[..., 3:]
    a = ta + ba * (1 - ta)
    rgb = (top[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) / np.maximum(a, 1e-6)
    return np.concatenate([rgb, a], axis=-1)


def composite(front: np.ndarray | None, behinds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """One layer frame: each behind pass (straight RGBA/BGRA uint8) cut out by its matte (any
    size, uint8), under the front pass. Channel order passes through."""
    shape = (front if front is not None else behinds[0][0]).shape
    h, w = shape[:2]
    out = np.zeros((h, w, 4), np.float32)
    for img, matte in behinds:
        layer = img.astype(np.float32) / 255
        layer[..., 3] *= 1 - at_size(matte, w, h)
        out = _over(layer, out)
    if front is not None:
        out = _over(front.astype(np.float32) / 255, out)
    return np.rint(np.clip(out, 0, 1) * 255).astype(np.uint8)


def hidden(img: np.ndarray, matte: np.ndarray) -> float:
    """Share of a behind pass's ink the subject covers."""
    h, w = img.shape[:2]
    a = img[..., 3].astype(np.float32) / 255
    return float((a * at_size(matte, w, h)).sum() / max(float(a.sum()), 1e-6))
