"""Place an item ``on`` a thing from the thing's box, and pick a ``scatter``'s doodle spots.

Boxes and places are frame fractions; ``aspect`` = height / width converts them to width units.
Only a ``doodle`` gets its own place (other items are drawn on the masks and take the box centre):

- ``u``, ``v``: that point of the box, 0–1 from its top left (beyond 0–1 = outside), e.g. an eye
  at (0.3, 0.4) of a face, a crown at (0.5, −0.1);
- else ``spot``: ``side`` = off the top-right corner (other side if no room; inside the corner if
  the box fills the frame), ``top`` = resting on the top edge, ``on`` = the box centre.

Without an explicit ``size`` a doodle is ``SIZE[spot]`` × the box width (``side``: its default).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SIZE = {"top": 0.8, "on": 0.8, "uv": 0.35}
SIZES = (0.04, 0.4)
EDGE = 0.06  # places stay this far in from the frame's sides
VARY = 0.35  # scattered doodles' sizes vary this much either way
TRIES = 80  # random spots tried per scattered doodle

Box = tuple[float, float, float, float]


def pick(boxes: list[Box], hint: tuple[float, float] | None) -> Box | None:
    """The smallest box containing ``hint`` (rough x, y), else the nearest centre; without a
    hint, the first (the detector's best)."""
    if not boxes:
        return None
    if hint is None:
        return boxes[0]
    x, y = hint
    inside = [b for b in boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if inside:
        return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return min(boxes, key=lambda b: math.hypot((b[0] + b[2]) / 2 - x, (b[1] + b[3]) / 2 - y))


def _in(v: float, lo: float = EDGE, hi: float = 1 - EDGE) -> float:
    return round(min(hi, max(lo, v)), 4)


def place(it: dict[str, Any], box: Box, given: set[str], aspect: float,
          uv: tuple[float, float] | None = None) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if it["type"] != "doodle":
        return {"x": round(cx, 4), "y": round(cy, 4)}
    spot, out = ("uv" if uv else it.get("spot", "side")), {}
    size = it["size"]
    if spot != "side" and "size" not in given:
        size = out["size"] = round(min(SIZES[1], max(SIZES[0], SIZE[spot] * (x1 - x0))), 4)
    r = size / 2
    if uv:
        return out | {"x": round(x0 + uv[0] * (x1 - x0), 4), "y": round(y0 + uv[1] * (y1 - y0), 4)}
    if spot == "top":  # the doodle's lower part rests on the top edge
        return out | {"x": _in(cx), "y": _in(y0 - 0.3 * size / aspect)}
    if spot == "on":
        return out | {"x": round(cx, 4), "y": round(cy, 4)}
    return out | {"x": _in(_beside(x0, x1, r, low_first=False)), "y": _in(_beside(y0, y1, r / aspect, low_first=True))}


def _beside(a: float, b: float, r: float, low_first: bool) -> float:
    """A doodle's centre on one axis: just past the preferred side (``low_first``), else the
    other side, else just inside the preferred one."""
    low, high = a - 0.6 * r, b + 0.6 * r
    fits = {"low": low - r >= EDGE, "high": high + r <= 1 - EDGE}
    order = ("low", "high") if low_first else ("high", "low")
    for side in order:
        if fits[side]:
            return low if side == "low" else high
    return a + 1.2 * r if order[0] == "low" else b - 1.2 * r


def scatter(n: int, size: float, area: Box, aspect: float, seed: int,
            busy: np.ndarray | None = None) -> list[tuple[float, float, float]]:
    """Up to ``n`` seeded spots ``(x, y, size)`` in ``area`` for doodles of about ``size`` (width
    units), clear of each other and of ``busy`` (a frame mask of the thing)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    spots: list[tuple[float, float, float]] = []
    for _ in range(n * TRIES):
        if len(spots) == n:
            break
        s = size * (1 + rng.uniform(-VARY, VARY))
        r = s / 2
        if x1 - x0 <= 2 * r or (y1 - y0) * aspect <= 2 * r:
            continue
        x, y = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r / aspect, y1 - r / aspect)
        if any(math.hypot(x - a, (y - b) * aspect) < 1.1 * (r + t / 2) for a, b, t in spots):
            continue
        if busy is not None and _hits(busy, x, y, r):
            continue
        spots.append((round(x, 4), round(y, 4), round(s, 4)))
    return spots


def _hits(busy: np.ndarray, x: float, y: float, r: float) -> bool:
    """Whether the square around (x, y) with half-side ``r`` (width units) touches ``busy``."""
    h, w = busy.shape
    px, py, pr = x * w, y * h, r * w
    return bool(busy[max(0, int(py - pr)):int(py + pr) + 1, max(0, int(px - pr)):int(px + pr) + 1].any())
