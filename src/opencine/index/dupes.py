"""Clips sent twice (e.g. chat-compressed and as a file): find them and keep the better copy.

The better copy (more pixels, then bigger file, then sent first) stays; the other gets
``ClipInfo.same_as`` and leaves L0, L1 and the match hints (its scene ids stay valid).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.model import ClipInfo

DURATION_S = 0.2
SAME_DETAIL = 0.95
SAME_SHARE = 0.8
MIN_SAMPLES = 4  # fewer comparable samples than this: too little to tell


def _detail(grid: list[int]) -> np.ndarray | None:
    from opencine.plan.match import _blur

    side = math.isqrt(len(grid))
    if side * side != len(grid) or side < 4:
        return None
    g = np.asarray(grid, float).reshape(side, side) / 255
    d = g - _blur(g)
    sd = d.std()
    return ((d - d.mean()) / sd).ravel() if sd > 0.01 else None


def details(rows: list[dict] | None) -> dict[int, np.ndarray]:
    """``{frame: normalised detail}`` of a clip's samples: the 16×16 grey thumbnail minus its
    blur, as in ``plan.match`` (colour and compression differ between copies, forms do not)."""
    out = {}
    for r in rows or []:
        if r.get("grid") and (d := _detail(r["grid"])) is not None:
            out[r["f"]] = d
    return out


def same_footage(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> bool:
    """The detail correlates (≥ ``SAME_DETAIL``) at ``SAME_SHARE`` of the shared sample times."""
    common = sorted(set(a) & set(b))
    if len(common) < MIN_SAMPLES:
        return False
    alike = sum(float(np.mean(a[f] * b[f])) >= SAME_DETAIL for f in common)
    return alike >= SAME_SHARE * len(common)


def _quality(project: Project, c: ClipInfo) -> tuple[int, int]:
    p = project.root / c.source
    size = p.stat().st_size if p.is_file() else 0
    return c.src_width * c.src_height, size


def mark(project: Project, clips: list[ClipInfo]) -> dict[str, str]:
    """Set ``same_as`` on every duplicate in ``clips`` (in place); returns ``{dup: kept}``.

    Candidates last as long (± ``DURATION_S``) and pass ``same_footage``."""
    from opencine.index.model import load_samples

    for c in clips:
        c.same_as = None
    found: dict[str, str] = {}
    desc = {c.id: details(load_samples(project, c.id)) for c in clips}
    ranked = sorted(clips, key=lambda c: _quality(project, c), reverse=True)  # ties: sent first stays
    for i, keep in enumerate(ranked):
        if keep.id in found:
            continue
        for other in ranked[i + 1:]:
            if other.id in found or abs(other.duration_s - keep.duration_s) > DURATION_S:
                continue
            if same_footage(desc[keep.id], desc[other.id]):
                found[other.id] = keep.id
                other.same_as = keep.id
    return found
