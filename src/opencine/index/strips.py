"""L2 for several scenes in one image: start, motion peak and end of each.

Comparing candidates costs one image of the per-project image count instead of one strip per
scene. Tiles are 216×384 (or 384×216), labelled with their L3 ref (``S03@00:01.25``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opencine.core.cache import cache_key
from opencine.core.time import Rate
from opencine.index.model import Scene, load_index
from opencine.index.sheets import (
    MAX_EDGE,
    VERSION,
    ImageRef,
    _aspect,
    _compose,
    _frame_jpg,
    _peak,
    strip_label,
)
from opencine.index.steps import StepCache, place

if TYPE_CHECKING:
    from opencine.core.project import Project

MAX_SCENES = 8  # 24 tiles of 216×384 fit one ≤ 1568 px sheet (~3k tokens)
TILE = 384


def three_frames(scene: Scene, rate: Rate) -> list[int]:
    """First, motion peak (else middle) and last frame of ``scene``; fewer for a tiny scene."""
    first, last = scene.start_frame, scene.end_frame - 1
    peak = _peak(scene, rate)
    if peak is None or not first < peak < last:
        peak = (first + last) // 2
    return sorted({first, peak, last})


def scenes_sheet(project: Project, sids: list[str], *, steps: StepCache | None = None) -> ImageRef:
    """``index/sheets/scenes-S03-S05.jpg``: 3 frames (start · peak · end) per scene."""
    if not sids:
        raise ValueError("give at least one scene id")
    if len(sids) > MAX_SCENES:
        raise ValueError(f"{len(sids)} scenes; at most {MAX_SCENES} per sheet (call again for the rest)")
    index = load_index(project)
    steps = steps or StepCache(project.cache_dir)
    rate = Rate(index.rate)
    scenes = [index.scene(s.strip().upper()) for s in dict.fromkeys(sids)]
    tiles = [(_frame_jpg(project, index, s.clip, f, steps), strip_label(s, f, rate))
             for s in scenes for f in three_frames(s, rate)]
    aspect = _aspect(index, scenes)
    clip_keys = [index.clip(s.clip).proxy_key or index.clip(s.clip).sha256 for s in scenes]
    key = cache_key("L2multi", [s.model_dump() for s in scenes], clip_keys, aspect, TILE, VERSION)
    sheet = steps.file("sheet", key, ".jpg", lambda out: _compose(
        tiles, aspect, MAX_EDGE, TILE, None, out))
    name = "scenes-" + "-".join(s.id for s in scenes) + ".jpg"
    return ImageRef.of(place(sheet, project.sheets_dir / name))
