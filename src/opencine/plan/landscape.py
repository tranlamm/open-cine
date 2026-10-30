"""A 16:9 version of an approved 9:16 EditPlan: same items, times, text and sound, so both last
the same to the frame. Footage at least ``WIDE_FROM`` wide gets a 16:9 ``fill`` crop (subject
tracked, zoom kept); taller media gets ``fit_blur`` (its 9:16 framing over a blurred copy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opencine.plan.schema import ClipSource, EditPlan, SceneSource

if TYPE_CHECKING:
    from opencine.core.project import Project

LANDSCAPE = "1920x1080"
WIDE_FROM = 1.25  # source width / height from which a 16:9 crop of it beats a fitted 9:16


def derive(project: Project, plan: EditPlan) -> EditPlan:
    from opencine.index.model import load_index
    from opencine.plan.media import Resolver
    from opencine.plan.reframe import reframe

    index = load_index(project) if any(isinstance(it.source, SceneSource | ClipSource)
                                       for it in plan.items) else None
    res = Resolver(project, plan.fps)
    items = []
    for it in plan.items:
        d = it.model_dump(by_alias=True)
        m = res.media(it)
        if m.full_width / m.full_height >= WIDE_FROM:
            d["layout"] = "fill"
            d["crop"] = None
            if isinstance(it.source, SceneSource | ClipSource) and index is not None:
                if isinstance(it.source, SceneSource):
                    sc = index.scene(it.source.scene)
                    clip, first = sc.clip, sc.start_frame
                else:
                    clip, first = it.source.clip, 0
                d["crop"] = reframe(project, clip, first, it.in_, it.out, it.length, speed=it.speed,
                                    ramp=it.ramp, deliverable=(1920, 1080))
        else:
            d["layout"] = "fit_blur"
        items.append(d)
    return EditPlan.model_validate({**plan.model_dump(by_alias=True), "deliverable": LANDSCAPE,
                                    "items": items})
