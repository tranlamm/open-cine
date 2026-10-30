"""Shot-to-shot colour matching: the CIELAB shift each item gets.

Each item's mean Lab is measured over its range (footage from the index's 2 fps samples,
generated clips from a few decoded frames, stills from the image); the reference is the median
of the footage items. ``color: "auto"`` moves an item that is a little off (ΔE76 in
``(AUTO_MIN, AUTO_MAX]``; beyond that the change is deliberate) ``AUTO_STRENGTH`` of the way;
``{match: "S03"}`` pulls toward a scene or moment by ``strength``; both clamp per channel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from opencine.core.cache import cache_key

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.media import Media
    from opencine.plan.schema import EditItem, EditPlan

AUTO_MIN, AUTO_MAX, GEN_AUTO_MAX = 4.0, 15.0, 20.0
AUTO_STRENGTH = 0.6
CLAMP = np.array([8.0, 6.0, 6.0])  # max |shift| in L, a, b
GEN_FRAMES = 6
NEIGHBOUR_DE = 10.0  # validator hint: adjacent items this far apart after matching


def _samples_lab(project: Project, clip: str, a: int, b: int) -> np.ndarray | None:
    from opencine.index.model import load_samples

    rows = [r for r in (load_samples(project, clip) or [])
            if a <= r["f"] < b and r.get("lab") and not r.get("bad")]
    if not rows:  # a short range between two samples: the nearest one
        rows = sorted((r for r in (load_samples(project, clip) or []) if r.get("lab")),
                      key=lambda r: abs(r["f"] - (a + b) / 2))[:1]
    return np.mean([r["lab"] for r in rows], axis=0) if rows else None


def _decoded_lab(project: Project, path: Path, a: int, b: int, rate) -> np.ndarray | None:
    from opencine.gen.qc import lab_stats, read_frames

    st = path.stat()
    key = cache_key("colour", str(path.resolve()), st.st_size, st.st_mtime_ns, a, b, str(rate))
    cache = project.cache_dir / "colour" / f"{key}.json"
    if cache.exists():
        return np.asarray(json.loads(cache.read_text("utf-8")))
    frames = read_frames(path, width=96)
    if not len(frames):
        return None
    lo, hi = max(0, min(a, len(frames) - 1)), max(1, min(b, len(frames)))
    picks = np.linspace(lo, hi - 1, min(GEN_FRAMES, hi - lo)).round().astype(int)
    mean = lab_stats(frames[picks])[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([round(float(x), 2) for x in mean]), "utf-8")
    return mean


def _image_lab(path: Path) -> np.ndarray:
    from opencine.gen.qc import lab_stats, load_rgb

    return lab_stats(load_rgb(path, (96, 170)))[0]


def item_lab(project: Project, it: EditItem, m: Media, rate) -> np.ndarray | None:
    """Mean Lab of what ``it`` shows (before any shift); ``rate`` = the plan's."""
    from opencine.plan.schema import ClipSource, GenSource, SceneSource

    src = it.source
    try:
        if isinstance(src, SceneSource | ClipSource):
            from opencine.index.model import load_index

            index = load_index(project)
            clip = index.scene(src.scene).clip if isinstance(src, SceneSource) else src.clip
            ir = float(index.rate)
            a = round((float(m.start_s) + it.in_ / float(rate)) * ir)
            b = round((float(m.start_s) + it.out / float(rate)) * ir)
            return _samples_lab(project, clip, a, b)
        if isinstance(src, GenSource):
            return _decoded_lab(project, m.path, it.in_, it.out, rate)
        return _image_lab(m.path)
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def ref_lab(project: Project, ref: str) -> np.ndarray | None:
    """Mean Lab of a scene (``S03``) or around a moment (``S03@2.1``: ±0.5 s)."""
    from opencine.index.model import load_index

    index = load_index(project)
    sid, _, at = ref.strip().upper().partition("@")
    sc = index.scene(sid)
    rate = float(index.rate)
    if at:
        f = sc.start_frame + round(float(at.replace(",", ".")) * rate)
        return _samples_lab(project, sc.clip, f - round(rate / 2), f + round(rate / 2) + 1)
    return _samples_lab(project, sc.clip, sc.start_frame, sc.end_frame)


def shifts(project: Project, plan: EditPlan, medias: list[Media]) -> list[np.ndarray | None]:
    """Per item: the Lab shift to apply, or None."""
    from opencine.plan.schema import ColorMatch, GenSource, ImageSource

    labs = [item_lab(project, it, m, plan.fps) if it.color != "none" else None
            for it, m in zip(plan.items, medias, strict=True)]
    foot = [lab for it, lab in zip(plan.items, labs, strict=True)
            if lab is not None and not isinstance(it.source, GenSource | ImageSource)]
    pool = foot or [lab for lab in labs if lab is not None]
    ref = np.median(pool, axis=0) if len(pool) >= 2 else None
    out: list[np.ndarray | None] = []
    for it, lab in zip(plan.items, labs, strict=True):
        if lab is None:
            out.append(None)
            continue
        if isinstance(it.color, ColorMatch):
            target = ref_lab(project, it.color.match)
            out.append(None if target is None else
                       np.clip((target - lab) * it.color.strength, -CLAMP, CLAMP))
            continue
        if ref is None:
            out.append(None)
            continue
        de = float(np.linalg.norm(ref - lab))
        top = GEN_AUTO_MAX if isinstance(it.source, GenSource) else AUTO_MAX
        out.append(np.clip((ref - lab) * AUTO_STRENGTH, -CLAMP, CLAMP) if AUTO_MIN < de <= top else None)
    return out


def neighbour_gaps(project: Project, plan: EditPlan, medias: list[Media]) -> list[tuple[int, float]]:
    """``(item k, ΔE)`` for each cut into ``k`` whose sides still differ by more than
    ``NEIGHBOUR_DE`` after the shifts (validator hint)."""
    labs = [item_lab(project, it, m, plan.fps) for it, m in zip(plan.items, medias, strict=True)]
    sh = shifts(project, plan, medias)
    after = [None if lab is None else lab + (s if s is not None else 0) for lab, s in zip(labs, sh, strict=True)]
    out = []
    for k in range(1, len(after)):
        if after[k] is not None and after[k - 1] is not None:
            de = float(np.linalg.norm(after[k] - after[k - 1]))
            if de > NEIGHBOUR_DE:
                out.append((k, round(de, 1)))
    return out


def summary(sh: list[np.ndarray | None]) -> dict[str, Any]:
    return {str(k + 1): [round(float(x), 1) for x in s] for k, s in enumerate(sh) if s is not None}
