"""``render(project, plan)``: validate → segments (cached by spec hash) → audio → concat + mux.
``renders/`` keeps the newest ``KEEP_RENDERS`` and ``cache/render`` what they use
(``opencine.core.retention``)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel

from opencine.core.cache import Cache, cache_key
from opencine.core.project import Project
from opencine.core.retention import remove, stale_render_cache, stale_renders
from opencine.plan.captions import cues as caption_cues
from opencine.plan.media import Resolver
from opencine.plan.schema import EditPlan
from opencine.plan.validate import Issue, errors, validate
from opencine.render.audio import build_mix, peak_trim, trimmed
from opencine.render.encode import PRESETS, Preset, encode_segment, mux, output_size
from opencine.render.segments import build, finalize_graph, split
from opencine.render.text import font_name
from opencine.render.textplace import resolve as resolve_text

PARALLEL_SEGMENTS = max(1, min(3, (os.cpu_count() or 2) // 4))
TP_RETRIES = 2  # re-mux with a trimmed mix while the encoded true peak is too high


def colour_luts(project: Project, plan: EditPlan, medias: list) -> list[Path | None]:
    """Per item: a LUT of its colour shift (``opencine.render.colour``) and the plan's look, or None."""
    from opencine.render.colour import shifts
    from opencine.render.look import cube_text, write_cube

    look = plan.look
    out: list[Path | None] = []
    for sh in shifts(project, plan, medias):
        if sh is None and look is None:
            out.append(None)
            continue
        text = cube_text(sh, look.name if look else None, look.strength if look else 1.0)
        out.append(write_cube(text, project.cache_dir / "lut"))
    return out


class PlanInvalid(ValueError):
    def __init__(self, issues: list[Issue]):
        super().__init__("; ".join(f"{i.path}: {i.message}" for i in issues))
        self.issues = issues


class RenderResult(BaseModel):
    path: Path
    duration_s: float
    segments_total: int
    segments_rendered: int
    elapsed_s: float
    notes: list[str] = []  # final renders: ESRGAN / RIFE intermediates used or skipped (ncnn)


def render(project: Project, plan: EditPlan, preset: Preset = "preview",
           out: Path | None = None) -> RenderResult:
    """Render ``plan`` to mp4: ``final`` at the plan's deliverable from the original sources,
    ``preview`` at 720×1280 from the proxies."""
    t0 = time.perf_counter()
    bad = errors(validate(plan, project))
    if bad:
        raise PlanInvalid(bad)
    enc = PRESETS[preset]
    rate = plan.fps
    res = Resolver(project, rate)
    medias = [res.media(it) for it in plan.items]
    plan = resolve_text(project, plan, medias)  # text "auto" → the band that hides least
    decoded = [m.decoded(enc.master) for m in medias]
    cache = Cache(project.cache_dir / "render")
    font = font_name()

    starts = plan.starts()
    size = output_size(preset, plan.size)
    notes: list[str] = []
    if enc.master:  # local GPU upscale / interpolation where it helps
        from opencine.render.ncnn import enhance

        decoded, notes = enhance(project, plan, medias, decoded, size)
    luts = colour_luts(project, plan, medias)
    mattes = None
    if any(it.transition_in.kind == "mask" for it in plan.items[1:]):  # mask wipes need mattes first
        from opencine.render.matte import prepare

        mattes, mask_notes = prepare(project, plan, decoded, size, enc, font, enc.master, luts, cache)
        notes += mask_notes
    cues = caption_cues(project, plan)
    segs = [build(plan, idxs, decoded, starts[idxs[0]], size, enc.args, font, final=enc.master,
                  luts=luts, mattes=mattes, captions=(plan.captions, cues) if cues else None)
            for idxs in split(plan)]
    paths = [cache.path("segment", s.key, ".mp4") for s in segs]
    # one job per distinct key: a repeated shot must not be encoded twice into the same file
    todo = list({s.key: (s, p) for s, p in zip(segs, paths, strict=True)
                 if cache.hit("segment", s.key, ".mp4") is None}.values())

    def work(job) -> None:
        seg, path = job
        ass = None
        if seg.ass:
            ass = cache.path("segment", seg.key, ".ass")
            ass.write_text(seg.ass, "utf-8")
        encode_segment(seg.inputs, finalize_graph(seg, ass), seg.out, seg.frames, enc, path)

    with ThreadPoolExecutor(PARALLEL_SEGMENTS) as pool:
        list(pool.map(work, todo))

    wav = build_mix(plan, medias, project, cache, [res.sound(it) for it in plan.items])
    key = cache_key("render", [s.key for s in segs], wav.name)
    tag = "-16x9" if plan.size[0] > plan.size[1] else ""
    out = Path(out) if out else project.renders_dir / f"{key[:10]}-{preset}{tag}.mp4"
    for attempt in range(TP_RETRIES + 1):
        mux([(p, s.frames) for s, p in zip(segs, paths, strict=True)], rate, wav,
            cache.path("concat", key, ".txt"), out)
        if attempt == TP_RETRIES or not (db := peak_trim(out)):
            break
        wav = trimmed(wav, db)  # AAC overshot −1 dBTP: lower the mix a little and re-mux
    cache.path("concat", key, ".json").write_text(json.dumps({"audio": wav.name}), "utf-8")
    remove(stale_renders(project))
    if out.parent.resolve() == project.renders_dir.resolve():  # a render kept elsewhere is not counted
        remove(stale_render_cache(project))
    return RenderResult(path=out, duration_s=float(Fraction(plan.total_frames) / rate),
                        segments_total=len(segs), segments_rendered=len(todo),
                        elapsed_s=round(time.perf_counter() - t0, 3), notes=notes)
