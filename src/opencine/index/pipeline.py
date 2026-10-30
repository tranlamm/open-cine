"""``ingest(project)``: probe → proxy → audio → cuts → frames/scenes → ``scenes.json``, per-clip
samples and subjects, L0 and L1.

Each step is cached by ``(source sha256, step, params, VERSION)``, downstream keys chaining the
upstream one, so a rerun is all cache hits.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opencine.core.cache import cache_key, sha256_file
from opencine.core.ffmpeg import probe_json
from opencine.index import frames, scenes, subjects
from opencine.index.dupes import mark as mark_dupes
from opencine.index.model import (
    ClipInfo,
    Scene,
    SceneIndex,
    load_index,
    samples_path,
    save_index,
    subjects_path,
)
from opencine.index.sheets import overview
from opencine.index.steps import StepCache, place
from opencine.index.table import describe_images, estimate_tokens, scenes_md
from opencine.ingest import audio, proxy
from opencine.ingest.probe import PARAMS as PROBE_PARAMS
from opencine.ingest.probe import ProbeInfo, low_res, probe, shot_from_name
from opencine.speech.service import transcribe_on_ingest

if TYPE_CHECKING:
    from opencine.core.project import Project, SourceRef

log = logging.getLogger(__name__)
VERSION = 2  # bump to invalidate every cached ingest step (and, via proxy_key, every image)
DERIVED_STEPS = ("frame", "sheet")  # image caches keyed by proxy_key, purged by ``force``


def _proxy_meta(path) -> dict[str, int]:
    v = next(s for s in probe_json(path)["streams"] if s["codec_type"] == "video")
    return {"width": int(v["width"]), "height": int(v["height"]), "frames": int(v["nb_frames"])}


def _ingest_clip(project: Project, src: SourceRef, steps: StepCache) -> tuple[ClipInfo, list[dict]]:
    path = project.root / src.file
    rate = project.rate
    sha = sha256_file(path)

    info = ProbeInfo.model_validate(
        steps.json("probe", cache_key(sha, "probe", PROBE_PARAMS, VERSION), lambda: probe(path).model_dump()))

    pkey = cache_key(sha, "proxy", str(rate), proxy.PARAMS, VERSION)
    cached = steps.file("proxy", pkey, ".mp4", lambda out: proxy.make_proxy(path, out, info, rate))
    proxy_path = place(cached, project.proxy_dir / f"{src.id}.mp4")
    meta = steps.json("proxy_meta", pkey, lambda: _proxy_meta(proxy_path))

    audio_rel = None
    if info.has_audio:
        akey = cache_key(sha, "audio", audio.PARAMS, VERSION)
        wav = steps.file("audio", akey, ".wav", lambda out: audio.extract_audio(path, out))
        audio_rel = place(wav, project.audio_dir / f"{src.id}.wav").relative_to(project.root).as_posix()

    # cuts are keyed on the source and the proxy's *geometry* only (not its colour), so a
    # colour-only proxy change keeps the cuts — and the scene ids saved plans point at
    gkey = cache_key(sha, "proxy", str(rate), proxy.GEOMETRY_PARAMS, VERSION)
    ckey = cache_key(gkey, "cuts", scenes.PARAMS, VERSION)
    cuts = steps.json("cuts", ckey, lambda: scenes.detect_cuts(proxy_path, rate))

    fkey = cache_key(ckey, pkey, "frames", frames.PARAMS, VERSION)  # frames read proxy pixels
    fdir = steps.dir("frames", fkey)
    clip_scenes = steps.json("frames", fkey, lambda: frames.analyse_clip(
        proxy_path, rate, meta["frames"], cuts, fdir,
        audio=project.root / audio_rel if audio_rel else None))
    for jpg in fdir.glob("*.jpg"):
        place(jpg, project.frames_dir / f"{src.id}_{jpg.name}")
    if (rows := fdir / frames.SAMPLES_FILE).exists():  # absent in caches older than samples
        place(rows, samples_path(project, src.id))
    skey = cache_key(ckey, pkey, "subjects", subjects.params(), VERSION)
    found = steps.json("subjects", skey, lambda: subjects.track(proxy_path, rate, meta["frames"], cuts))
    out = subjects_path(project, src.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found), "utf-8")

    shot_at, shot_from = info.shot_at, "tag" if info.shot_at else None
    if shot_at is None and (shot_at := shot_from_name(Path(src.original).name)):
        shot_from = "name"
    clip = ClipInfo(
        id=src.id,
        source=src.file,
        proxy=proxy_path.relative_to(project.root).as_posix(),
        audio=audio_rel,
        sha256=sha,
        proxy_key=pkey,
        codec=info.codec,
        fps_nominal=info.fps_nominal,
        fps_avg=info.fps_avg,
        vfr=info.vfr,
        hdr=info.hdr,
        rotation=info.rotation,
        src_width=info.width,
        src_height=info.height,
        width=meta["width"],
        height=meta["height"],
        frames=meta["frames"],
        duration_s=round(meta["frames"] / rate.fps, 3),
        shot_at=shot_at,
        shot_from=shot_from,
    )
    return clip, clip_scenes


def _moved_scenes(project: Project, index: SceneIndex) -> list[str]:
    """Ids of the previous index whose scene (clip, start, end) differs in ``index`` (or is
    gone): plans referencing them by id would silently show other footage."""
    try:
        old = load_index(project)
    except (FileNotFoundError, ValueError):
        return []
    new = {s.id: (s.clip, s.start_frame, s.end_frame) for s in index.scenes}
    return [s.id for s in old.scenes if new.get(s.id) != (s.clip, s.start_frame, s.end_frame)]


def ingest(project: Project, *, force: bool = False) -> dict[str, Any]:
    """Index every source of ``project``; returns a summary for the CLI/MCP."""
    t0 = time.perf_counter()
    steps = StepCache(project.cache_dir, force=force)
    if force:  # a forced re-encode keeps proxy_key, so drop the L1/L2/L3 images made from it
        for step in DERIVED_STEPS:
            shutil.rmtree(project.cache_dir / step, ignore_errors=True)
    clips: list[ClipInfo] = []
    all_scenes: list[Scene] = []
    for src in project.meta.sources:
        clip, clip_scenes = _ingest_clip(project, src, steps)
        clips.append(clip)
        for d in clip_scenes:
            all_scenes.append(Scene(id=f"S{len(all_scenes) + 1:02d}", clip=clip.id, **d))
    dupes = mark_dupes(project, clips)  # the same footage sent twice: keep the better copy
    index = SceneIndex(rate=str(project.rate), clips=clips, scenes=all_scenes)
    moved = _moved_scenes(project, index)
    save_index(project, index)
    speech = transcribe_on_ingest(project, index, steps)  # before L0: scene lines carry speech
    try:  # L0 lists the strongest match-cut pairs; never worth failing an ingest over
        from opencine.plan.match import save_top

        save_top(project)
    except Exception as e:  # noqa: BLE001
        log.warning("match candidates skipped: %s", e)
    images = describe_images(project)

    l0_tokens = estimate_tokens(scenes_md(project))
    l1 = overview(project, steps=steps, l0_tokens=l0_tokens)
    l1_tokens = sum(r.tokens for r in l1)
    low = [c.id for c in clips if low_res(c.src_width, c.src_height)]  # finals upscale these
    return {
        "pid": project.pid,
        "clips": len(clips),
        "scenes": len(all_scenes),
        "l0_tokens_est": l0_tokens,
        "l1": [r.model_dump() for r in l1],
        "l1_tokens": l1_tokens,
        "l0_l1_tokens": l0_tokens + l1_tokens,
        "cache": {"hits": steps.hits, "misses": steps.misses},
        "elapsed_s": round(time.perf_counter() - t0, 2),
    } | ({"speech": speech} if speech else {}) | ({"images": images} if images else {}) \
      | ({"duplicates": dupes} if dupes else {}) \
      | ({"low_res_sources": low} if low else {}) \
      | ({"scene_changes": {"changed": moved, "warning": (
          "scene boundaries moved since the last ingest: saved plans that use these "
          "{\"scene\": …} ids now point at different footage; re-check them")}} if moved else {})
