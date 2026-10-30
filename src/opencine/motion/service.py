"""Motion pass: the Director's animation code (a TSX layer) drawn over one rendered EditPlan.

The layer imports ``opencine-motion`` (``fx/src/motion/kit.tsx``) and is saved as
``motion/v<NNN>.tsx`` with its ``marks``/``behind`` (``v<NNN>.json``). ``useMotion()`` gives it
beats, cuts, voice-over words, named marks, safe margins and matte names in timeline seconds.
Remotion draws a front pass plus one pass per ``<Behind>`` matte; each behind pass is cut by its
subject's matte and laid under the front, then FFmpeg overlays the result
(``renders/<render>-motion.mp4``; scribble draws on top of it). ``sheet`` returns check stills.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from opencine.core.cache import cache_key
from opencine.core.config import REPO_ROOT
from opencine.core.retention import remove, stale_renders
from opencine.motion import matte as mt

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.sheets import ImageRef
    from opencine.plan.schema import EditPlan

KIT = [REPO_ROOT / "fx" / "src" / "motion" / f for f in ("kit.tsx", "root.tsx")]
VERSION = 2  # cache version: bump when the output changes
WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))
TOO_HIDDEN, BARELY = 0.4, 0.03  # covered share of behind text that earns a warning


class MotionError(ValueError):
    pass


# ---- the Director's code -------------------------------------------------------------------
def code_path(project: Project, version: int) -> Path:
    return project.root / "motion" / f"v{version:03d}.tsx"


def _saved(project: Project, version: int) -> tuple[Path, int]:
    d = project.root / "motion"
    found = sorted((int(p.stem[1:]), p) for p in d.glob("v[0-9][0-9][0-9].tsx")) if d.is_dir() else []
    for v, p in reversed(found):
        if v <= version:
            return p, v
    raise MotionError("no motion layer yet: pass code")


def take(project: Project, version: int, code: str | None, marks: dict[str, Any] | None = None,
         behind: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any], dict[str, Any], list[str]]:
    """Save ``code`` (with ``marks``, ``behind``) for ``version``, or load the newest saved one
    (given ``marks``/``behind`` override). Returns ``(layer file, marks, behind, notes)``."""
    notes: list[str] = []
    if code is None:
        path, v = _saved(project, version)
        if v != version:
            notes.append(f"layer saved for editplan v{v}")
        side = path.with_suffix(".json")
        saved = json.loads(side.read_text("utf-8")) if side.is_file() else {}
        marks = saved.get("marks", {}) if marks is None else marks
        behind = saved.get("behind", {}) if behind is None else behind
    else:
        if "export default" not in code:
            raise MotionError("the layer must `export default` a component (see the opencine-edit skill, motion.md)")
        path = code_path(project, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, "utf-8")
        marks, behind = marks or {}, behind or {}
    try:
        mt.check(behind)
    except mt.MatteError as e:
        raise MotionError(str(e)) from None
    path.with_suffix(".json").write_text(json.dumps({"marks": marks, "behind": behind}, ensure_ascii=False,
                                                    indent=1), "utf-8")
    return path, marks, behind, notes


# ---- what the code times things by ---------------------------------------------------------
def layer_data(project: Project, plan: EditPlan, marks: dict[str, Any], width: int, height: int,
               mattes: list[str] | tuple = ()) -> dict[str, Any]:
    """``MotionData`` for ``fx/src/motion/kit.tsx``: beats, strong beats, cuts, voice-over words,
    resolved marks, safe margins (timeline seconds) and matte names."""
    from opencine.plan.captions import line_words
    from opencine.render.text import margins
    from opencine.scribble.anchor import timeline_s

    rate = float(plan.fps)
    end = plan.total_frames / rate
    beats: list[float] = []
    strong: list[float] = []
    if plan.music is not None:
        from opencine.music.beats import load_beats
        from opencine.music.library import beats_path, strong_threshold

        track = Path(plan.music.file)
        track = track if track.is_absolute() else project.root / track
        grid = beats_path(track)
        if grid.is_file() and (found := load_beats(grid)):
            thr = strong_threshold(found)
            off = plan.music.offset / rate
            for b in found:
                if 0 <= b.t - off < end:
                    beats.append(round(b.t - off, 3))
                    if b.strength >= thr:
                        strong.append(round(b.t - off, 3))
    resolved = {}
    for name, at in marks.items():
        try:
            resolved[name] = round(timeline_s(project, plan, at), 3)
        except (ValueError, KeyError) as e:
            raise MotionError(f"mark {name!r}: {e}") from None
    bottom, top, side = margins(width, height)
    return {
        "beats": beats, "strong": strong,
        "cuts": sorted({round(s / rate, 3) for s in plan.starts()}),
        "words": [{"text": w.text, "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
                  for _, ws in line_words(project, plan) for w in ws],
        "marks": resolved, "safe": {"top": top, "bottom": bottom, "side": side}, "mattes": list(mattes),
    }


# ---- fx/render.mjs -------------------------------------------------------------------------
class _workdir:
    """A scratch folder under the project's cache, removed afterwards."""

    def __init__(self, project: Project):
        self.path = project.cache_dir / "motion" / f"run-{time.time_ns()}"

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _node(job: dict[str, Any], wd: Path) -> dict[str, Any]:
    from opencine.scribble.service import ScribbleError, node

    try:
        return node(job, wd)
    except ScribbleError as e:
        raise MotionError(str(e)) from None


def with_motion(video: Path) -> Path:
    """``video``'s motion version if one is newer than this render, else ``video``."""
    m = video.with_name(f"{video.stem}-motion.mp4")
    return m if m.is_file() and m.stat().st_mtime >= video.stat().st_mtime else video


def mattes(project: Project, plan: EditPlan, version: int, video: Path, behind: dict[str, Any]
           ) -> dict[str, mt.Matte]:
    """Each ``behind`` spec's matte (vision models shared across specs)."""
    if not behind:
        return {}
    from opencine.scribble.service import _Vision

    vis = _Vision()
    try:
        return {name: mt.compute(project, plan, version, video, name, spec, vis) for name, spec in behind.items()}
    except mt.MatteError as e:
        raise MotionError(str(e)) from None


def _read(path: Path) -> np.ndarray:
    """A pass frame as BGRA (fully opaque PNGs are saved without alpha)."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    return img if img.ndim == 3 and img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def _write(path: Path, img: np.ndarray) -> None:
    cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tofile(path)


def lay(frames_dir: Path, res: dict[str, Any], found: dict[str, mt.Matte]) -> list[int]:
    """Composite front over matte-cut behind passes into the front pass files; returns all layer
    frames, sorted."""
    front = {f for a, b in res["runs"] for f in range(a, b + 1)}
    behind = {n: set(r["frames"]) for n, r in (res.get("behind") or {}).items()}
    todo = sorted(front.union(*behind.values()))

    def one(f: int) -> None:
        name = f"f{f:06d}.png"
        under = [(_read(frames_dir / f"behind-{n}" / name), m) for n, fs in behind.items()
                 if f in fs and (m := found[n].at(f)) is not None]
        if under:
            _write(frames_dir / name, mt.composite(_read(frames_dir / name) if f in front else None, under))

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    return todo


def runs_of(frames: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return runs


def _matte_lines(found: dict[str, mt.Matte], rate: float) -> list[str]:
    return [f"matte {n}: '{m.on}' · {m.first / rate:.2f}–{m.end / rate:.2f} s"
            for n, m in found.items()] + [n for m in found.values() for n in m.notes]


def _spans(spans: list[list[float]] | None) -> str:
    if spans is None:
        return "whole video (the layer declares no spans)"
    return ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in spans) or "none"


def render_motion(project: Project, code: str | None = None, version: int | None = None,
                  preset: str = "preview", marks: dict[str, Any] | None = None,
                  behind: dict[str, Any] | None = None) -> dict[str, Any]:
    """Draw the layer ``code`` (or the saved one) over the ``preset`` render of EditPlan
    ``version`` (latest); returns the new video's path, its spans and notes."""
    from opencine.plan.store import load_plan
    from opencine.scribble.service import base_render, overlay, video_meta

    t0 = time.perf_counter()
    v, plan = load_plan(project, "editplan", version)
    layer, marks, behind, notes = take(project, v, code, marks, behind)
    video = base_render(project, v, preset)
    meta = video_meta(video, plan)
    data = layer_data(project, plan, marks, meta["width"], meta["height"], list(behind))
    found = mattes(project, plan, v, video, behind)
    notes += _matte_lines(found, float(plan.fps))
    out = video.with_name(f"{video.stem}-motion.mp4")
    key = cache_key(video.name, video.stat().st_size, layer.read_text("utf-8"), data,
                    {n: m.key for n, m in found.items()}, [k.read_text("utf-8") for k in KIT], VERSION)
    stamp = out.with_suffix(".key")
    spans = None
    if out.is_file() and stamp.is_file() and json.loads(stamp.read_text())["key"] == key:
        spans = json.loads(stamp.read_text())["spans"]
    else:
        with _workdir(project) as wd:
            res = _node({"meta": meta, "layer": str(layer), "data": data, "out_dir": str(wd / "frames"),
                         "behind": {n: list(range(m.first, m.end)) for n, m in found.items()}}, wd)
            frames = lay(wd / "frames", res, found)
            overlay(video, runs_of(frames), res["pattern"], Fraction(plan.fps), preset, out, plan.total_frames)
        spans = res["spans"]
        stamp.write_text(json.dumps({"key": key, "spans": spans}))
    os.utime(out)  # a reused output must still count as the newest render
    remove(stale_renders(project))
    return {"path": str(out), "version": v, "preset": preset, "video": str(video),
            "spans": _spans(spans), "marks": data["marks"], "notes": notes,
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def sheet(project: Project, code: str | None = None, version: int | None = None,
          times: list[float] | None = None, marks: dict[str, Any] | None = None,
          behind: dict[str, Any] | None = None) -> tuple[list[ImageRef], str]:
    """Check stills of the layer over the newest render of ``version`` (preview, else final): one
    per span once its elements are in, or at ``times``. Returns the sheet(s) and a text table of
    the layer's data and how much behind text each subject covers."""
    from opencine.plan.store import load_plan
    from opencine.review.selfcheck import TILE, grab_frames, render_path, sheets
    from opencine.scribble.service import video_meta

    v, plan = load_plan(project, "editplan", version)
    layer, marks, behind, notes = take(project, v, code, marks, behind)
    video = render_path(project, v)
    meta = video_meta(video, plan)
    data = layer_data(project, plan, marks, meta["width"], meta["height"], list(behind))
    found = mattes(project, plan, v, video, behind)
    rate = float(plan.fps)
    last = plan.total_frames - 1
    covered: dict[str, list[tuple[float, float]]] = {n: [] for n in found}
    with _workdir(project) as wd:
        res = _node({"meta": meta, "layer": str(layer), "data": data, "out_dir": str(wd / "stills"),
                     "stills": times or "auto", "scale": TILE / meta["height"],
                     "behind": {n: list(range(m.first, m.end)) for n, m in found.items()}}, wd)
        frames = [min(last, s["frame"]) for s in res["stills"]]
        under = grab_frames(video, frames)
        tiles = []
        for f, still in zip(frames, res["stills"], strict=True):
            base = under[f].convert("RGBA")
            parts = []
            for n, path in (still.get("behind") or {}).items():
                if (m := found[n].at(still["frame"])) is not None:
                    img = _rgba(path)
                    covered[n].append((still["t"], mt.hidden(img, m)))
                    parts.append((img, m))
            top = mt.composite(_rgba(still["path"]), parts) if parts else _rgba(still["path"])
            base.alpha_composite(Image.fromarray(top).resize(base.size, Image.Resampling.LANCZOS))
            tiles.append((base.convert("RGB"), f"{still['t']:.2f}s"))
    refs = sheets(project, f"motion-v{v:03d}", tiles)
    named = ", ".join(f"{k} = {s:.2f} s" for k, s in data["marks"].items()) or "none"
    table = "\n".join([f"# motion · editplan v{v} · render {video.name}",
                       f"spans: {_spans(res['spans'])}", f"marks: {named}",
                       f"beats: {len(data['beats'])} (strong {len(data['strong'])}) · cuts: {len(data['cuts'])}"
                       f" · words: {len(data['words'])}", *_matte_lines(found, rate), *_covered(covered), *notes])
    return refs, table


def _rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def _covered(covered: dict[str, list[tuple[float, float]]]) -> list[str]:
    """Per matte: the share of behind text its subject covers in each still, with a warning when
    it is too much to read or next to nothing."""
    out = []
    for n, seen in covered.items():
        if not seen:
            out.append(f"matte {n}: no still falls inside its window; add times to check it")
            continue
        worst = max(share for _, share in seen)
        line = f"matte {n}: covers " + ", ".join(f"{share:.0%} at {t:.2f} s" for t, share in seen)
        if worst > TOO_HIDDEN:
            line += " · covers too much, text hard to read: move or resize the text, or shorten the window"
        elif worst < BARELY:
            line += " · barely covers: the subject does not pass in front of the text in these stills"
        out.append(line)
    return out
