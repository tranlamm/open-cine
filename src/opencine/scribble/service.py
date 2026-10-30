"""Scribble pass: hand-drawn marks over one rendered EditPlan version.

Items (saved as ``fx/v<NNN>.json``) get timeline seconds; a thing named by ``on`` is boxed by
Florence-2 once per shot and followed by EdgeTAM (else SAM 2.1; optical flow without masks), cached
per plan version. ``fx/render.mjs`` (zod schema in ``fx/src/schema.ts``) fills defaults and draws
transparent PNG runs; FFmpeg lays them over the render (``renders/<render>-scribble.mp4``, audio
copied). ``sheet`` returns check stills instead of a video.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from opencine.core.cache import Cache, cache_key
from opencine.core.config import REPO_ROOT, get_settings
from opencine.core.ffmpeg import BT709_TAGS, TO_BT709, probe_json, run
from opencine.core.proc import NO_WINDOW
from opencine.core.retention import remove, stale_renders
from opencine.scribble import track
from opencine.scribble.anchor import timeline_s
from opencine.scribble.place import pick, place, scatter

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.sheets import ImageRef
    from opencine.plan.schema import EditPlan

log = logging.getLogger(__name__)

FX_DIR = REPO_ROOT / "fx"
RENDER_JS = FX_DIR / "render.mjs"
ON = ("outline", "fill", "ring", "trail", "doodle", "scatter")
SHAPED = ("outline", "fill", "ring")  # drawn on the thing's contour keys
MAX_BOX = 0.88  # a found box over this much of the frame is not one thing
MAX_EACH = 6  # `each`: at most this many things, left to right
BUSY_WIDTH = 144  # analysis width of where scattered doodles may not go
BUSY_PAD = 0.03  # scattered doodles keep this far (of the width) from the thing
SCALE = (0.5, 2.0)  # a riding doodle grows and shrinks with its thing within this
DRAWN = 0.6  # an item cut short at the cut is drawn within this share of what is left
NODE_TIMEOUT_S = 900
VERSION = 2


class ScribbleError(ValueError):
    pass


# ---- the Director's items ------------------------------------------------------------------
def items_path(project: Project, version: int) -> Path:
    return project.root / "fx" / f"v{version:03d}.json"


def load_items(project: Project, version: int) -> tuple[list[dict], int]:
    """The items saved for ``version``, else the newest saved before it: ``(items, from v)``."""
    d = project.root / "fx"
    saved = sorted((int(p.stem[1:]), p) for p in d.glob("v[0-9][0-9][0-9].json")) if d.is_dir() else []
    for v, p in reversed(saved):
        if v <= version:
            return json.loads(p.read_text("utf-8"))["items"], v
    raise ScribbleError("no scribble items yet: pass items")


def save_items(project: Project, version: int, items: list[dict]) -> None:
    p = items_path(project, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": version, "items": items}, ensure_ascii=False, indent=1), "utf-8")


# ---- fx/render.mjs -------------------------------------------------------------------------
def node(job: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Run ``fx/render.mjs`` on ``job``; its JSON answer, or ScribbleError with its reason."""
    if not (FX_DIR / "node_modules").is_dir():
        raise ScribbleError("fx/node_modules is missing: run `cd fx && npm ci`")
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "job.json"
    path.write_text(json.dumps(job, ensure_ascii=False), "utf-8")
    cmd = [get_settings().node, str(RENDER_JS), str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=NODE_TIMEOUT_S, cwd=REPO_ROOT, **NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ScribbleError(f"fx/render.mjs did not run: {e}") from e
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        res = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise ScribbleError(f"fx/render.mjs exited {proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}") from None
    if not res.get("ok"):
        raise ScribbleError(res.get("error", "fx/render.mjs failed"))
    return res


def video_meta(video: Path, plan: EditPlan) -> dict[str, Any]:
    v = next(s for s in probe_json(video)["streams"] if s["codec_type"] == "video")
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": float(plan.fps),
            "frames": plan.total_frames}


# ---- resolve -------------------------------------------------------------------------------
def _point(it: dict[str, Any]) -> tuple[float, float]:
    """What a tracked item follows: ``track: [x, y]``, else its own place (an arrow's tip)."""
    t = it.get("track")
    if isinstance(t, list) and len(t) == 2 and all(isinstance(v, int | float) for v in t):
        return float(t[0]), float(t[1])
    return float(it["x"]), float(it["y"])


def shot_end(plan: EditPlan, frame: int) -> int:
    """Exclusive end frame of the shot on screen at ``frame`` (during a transition: the
    incoming one)."""
    end = plan.total_frames
    for start, it in zip(plan.starts(), plan.items, strict=True):
        if start <= frame < start + it.length:
            end = start + it.length
    return end


def shot_clear(plan: EditPlan, frame: int) -> int:
    """``frame``, or the first frame after the transition into its shot when it falls in it: the
    shot is seen alone from there (a dissolve's blend would mislead the detector)."""
    clear, end = frame, plan.total_frames
    for start, it in zip(plan.starts(), plan.items, strict=True):
        if start <= frame < start + it.length:
            clear = start if it.is_cut or start == 0 else start + it.transition_in.frames
            end = start + it.length
    return min(max(frame, clear), end - 1)


def _window(it: dict[str, Any], plan: EditPlan) -> tuple[int, int]:
    """First frame and frame count to follow: the item's span, within its shot."""
    rate = float(plan.fps)
    first = min(plan.total_frames - 1, round(it["at_s"] * rate))
    n = min(shot_end(plan, first), plan.total_frames) - first
    return first, max(1, min(n, round(it["dur_s"] * rate)))


def _follow(project: Project, plan: EditPlan, version: int, video: Path, it: dict[str, Any],
            point: tuple[float, float] | None = None) -> list[dict]:
    point = point or _point(it)
    first, n = _window(it, plan)
    cache = Cache(project.cache_dir / "scribble")
    key = cache_key("track", version, first, n, point, track.VERSION)
    if hit := cache.hit("track", key, ".json"):
        keys = json.loads(hit.read_text("utf-8"))
    else:
        keys = track.follow(track.frames(video, first, n, float(plan.fps)), point, float(plan.fps))
        cache.path("track", key, ".json").write_text(json.dumps(keys), "utf-8")
    dx, dy = it["x"] - point[0], it["y"] - point[1]  # the item keeps its offset from the point
    return [{"t": k["t"], "x": round(min(1.0, max(0.0, k["x"] + dx)), 4),
             "y": round(min(1.0, max(0.0, k["y"] + dy)), 4)} for k in keys]


class _Vision:
    """Florence-2, EdgeTAM and SAM 2.1 (fallback) for one pass, loaded lazily (~2 GB together);
    ``get`` returns None when a model is not installed or fetched."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name not in self.loaded:
            try:
                if name == "florence":
                    from opencine.vision.florence import Florence as model
                elif name == "tam":
                    from opencine.vision.tam import Tam as model  # type: ignore[assignment]
                else:
                    from opencine.render.matte import Sam as model  # type: ignore[assignment]
                self.loaded[name] = model.load()
            except Exception as e:  # noqa: BLE001 — a missing package or a broken file: fall back
                log.warning("%s did not load: %s", name, e)
                self.loaded[name] = None
        return self.loaded[name]


@dataclass
class _Ask:
    """What the Director asked of an item beyond the fields the renderer takes."""

    on: str | None
    given: set[str]
    each: bool = False
    colors: list[str] | None = None
    uv: tuple[float, float] | None = None


def _look(project: Project, plan: EditPlan, version: int, video: Path, phrase: str, first: int,
          vis: _Vision) -> tuple[list[list[float]], str]:
    """Boxes of ``phrase`` on frame ``first`` (one per thing, best first), or ``[]`` and why."""
    cache = Cache(project.cache_dir / "scribble")
    key = cache_key("on", version, first, phrase.lower(), track.VERSION)
    if hit := cache.hit("on", key, ".json"):
        found = json.loads(hit.read_text("utf-8"))
    else:
        fl = vis.get("florence")
        if fl is None:  # not cached: installing should take effect
            return [], "cannot be looked for (vision models off or not installed: extra `vision`)"
        rgb = track.frames(video, first, 1, float(plan.fps), width=track.TAM_WIDTH)[0]
        found = [list(f.box) for f in fl.find(fl.see(rgb), phrase)]
        cache.path("on", key, ".json").write_text(json.dumps(found), "utf-8")
    things = [b for b in found if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_BOX]
    return things, "" if things else "fills the frame (a close-up)" if found else "is not found there"


def _group(plan: EditPlan, it: dict[str, Any], on: str) -> tuple[str, int]:
    """Items on the same thing in the same shot share one look and one run of masks."""
    return on.lower(), shot_end(plan, round(it["at_s"] * float(plan.fps)))


def _things(project: Project, plan: EditPlan, version: int, video: Path, items: list[dict[str, Any]],
            asked: list[_Ask], vis: _Vision) -> dict[tuple[str, int], dict[str, Any]]:
    """Per (thing, shot): its boxes on the first clear frame of its first item, and the frame span
    and sample rate (trail rate if any item is a trail) its masks must cover."""
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for it, ask in zip(items, asked, strict=True):
        if not ask.on:
            continue
        first, n = _window(it, plan)
        g = groups.setdefault(_group(plan, it, ask.on), {"first": first, "end": first + n, "fps": track.MASK_FPS})
        g["first"], g["end"] = min(g["first"], first), max(g["end"], first + n)
        if it["type"] == "trail":
            g["fps"] = track.TRAIL_FPS
    for (phrase, _), g in groups.items():
        g["first"] = min(shot_clear(plan, g["first"]), g["end"] - 1)
        g["things"], g["why"] = _look(project, plan, version, video, phrase, g["first"], vis)
    return groups


def _masks(project: Project, plan: EditPlan, version: int, video: Path, first: int, n: int,
           box: list[float] | None, point: tuple[float, float], fps: float,
           vis: _Vision) -> tuple[track.Found | None, str]:
    """Masks of the thing in ``box`` (else at ``point``) from frame ``first`` over ``n`` frames
    (EdgeTAM, else SAM 2.1); ``(None, why)`` with neither. Cached per plan version."""
    cache = Cache(project.cache_dir / "scribble")
    key = cache_key("masks", version, first, n, box or point, fps, track.VERSION)
    if hit := cache.hit("masks", key, ".npz"):
        return track.load_masks(hit), "nothing segmented there"
    rate = float(plan.fps)
    if (tam := vis.get("tam")) is not None:
        rgb = track.frames(video, first, n, rate, width=track.TAM_WIDTH)
        found = track.masks(tam, rgb, rate, box=tuple(box) if box else None,  # type: ignore[arg-type]
                            point=None if box else point, fps=fps)
    elif (sam := vis.get("sam")) is not None:
        rgb = track.frames(video, first, n, rate, width=track.SAM_WIDTH)
        found = track.sam_masks(sam, rgb, rate, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else point, fps)
    else:  # not cached: installing should take effect
        return None, "no segmentation model (extra `vision`)"
    track.save_masks(cache.path("masks", key, ".npz"), found)
    return found, "nothing segmented there"


def _found(project: Project, plan: EditPlan, version: int, video: Path, it: dict[str, Any],
           box: list[float] | None, g: dict[str, Any] | None, vis: _Vision) -> tuple[track.Found | None, str]:
    """The thing's masks over the item's span (times from the item's start): cut from the shared
    run when ``on`` found it, else segmented from the item's x, y."""
    first, n = _window(it, plan)
    rate = float(plan.fps)
    if box is None or g is None:
        fps = track.TRAIL_FPS if it["type"] == "trail" else track.MASK_FPS
        return _masks(project, plan, version, video, first, n, None, (float(it["x"]), float(it["y"])), fps, vis)
    found, why = _masks(project, plan, version, video, g["first"], g["end"] - g["first"], box,
                        (float(it["x"]), float(it["y"])), g["fps"], vis)
    if found is None:
        return None, why
    off, span = (first - g["first"]) / rate, n / rate
    return [(round(t - off, 3), m) for t, m in found if off - 1e-3 <= t <= off + span + 1e-3], why


def _ride(it: dict[str, Any], found: track.Found) -> list[dict] | None:
    """Track keys of a doodle riding a thing: same place on its box, scaled with it (``s``)."""
    boxes = track.boxes(found)
    if not boxes:
        return None
    b0 = boxes[0]
    c0, w0 = ((b0["x0"] + b0["x1"]) / 2, (b0["y0"] + b0["y1"]) / 2), b0["x1"] - b0["x0"]
    keys = []
    for b in boxes:
        s = min(SCALE[1], max(SCALE[0], (b["x1"] - b["x0"]) / w0)) if w0 > 0 else 1.0
        keys.append({"t": b["t"], "x": _unit((b["x0"] + b["x1"]) / 2 + (it["x"] - c0[0]) * s),
                     "y": _unit((b["y0"] + b["y1"]) / 2 + (it["y"] - c0[1]) * s), "s": round(s, 3)})
    return keys


def _busy(found: track.Found) -> np.ndarray | None:
    """Union of the thing's masks over the span, holes filled and dilated, ``BUSY_WIDTH`` wide."""
    if not found:
        return None
    m = np.any(np.stack([m for _, m in found]), axis=0).astype(np.uint8)
    outer, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, outer, -1, 1, cv2.FILLED)
    h, w = m.shape
    small = (cv2.resize(m, (BUSY_WIDTH, max(1, round(BUSY_WIDTH * h / w))), interpolation=cv2.INTER_AREA) > 0)
    grow = max(1, round(BUSY_PAD * BUSY_WIDTH))
    return cv2.dilate(small.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)) > 0


def _marks(it: dict[str, Any], busy: np.ndarray | None, aspect: float) -> list[dict]:
    """A scatter's doodles: spots clear of the thing and each other, kinds and colours dealt round
    (seeded), ordered outwards from the thing (else the frame centre)."""
    spots = scatter(it["count"], it["size"], tuple(it["area"]), aspect, it["seed"], busy)  # type: ignore[arg-type]
    if busy is not None and busy.any():
        ys, xs = np.nonzero(busy)
        cx, cy = float(xs.mean()) / busy.shape[1], float(ys.mean()) / busy.shape[0]
    else:
        cx, cy = 0.5, 0.45
    spots.sort(key=lambda p: math.hypot(p[0] - cx, (p[1] - cy) * aspect))
    rng = np.random.default_rng(it["seed"] + 1)
    kinds = [str(k) for k in rng.permutation(it["kinds"])]
    colors = it["colors"] or [it["color"]]
    return [{"kind": kinds[k % len(kinds)], "x": x, "y": y, "size": size,
             "rotate": round(float(rng.uniform(-20, 20)), 1), "color": colors[k % len(colors)],
             "delay": round(k * it["stagger_s"], 3)} for k, (x, y, size) in enumerate(spots)]


def _unit(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 4)


def _fit_shot(it: dict[str, Any], plan: EditPlan, n: int, notes: list[str]) -> None:
    """End an item at its shot's cut (drawn within ``DRAWN`` of what is left); an untracked
    doodle may run over the cut (noted)."""
    if it["type"] == "frame":
        return
    rate = float(plan.fps)
    first = round(it["at_s"] * rate)
    cut = shot_end(plan, first)
    if round((it["at_s"] + it["dur_s"]) * rate) <= cut + 2:
        return
    cut_s = f"{cut / rate:.2f}"
    if it["type"] == "doodle" and it.get("track") is None:
        notes.append(f"item {n}: runs over the cut at {cut_s} s")
        return
    it["dur_s"] = round(max(1, cut - first) / rate, 3)
    note = f"item {n}: ends at the cut {cut_s} s (it is drawn on that shot)"
    if it["draw_s"] > DRAWN * it["dur_s"]:  # finish drawing while the shot shows
        it["draw_s"] = round(DRAWN * it["dur_s"], 3)
        note += f", drawn in {it['draw_s']:.2f} s"
    notes.append(note)


def _shape(project: Project, plan: EditPlan, version: int, video: Path, it: dict[str, Any],
           box: list[float] | None, g: dict[str, Any] | None, aspect: float, vis: _Vision) -> tuple[bool, str]:
    """Fill in what the renderer needs from the video (contour keys, trail path, doodle ride,
    scatter marks); ``(False, why)`` when there is nothing to draw."""
    kind = it["type"]
    if kind in SHAPED:
        found, why = _found(project, plan, version, video, it, box, g, vis)
        it["keys"] = track.contours(found, it["offset"] if kind == "outline" else 0.0) if found else []
        return bool(it["keys"]), why
    if kind == "scatter":
        found = _found(project, plan, version, video, it, box, g, vis)[0] if box else None
        it["marks"] = _marks(it, _busy(found) if found else None, aspect)
        return bool(it["marks"]), "no room for its doodles in its area"
    if (kind == "trail" or (kind == "doodle" and "track" in it)) and not _has_keys(it.get("track")):
        found = _found(project, plan, version, video, it, box, g, vis)[0] if box else None
        keys = (track.centroids(found) if kind == "trail" else _ride(it, found)) if found else None
        it["track"] = keys or _follow(project, plan, version, video, it)
    return True, ""


def _ask(n: int, it: dict[str, Any], raw: dict[str, Any]) -> _Ask:
    """Take the engine's own fields off an item (``on``, ``each``, ``colors``, ``u``/``v``)."""
    kind = it.get("type")
    on = it.pop("on", None)
    each = it.pop("each", False)
    colors = it.pop("colors", None) if kind != "scatter" else None
    u, v = it.pop("u", None), it.pop("v", None)
    if on is not None:
        if not isinstance(on, str) or not on.strip():
            raise ScribbleError(f"item {n}: on: a short English phrase, e.g. 'dancer', 'red jacket'")
        if kind not in ON:
            raise ScribbleError(f"item {n}: on works for {', '.join(ON)}")
        if isinstance(it.get("track"), list):
            raise ScribbleError(f"item {n}: give on or track [x, y], not both")
        on = on.strip()
    elif kind in (*SHAPED, "trail") and not {"x", "y"} <= set(it) and not isinstance(it.get("track"), list):
        raise ScribbleError(f"item {n}: {kind} is drawn on a thing: give on (a few English words) or x, y on it")
    if each is not False and (each is not True or not on or kind == "scatter"):
        raise ScribbleError(f"item {n}: each: true needs on (every thing found gets the item; not a scatter)")
    if colors is not None and (not isinstance(colors, list) or not colors or not all(isinstance(c, str) for c in colors)):
        raise ScribbleError(f"item {n}: colors: a list of colours, e.g. ['#FF2BD6', '#00E5FF']")
    if (u, v) != (None, None) and (kind != "doodle" or not on or not all(isinstance(t, int | float) for t in (u, v))):
        raise ScribbleError(f"item {n}: u, v (a point of the thing's box, 0–1 from its top left) go together, "
                            f"on a doodle with on")
    if it.get("track") is False:
        del it["track"]
    return _Ask(on, set(raw), each, colors, (float(u), float(v)) if u is not None else None)  # type: ignore[arg-type]


def resolve(project: Project, plan: EditPlan, version: int, video: Path,
            items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Items ready to draw (times, defaults, things, shapes, paths, rides and scatters resolved;
    ``_n`` = the Director's item number) and notes on what changed."""
    if not items:
        raise ScribbleError("no items")
    total = float(Fraction(plan.total_frames) / plan.fps)
    timed, asked = [], []
    for n, raw in enumerate(items, 1):
        if not isinstance(raw, dict):
            raise ScribbleError(f"item {n}: expected an object, got {raw!r}")
        it = dict(raw)
        if "at" in it:
            if "at_s" in it:
                raise ScribbleError(f"item {n}: give at or at_s, not both")
            try:
                it["at_s"] = timeline_s(project, plan, it.pop("at"))
            except (ValueError, KeyError) as e:
                raise ScribbleError(f"item {n}: {e}") from None
        if "at_s" not in it:
            raise ScribbleError(f"item {n}: give at (seconds, 'S03@2.1' or 'say:<word>')")
        if it["at_s"] >= total:
            raise ScribbleError(f"item {n}: at {it['at_s']:.2f} s is past the end ({total:.2f} s)")
        asked.append(_ask(n, it, raw))
        timed.append(it)
    meta = video_meta(video, plan)
    with _workdir(project) as wd:
        filled = node({"meta": meta, "items": timed, "normalize": True}, wd)["items"]
    notes: list[str] = []
    for n, (it, ask) in enumerate(zip(filled, asked, strict=True), 1):
        if ask.on and it["type"] == "doodle" and "track" not in ask.given:
            it["track"] = True  # it rides on what it is on
        _fit_shot(it, plan, n, notes)
    vis = _Vision()
    groups = _things(project, plan, version, video, filled, asked, vis)
    aspect = meta["height"] / meta["width"]
    out = []
    for n, (it, ask) in enumerate(zip(filled, asked, strict=True), 1):
        g, targets = None, [None]
        if ask.on:
            g = groups[_group(plan, it, ask.on)]
            hint = (it["x"], it["y"]) if {"x", "y"} <= ask.given else None
            things = [tuple(b) for b in g["things"]]
            targets = sorted(things)[:MAX_EACH] if ask.each and things else [pick(things, hint)]  # type: ignore[arg-type]
            at = f"{it['at_s']:.2f}"
            if targets[0] is None:
                if not hint:  # drop this item; don't fail the rest
                    notes.append(f"item {n}: dropped: on '{ask.on}' {g['why']} at {at} s: try another English "
                                 f"phrase, or give x, y on it")
                    continue
                notes.append(f"item {n}: on '{ask.on}' {g['why']} at {at} s → from x, y")
            else:
                notes.append(f"item {n}: on '{ask.on}' → " + "; ".join(
                    f"x {_span(b[0], b[2])} · y {_span(b[1], b[3])}" for b in targets))  # type: ignore[index]
        for k, box in enumerate(targets):
            one = copy.deepcopy(it) if len(targets) > 1 else it
            one["seed"] = it["seed"] + 7 * k
            if ask.colors:
                one["color"] = ask.colors[k % len(ask.colors)]
            if box is not None:
                one.update(place(one, box, ask.given, aspect, ask.uv))
            drawn, why = _shape(project, plan, version, video, one, list(box) if box else None,
                                g if box else None, aspect, vis)
            if drawn:
                out.append(one | {"_n": f"{n}" if len(targets) == 1 else f"{n}.{k + 1}"})
            else:
                notes.append(f"item {n}: {one['type']} dropped: {why}")
    if not out:
        raise ScribbleError("nothing left to draw: " + "; ".join(notes))
    return out, notes


def _bare(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Items as the renderer takes them (without the engine's ``_n``)."""
    return [{k: v for k, v in it.items() if k != "_n"} for it in items]


def _span(a: float, b: float) -> str:
    return f"{a:.2f}–{b:.2f}"


def _has_keys(t: Any) -> bool:
    return isinstance(t, list) and bool(t) and isinstance(t[0], dict)


class _workdir:
    """A scratch folder under the project's cache, removed afterwards."""

    def __init__(self, project: Project):
        self.path = project.cache_dir / "scribble" / f"run-{time.time_ns()}"

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


# ---- the video -----------------------------------------------------------------------------
def base_render(project: Project, version: int, preset: str) -> Path:
    """The newest render of ``version`` at ``preset``, rendering it if missing."""
    from opencine.review.selfcheck import render_path

    try:
        return render_path(project, version, preset)
    except FileNotFoundError:
        from opencine.plan.store import load_plan
        from opencine.render.compile import render

        return render(project, load_plan(project, "editplan", version)[1], preset).path  # type: ignore[arg-type]


def rendered(project: Project, version: int, preset: str) -> Path:
    """What scribble draws on: ``base_render``, with its motion layer when one was drawn on it."""
    from opencine.motion.service import with_motion

    return with_motion(base_render(project, version, preset))


def overlay(video: Path, runs: list[list[int]], pattern: str, rate: Fraction, preset: str, out: Path,
            frames: int) -> None:
    """Lay each run of PNG frames ``pattern % f`` over ``video`` at its frame; one x264 encode."""
    from opencine.render.encode import PRESETS

    num, den = rate.numerator, rate.denominator
    inputs = ["-i", str(video)]
    graph, last = [], "0:v"
    for k, (a, _b) in enumerate(runs, 1):
        inputs += ["-framerate", f"{num}/{den}", "-start_number", str(a), "-i", pattern]
        graph.append(f"[{k}:v]setpts=PTS-STARTPTS+{a * den}/{num}/TB,"
                     f"scale={TO_BT709},format=yuva420p[o{k}]")
        graph.append(f"[{last}][o{k}]overlay=eof_action=pass:format=yuv420[v{k}]")
        last = f"v{k}"
    graph.append(f"[{last}]{BT709_TAGS}[vout]")
    script = out.with_suffix(".filtergraph")
    script.write_text(";\n".join(graph), "utf-8")
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    try:
        run(["-y", "-loglevel", "error", *inputs, "-/filter_complex", str(script), "-map", "[vout]",
             "-map", "0:a?", "-c:a", "copy", *PRESETS[preset].args, "-frames:v", str(frames),
             "-movflags", "+faststart", str(tmp)])
        tmp.replace(out)
    finally:
        script.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def render_scribble(project: Project, items: list[dict] | None = None, version: int | None = None,
                    preset: str = "preview") -> dict[str, Any]:
    """Draw ``items`` (or the saved ones) over the ``preset`` render of EditPlan ``version``
    (latest); returns the new video's path, the resolved items and notes."""
    from opencine.plan.store import load_plan

    t0 = time.perf_counter()
    v, plan = load_plan(project, "editplan", version)
    notes: list[str] = []
    if items is None:
        items, src = load_items(project, v)
        if src != v:
            notes.append(f"items saved for editplan v{src}")
    else:
        save_items(project, v, items)
    video = rendered(project, v, preset)
    resolved, more = resolve(project, plan, v, video, items)
    notes += more
    out = video.with_name(f"{video.stem}-scribble.mp4")
    key = cache_key(video.name, video.stat().st_size, resolved, VERSION)
    stamp = out.with_suffix(".key")
    if not (out.is_file() and stamp.is_file() and stamp.read_text() == key):
        with _workdir(project) as wd:
            meta = video_meta(video, plan)
            res = node({"meta": meta, "items": _bare(resolved), "out_dir": str(wd / "frames")}, wd)
            overlay(video, res["runs"], res["pattern"], Fraction(plan.fps), preset, out, plan.total_frames)
        stamp.write_text(key)
    os.utime(out)  # a reused output must still count as the newest render
    remove(stale_renders(project))
    return {"path": str(out), "version": v, "preset": preset, "video": str(video),
            "items": summary(resolved), "notes": notes, "elapsed_s": round(time.perf_counter() - t0, 1)}


def _num(x: float) -> str:
    return f"{x:.2f}"


def summary(items: list[dict[str, Any]]) -> list[str]:
    """One line per item: ``#2 ring neon · 4.30–5.90 s · (0.52, 0.47) · 15 shape keys``
    (``#3.2`` = the 2nd thing of item 3's ``each``)."""
    num = _num
    out = []
    for n, it in enumerate(items, 1):
        kind = it.get("kind") or it.get("pattern") or it.get("style") or ""
        look = "" if it.get("look", "pen") == "pen" else it["look"]
        parts = [" ".join(filter(None, [f"#{it.get('_n', n)}", it["type"], kind, look])),
                 f"{num(it['at_s'])}–{num(it['at_s'] + it['dur_s'])} s"]
        if "x" in it:
            parts.append(f"({num(it['x'])}, {num(it['y'])})")
        if it.get("track"):
            parts.append("tracked")
        if it["type"] in SHAPED:
            parts.append(f"{len(it['keys'])} shape keys")
        if it["type"] == "scatter":
            parts.append(f"{len(it['marks'])} doodle")
        out.append(" · ".join(parts))
    return out


def shots(plan: EditPlan, items: list[dict[str, Any]]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Each shot's frames ``[start, end)`` and the items starting on it."""
    rate = float(plan.fps)
    spans = sorted({(s, s + it.length) for s, it in zip(plan.starts(), plan.items, strict=True)})
    return [(a, b, [it for it in items if shot_end(plan, round(it["at_s"] * rate)) == b]) for a, b in spans]


def _moment(a: int, b: int, inside: list[dict[str, Any]], rate: float) -> float:
    """The time most of a shot's items show fully drawn (a trail: halfway through it)."""
    def ready(it: dict[str, Any]) -> float:
        lag = 0.5 * it["dur_s"] if it["type"] == "trail" else it["draw_s"] + 0.15
        return it["at_s"] + min(lag, max(0.6 * it["dur_s"], it["dur_s"] - 0.25))

    def showing(t: float) -> int:
        return sum(ready(it) - 1e-6 <= t <= it["at_s"] + it["dur_s"] for it in inside)

    t = max((ready(it) for it in inside), key=showing)
    return min(max(t, a / rate), (b - 1) / rate)


# ---- the check sheet -----------------------------------------------------------------------
def sheet(project: Project, items: list[dict] | None = None, version: int | None = None,
          times: list[float] | None = None) -> tuple[list[ImageRef], str]:
    """Check stills of the layer over the newest render of ``version`` (preview, else final): one
    per shot with items (when most show drawn), or at ``times``. Returns the sheet(s) and a text
    table of the items and how many start on each shot."""
    from opencine.plan.store import load_plan
    from opencine.review.selfcheck import TILE, grab_frames, render_path, sheets

    v, plan = load_plan(project, "editplan", version)
    if items is None:
        items, _ = load_items(project, v)
    else:
        save_items(project, v, items)
    video = render_path(project, v)
    resolved, notes = resolve(project, plan, v, video, items)
    rate = float(plan.fps)
    last = plan.total_frames - 1
    cuts = shots(plan, resolved)
    if times:
        picks = [(min(last, round(t * rate)), f"{t:.1f}s") for t in times]
    else:
        picks = []
        for k, (a, b, inside) in enumerate(cuts, 1):
            if inside:
                t = _moment(a, b, inside, rate)
                picks.append((min(last, round(t * rate)), f"shot {k} {t:.1f}s · {len(inside)}"))
    meta = video_meta(video, plan)
    with _workdir(project) as wd:
        res = node({"meta": meta, "items": _bare(resolved), "out_dir": str(wd / "stills"),
                    "stills": [f / rate for f, _ in picks], "scale": TILE / meta["height"]}, wd)
        under = grab_frames(video, [f for f, _ in picks])
        tiles = []
        for (f, label), still in zip(picks, res["stills"], strict=True):
            base = under[f].convert("RGBA")
            with Image.open(still["path"]) as layer:
                base.alpha_composite(layer.convert("RGBA").resize(base.size, Image.Resampling.LANCZOS))
            tiles.append((base.convert("RGB"), label))
    refs = sheets(project, f"scribble-v{v:03d}", tiles)
    per_shot = [f"shot {k} · {_num(a / rate)}–{_num(b / rate)} s · {len(inside)} strokes"
                for k, (a, b, inside) in enumerate(cuts, 1)]
    table = "\n".join([f"# scribble · editplan v{v} · render {video.name}", *summary(resolved), *per_shot, *notes])
    return refs, table
