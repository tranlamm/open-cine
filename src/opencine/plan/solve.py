"""``plan_solve``: ShotPlan (intent) → draft EditPlan (frames), deterministically.

Per slot: the media (G3-approved clip, else the latest clip or still), a length inside ``dur_s`` by
role with cuts snapped to music beats, the best unused footage window, and items built from the
intent (transition, speed or impact ramp, text, sfx, template gains, a subject-tracking 9:16 crop).
Locked slots and items edited by hand since the last solve are kept as they are.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, NamedTuple

from opencine.core.time import Rate, frames_for, to_frame
from opencine.plan.schema import EditItem, EditPlan, Slot
from opencine.plan.templates import Template, template

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.model import SceneIndex
    from opencine.plan.schema import Brief, ShotPlan

SPEED = {"normal": 1.0, "fast": 1.5, "slow": 0.5, "ramp_slow_at_impact": 0.5}
# ShotPlan transition → (EditPlan kind, seconds); a camera whipping left slides the picture right
TRANSITION = {"dissolve": ("dissolve", 0.27), "flash": ("flash", 0.2), "dip": ("dip", 0.33),
              "whip": ("whip", 0.2), "whip_right": ("whip", 0.2), "whip_left": ("whip_left", 0.2),
              "zoom": ("zoom", 0.27), "zoom_out": ("zoom_out", 0.27), "mask": ("mask", 0.5),
              **{f"slide_{d}": (f"slide_{d}", 0.27) for d in ("left", "right", "up", "down")}}
MOTION_ROLES = {"hook", "build", "process", "climax"}
PHRASE = (1.0, 0.6, 1.4)  # along a run of one role: medium, short, long
STRONG_OUT = {"hook", "climax"}  # the cut out of these lands on a strong beat when one fits
STRONG_IN = {"payoff"}  # … and the cut into these
MOTION_WEIGHT, STILL_WEIGHT = 0.6, 0.15
REUSE_PENALTY, BAD_PENALTY = 0.5, 1.0
MATCH_WEIGHT = 1.0  # a perfect match is worth as much as a sharp, usable window
AT_SPAN = (0.25, 0.75)  # where `at` sits inside the window
GEN_HEAD_S = 0.3
RAMP_SLOW, RAMP_HOLD_S, RAMP_EASE_S = 0.35, 0.5, 0.2  # slow speed; source seconds held / eased
RAMP_LEAD = 0.45  # share of the shot before the impact
RAMP_SPEEDS = (0.25, 0.6)  # the slow speed may move inside this to make the length exact
MUSIC_FADE_S = 1.0
SFX_GAP_S = 0.5  # sounds closer than this: the less important one is dropped
SFX_PRIORITY = {"slot": 0, "payoff": 1, "title": 2, "push": 3, "text": 4}
PUSH_KINDS = {"whip", "whip_left", "slide_left", "slide_right", "slide_up", "slide_down"}
SOLVE_VERSION = 1


class SolveError(ValueError):
    pass


@dataclass
class _Slot:
    slot: Slot
    source: dict[str, Any]
    kind: str  # scene | gen | image
    avail: int | None  # source frames usable (None = a still: any length)
    speed: float
    lo: int  # output frames
    hi: int
    clip: str | None = None  # footage: clip id, for reuse tracking
    offset: int = 0  # footage: scene start in clip frames
    samples: list[dict] = field(default_factory=list)  # footage: scene-relative frames
    locked: dict[str, Any] | None = None  # base item kept as is
    length: int = 0
    weight: float = 1.0  # how far into [lo, hi] it leans (_weights)
    notes: list[str] = field(default_factory=list)
    edit: dict[str, Any] | None = None  # a video-to-video clip: its mode and footage window


def _speed(s: Slot) -> float:
    return SPEED[s.speed or "normal"]


def _gen_media(project: Project, sid: str) -> tuple[str, str] | None:
    """``(kind, path)`` a generated slot plays: approved clip > latest clip > approved/latest still."""
    from opencine.gen import slots

    for kind, d in (("video", project.clips_dir), ("image", project.stills_dir)):
        if rel := slots.approved(project, kind, sid):
            return kind, rel
        side = d / f"{sid}.json"
        if side.exists():
            try:
                rel = json.loads(side.read_text("utf-8")).get("path")
            except ValueError:
                rel = None
            if rel and (project.root / rel).is_file():
                return kind, rel
    return None


def _scene_samples(project: Project, index: SceneIndex, sid: str) -> list[dict]:
    """Samples of a scene, frames relative to its start; without ``index/samples``, one
    pseudo-sample per 0.5 s from the scene's own numbers, peaking at its motion peak."""
    from opencine.index.model import load_samples

    sc = index.scene(sid)
    rows = load_samples(project, sc.clip)
    if rows is not None:
        return [{**r, "f": r["f"] - sc.start_frame} for r in rows
                if sc.start_frame <= r["f"] < sc.end_frame]
    step = max(1, frames_for(0.5, Rate(index.rate)))
    peak = sc.motion_peak_frame
    return [{"f": f - sc.start_frame, "sharp": sc.sharpness, "bright": sc.brightness, "bad": False,
             "motion": min(1.0, sc.motion * (2.0 if peak is not None and abs(f - peak) < step else 1.0))}
            for f in range(sc.start_frame, sc.end_frame, step)]


def _resolve(project: Project, index: SceneIndex, sp: ShotPlan, rate: Rate) -> list[_Slot]:
    from opencine.plan.media import Resolver

    res, out, missing = Resolver(project, rate), [], []
    for s in sp.slots:
        speed = _speed(s)
        lo, hi = frames_for(s.dur_s[0], rate), frames_for(s.dur_s[1], rate)
        if s.source.footage is not None:
            sid = s.source.footage.scene.strip().upper()
            sc = index.scene(sid)
            src = {"scene": sid}
            x = _Slot(s, src, "scene", sc.end_frame - sc.start_frame, speed, lo, hi, clip=sc.clip,
                      offset=sc.start_frame, samples=_scene_samples(project, index, sid))
        else:
            found = _gen_media(project, str(s.slot))
            if found is None:
                missing.append(s.slot)
                continue
            kind, rel = found
            if kind == "video":
                from opencine.gen.source import edit_record

                src = {"gen": rel}
                m = res.media(EditItem.model_validate({"source": src, "in": 0, "out": 1}))
                rec = edit_record(project, rel)  # in step with its footage: no wobbly head
                head = 0 if rec else min(frames_for(GEN_HEAD_S, rate), max(0, (m.hard_frames or 0) - lo))
                x = _Slot(s, src, "gen", (m.hard_frames or 0) - head, speed, lo, hi, offset=head,
                          edit=rec)
            else:
                x = _Slot(s, {"image": rel}, "image", None, 1.0, lo, hi)
        if x.avail is not None:
            most = int(x.avail / x.speed)
            if most < x.lo:
                x.notes.append(f"media holds {most / rate.fps:.2f} s < dur_s min")
                x.lo = max(1, most)
            x.hi = max(x.lo, min(x.hi, most))
        out.append(x)
    if missing:
        raise SolveError(f"slot(s) {', '.join(map(str, missing))} have no generated clip or still "
                         "yet: run gen_image/gen_video/gen_edit (or review_decide picks) first")
    return out


# ---- lengths and beats -------------------------------------------------------------
def _weights(slots: list[_Slot], role_weight: dict[str, float]) -> None:
    """How far into its range each slot leans: by role (a climax burst short, a payoff held;
    the template's table), and varied along a run of one role (``PHRASE``) so equal ranges do
    not tick like a metronome."""
    run_role, j = object(), 0
    for x in slots:
        j = j + 1 if x.slot.role == run_role else 0
        run_role = x.slot.role
        x.weight = role_weight.get(x.slot.role or "", 1.0) * PHRASE[j % len(PHRASE)]


def _spread(slots: list[_Slot], target: int) -> list[int]:
    """Lengths ``lo + min(1, α·weight)·(hi − lo)`` in each slot's range, with the one α
    (bisection) that sums closest to ``target`` the ranges allow; rounding remainders go to
    the first slots with room."""
    if not slots:
        return []
    lo, hi = sum(s.lo for s in slots), sum(s.hi for s in slots)
    goal = min(max(target, lo), hi)

    def total(alpha: float) -> float:
        return sum(s.lo + min(1.0, alpha * s.weight) * (s.hi - s.lo) for s in slots)

    a, b = 0.0, 1.0 / min(s.weight for s in slots)  # total(b) == hi
    for _ in range(50):
        mid = (a + b) / 2
        a, b = (mid, b) if total(mid) < goal else (a, mid)
    out = [s.lo + int(min(1.0, a * s.weight) * (s.hi - s.lo)) for s in slots]
    gap = goal - sum(out)
    for i, s in enumerate(slots):
        if gap <= 0:
            break
        add = min(gap, s.hi - out[i])
        out[i] += add
        gap -= add
    return out


@dataclass(frozen=True)
class _Beats:
    frames: list[int]
    strong: set[int]


def _beats(project: Project, music: str | None, rate: Rate) -> _Beats | None:
    if not music:
        return None
    from opencine.music.beats import load_beats
    from opencine.music.library import beats_path, strong_threshold, track

    beats = load_beats(beats_path(track(project, music)))
    if not beats:
        return None
    thr = strong_threshold(beats)
    frames = sorted({to_frame(b.t, rate, "nearest") for b in beats})
    return _Beats(frames, {to_frame(b.t, rate, "nearest") for b in beats if b.strength >= thr})


def _lengths(slots: list[_Slot], target: int, beats: _Beats | None) -> list[int]:
    """Left to right: plan the rest, snap this slot's out-cut to a beat in its range."""
    n, start, lengths = len(slots), 0, []
    for k, x in enumerate(slots):
        if x.locked is not None:
            lengths.append(x.lo)
            start += x.lo
            continue
        rest = slots[k:]
        fixed = sum(s.lo for s in rest if s.locked is not None)
        free = [s for s in rest if s.locked is None]
        planned = _spread(free, target - start - fixed)[0]
        length = planned
        cut_on = x.slot.cut_on
        if beats and k < n - 1 and cut_on not in ("word", "free"):
            a, b = start + x.lo, start + x.hi
            inside = [f for f in beats.frames if a <= f <= b]
            nxt = slots[k + 1].slot.role
            strong = [f for f in inside if f in beats.strong]
            pool = strong if strong and (x.slot.role in STRONG_OUT or nxt in STRONG_IN) else inside
            if pool:
                cut = min(pool, key=lambda f: (abs(f - (start + planned)), f))
                length = cut - start
            else:
                x.notes.append("no beat inside dur_s")
        lengths.append(length)
        start += length
    return lengths


# ---- windows -----------------------------------------------------------------------
def _score(x: _Slot, used: dict[str, list[tuple[int, int]]], a: int, b: int) -> float:
    rows = [r for r in x.samples if a <= r["f"] < b]
    if not rows and x.samples:  # a window shorter than the sample step
        rows = [min(x.samples, key=lambda r: abs(r["f"] - (a + b) / 2))]
    w = MOTION_WEIGHT if x.slot.role in MOTION_ROLES else STILL_WEIGHT
    q = sum((r["sharp"] + w * r["motion"]) - (BAD_PENALTY if r["bad"] else 0) for r in rows)
    q = q / len(rows) if rows else 0.0
    ca, cb = x.offset + a, x.offset + b
    if any(min(cb, ub) - max(ca, ua) > 0 for ua, ub in used.get(x.clip or "", [])):
        q -= REUSE_PENALTY
    return q


def _window(x: _Slot, need: int, used: dict[str, list[tuple[int, int]]], rate: Rate,
            bonus: Callable[[int], float] | None = None) -> int:
    """Start frame (scene-relative) of the best ``need``-frame window; ``bonus(start)`` adds to
    a window's score (how well its first frame matches the previous slot's last)."""
    span = x.avail - need
    if span <= 0:
        return 0
    f = x.slot.source.footage
    at = None if f is None or f.at is None else to_frame(f.at, rate, "nearest")
    lo_s, hi_s = 0, span
    if at is not None:
        lo_s = min(span, max(0, at - int(need * AT_SPAN[1])))  # `at` near the scene end
        hi_s = min(span, max(lo_s, at - int(need * AT_SPAN[0])))
        centre = at - need // 2
    else:
        centre = span // 2
    best = max(range(lo_s, hi_s + 1),
               key=lambda s: (round(_score(x, used, s, s + need) + (bonus(s) if bonus else 0.0), 6),
                              -abs(s - centre), -s))
    return best


# ---- speed ramps --------------------------------------------------------------------
class _K(NamedTuple):
    at: int
    speed: float


def _ramp_keys(p: int, hold: int, ease: int, v: float, n: int | None = None) -> list[_K]:
    """Keys (source frames from ``in``) slowing to ``v`` for ``hold`` frames around offset ``p``,
    easing over ``ease`` frames each side; cut at 0 and at ``n`` (speeds interpolated there)."""
    a = p - hold // 2
    pts = [(a - ease, 1.0), (a, v), (a + hold, v), (a + hold + ease, 1.0)]
    pts = _cut(pts, 0, keep_after=True)
    if n is not None:
        pts = _cut(pts, n, keep_after=False)
    return [_K(f, round(sp, 3)) for f, sp in pts]


def _cut(pts: list[tuple[int, float]], at: int, *, keep_after: bool) -> list[tuple[int, float]]:
    """Keys on one side of ``at`` (inclusive), plus an interpolated key at ``at`` when the
    curve crosses it."""
    side = [(f, sp) for f, sp in pts if (f >= at if keep_after else f <= at)]
    other = [(f, sp) for f, sp in pts if (f < at if keep_after else f > at)]
    if not other or (side and side[0 if keep_after else -1][0] == at):
        return side
    o = other[-1] if keep_after else other[0]
    if not side:
        return [(at, o[1])]
    s = side[0] if keep_after else side[-1]
    v = o[1] + (s[1] - o[1]) * (at - o[0]) / (s[0] - o[0])
    return [(at, v), *side] if keep_after else [*side, (at, v)]


def _impact(x: _Slot, rate: Rate, used: dict[str, list[tuple[int, int]]]) -> tuple[int, str]:
    """Source frame (media-relative, before ``x.offset`` for generated clips) to slow at, and why."""
    f = x.slot.source.footage
    if f is not None and f.at is not None:
        return min(x.avail - 1, to_frame(f.at, rate, "nearest")), f"at {f.at:g} s"
    good = [r for r in x.samples if not r["bad"] and 0 <= r["f"] < x.avail]
    if good:
        taken = used.get(x.clip or "", [])
        fresh = [r for r in good if not any(ua <= x.offset + r["f"] < ub for ua, ub in taken)]
        r = max(fresh or good, key=lambda r: (r["motion"], -r["f"]))
        return r["f"], f"motion peak +{r['f'] / rate.fps:.1f} s"
    return x.avail // 2, "middle"


def _ramp_window(x: _Slot, out_len: int, used: dict[str, list[tuple[int, int]]],
                 rate: Rate) -> tuple[int, int, list[dict[str, Any]], str]:
    """``(in, out, keys, note)`` (media-relative) of a ramp lasting ``out_len`` output frames:
    the impact ~``RAMP_LEAD`` into the shot; the slow speed is nudged inside ``RAMP_SPEEDS`` so
    the length comes out exact (cut times and beats hold)."""
    from opencine.plan import ramp

    p, why = _impact(x, rate, used)
    hold, ease = frames_for(RAMP_HOLD_S, rate), frames_for(RAMP_EASE_S, rate)
    lead = RAMP_LEAD * out_len

    def keys(a: int, v: float, n: int | None = None) -> list[_K]:
        return _ramp_keys(p - a, hold, ease, v, n)

    a = min(range(max(0, p - 4 * out_len), p + 1),
            key=lambda a: (abs(ramp.time_at(keys(a, RAMP_SLOW), p - a) - lead), -a))
    n = ramp.source_for(keys(a, RAMP_SLOW), out_len, limit=x.avail - a)
    while a > 0 and ramp.time_at(keys(a, RAMP_SLOW), n) < out_len - 0.5:  # media ends: start earlier
        a -= 1
        n = ramp.source_for(keys(a, RAMP_SLOW), out_len, limit=x.avail - a)
    best = keys(a, RAMP_SLOW, n), n
    for m in (n, n - 1, n + 1, n - 2, n + 2):
        if m < 1 or a + m > x.avail:
            continue
        lo, hi = RAMP_SPEEDS
        for _ in range(40):  # the output shortens as the slow speed rises
            v = (lo + hi) / 2
            lo, hi = (v, hi) if ramp.time_at(keys(a, v, m), m) > out_len else (lo, v)
        k = keys(a, round((lo + hi) / 2, 3), m)
        if ramp.length(k, m) == out_len:
            best = k, m
            break
    k, n = best
    slow = min(q.speed for q in k)
    return a, a + n, [q._asdict() for q in k], f"ramp 1→{slow:g}→1 at {why}"


# ---- assembly ----------------------------------------------------------------------
def _sfx_file(project: Project, name: str) -> str | None:
    """Plan path of a sound: the project's / ``OPENCINE_SFX_DIR``'s file, else the synthesised one."""
    from opencine.music.sfx import find

    p = find(project, name)
    if p is None:
        return None
    try:
        return p.relative_to(project.root).as_posix()
    except ValueError:
        return str(p)


def _sfx_len(project: Project, rel: str, rate: Rate) -> int:
    from opencine.music.sfx import duration
    from opencine.plan.media import project_path

    return frames_for(duration(project_path(project, rel)), rate)


def _place_sfx(project: Project, events: list[tuple[str, int, str, int]], gain: float,
               rate: Rate, total: int) -> tuple[list[dict[str, Any]], list[str]]:
    """``events``: ``(kind, frame, name, slot)``. Files resolved, sounds within ``SFX_GAP_S``
    of a more important one dropped, the rest sorted by time."""
    gap = frames_for(SFX_GAP_S, rate)
    kept: list[tuple[str, int, str, int]] = []
    notes = []
    for ev in sorted(events, key=lambda e: (SFX_PRIORITY[e[0]], e[1])):
        if not 0 <= ev[1] < total:
            continue
        if any(abs(ev[1] - k[1]) < gap for k in kept):
            if ev[0] == "slot":
                notes.append(f"slot {ev[3]}: sfx {ev[2]} overlaps a more important sound")
            continue
        kept.append(ev)
    out = []
    for _, at, name, sid in sorted(kept, key=lambda e: e[1]):
        if f := _sfx_file(project, name):
            out.append({"file": f, "at": at, "gain": gain})
        else:
            notes.append(f"slot {sid}: no sfx/{name}.* file and no synthesised '{name}'")
    return out, notes


def _auto_sfx(project: Project, tpl: Template, slots: list[_Slot], items: list[dict[str, Any]],
              cuts: list[int], rate: Rate) -> list[tuple[str, int, str, int]]:
    """Template sounds: a whoosh centred on each whip/slide, a pop on a text card, a riser into
    a title, an impact on the cut into the first payoff. A slot with ``sfx: "none"`` gets none."""
    events: list[tuple[str, int, str, int]] = []
    payoff_done = False
    for k, (x, item) in enumerate(zip(slots, items, strict=True)):
        if (x.slot.sfx or "").lower() == "none":
            continue
        sid, cut = x.slot.slot, cuts[k]
        tr = item.get("transition_in") or {}
        if (name := tpl.auto_sfx.get("push")) and tr.get("kind") in PUSH_KINDS and tr.get("frames"):
            f = _sfx_file(project, name)
            n = _sfx_len(project, f, rate) if f else 0
            events.append(("push", max(0, cut - tr["frames"] // 2 - n // 2), name, sid))
        text = item.get("text")
        if text and text.get("style") != "time":  # a clock stays quiet
            at = cut + max(0, text.get("start", 0) - (tr.get("frames") or 0))
            riser = tpl.auto_sfx.get("title") if text.get("style") == "title" else None
            f = _sfx_file(project, riser) if riser else None
            n = _sfx_len(project, f, rate) if f else 0
            if riser and f and at - n >= 0:
                events.append(("title", at - n, riser, sid))
            elif name := tpl.auto_sfx.get("text"):
                events.append(("text", at, name, sid))
        if x.slot.role == "payoff" and not payoff_done and (name := tpl.auto_sfx.get("payoff")) and k:
            events.append(("payoff", cut, name, sid))
            payoff_done = True
    return events


def _match_bonus(project: Project, index: SceneIndex, prev: _Slot | None, prev_item: dict | None,
                 x: _Slot, descs: list | None) -> Callable[[int], float] | None:
    """When the previous slot asks for a match cut into this one: start → weighted score of
    this slot's frame there against the previous item's last frame (footage on both sides)."""
    if prev is None or prev.slot.match is None or not descs or prev_item is None             or "scene" not in prev_item["source"] or x.clip is None:
        return None
    from opencine.plan.match import score

    kind = prev.slot.match.kind
    if kind == "sound":
        return None
    sc = index.scene(prev_item["source"]["scene"])
    near = round(project.rate.fps / 2)

    def at(clip: str, f: int):
        pool = [d for d in descs if d.clip == clip and abs(d.f - f) <= near]
        return min(pool, key=lambda d: abs(d.f - f)) if pool else None

    last = at(sc.clip, sc.start_frame + prev_item["out"] - 1)
    if last is None:
        return None

    def bonus(start: int) -> float:
        d = at(x.clip, x.offset + start)
        return MATCH_WEIGHT * score(last, d, kind) if d is not None else 0.0

    return bonus


def _speaks(trs: dict[str, Any], clip: str | None, a: int, b: int, rate: Rate) -> bool:
    """Someone speaks during clip frames ``[a, b)`` (a transcribed word overlaps them)."""
    t0, t1 = a / rate.fps, b / rate.fps
    for seg in (trs.get(clip or "") or {}).get("segments") or []:
        for w in seg.get("words") or [seg]:
            if w.get("start") is not None and w.get("end") is not None and w["start"] < t1 and w["end"] > t0:
                return True
    return False


def _item_hash(item: dict[str, Any]) -> str:
    keep = {k: v for k, v in item.items() if k != "note"}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def record_path(project: Project, version: int) -> Path:
    return project.plans_dir / f"editplan.v{version:03d}.solve.json"


def _base_locks(project: Project, base: int | None, lock: set[int]) -> tuple[dict[int, dict], list[str]]:
    """Base-plan items to keep, by slot: the ``lock`` slots and items edited by hand since
    the solve that produced them (the newest solve record at or before ``base``)."""
    from opencine.plan.store import load_plan, versions

    have = versions(project, "editplan")
    if not have:
        if lock:
            return {}, [f"lock {sorted(lock)} ignored: no EditPlan to keep slots from"]
        return {}, []
    v = base if base is not None else have[-1]
    _, plan = load_plan(project, "editplan", v)
    items = {it.slot: it.model_dump(mode="json", by_alias=True) for it in plan.items
             if it.slot is not None}
    rec = next((json.loads(record_path(project, r).read_text("utf-8"))
                for r in sorted(have, reverse=True) if r <= v and record_path(project, r).exists()),
               None)
    hashes = (rec or {}).get("items", {})
    keep, notes = {}, []
    for sid, item in items.items():
        edited = str(sid) in hashes and hashes[str(sid)] != _item_hash(item)
        if sid in lock or edited:
            keep[sid] = item
            if edited and sid not in lock:
                notes.append(f"slot {sid}: edited by hand in v{v}, kept")
    for sid in sorted(lock - set(items)):
        notes.append(f"lock {sid}: no item with slot {sid} in v{v}")
    return keep, notes


def solve(project: Project, sp: ShotPlan, brief: Brief | None = None, *, lock: list[int] | None = None,
          base: int | None = None, target_s: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(EditPlan dict, report)`` for ``sp``; nothing is saved."""
    from opencine.index.model import load_index

    rate = project.rate
    index = load_index(project)
    slots = _resolve(project, index, sp, rate)
    keep, lock_notes = _base_locks(project, base, set(lock or []))
    for x in slots:
        if (item := keep.get(x.slot.slot)) is not None:
            x.locked = item
            it = EditItem.model_validate(item)  # its cut-to-cut length: without the overlap
            x.lo = x.hi = max(1, it.length - (0 if it.is_cut else it.transition_in.frames))
    if target_s is None and brief is not None:
        target_s = brief.target_s or sum(brief.duration_s) / 2
    target = frames_for(target_s, rate) if target_s else sum((s.lo + s.hi) // 2 for s in slots)
    from opencine.plan.taste import adjust

    tpl, taste_notes = adjust(template(sp.template))  # learned from approved projects
    _weights(slots, tpl.role_weight)
    music = sp.music or (brief.music if brief else None)
    beats = _beats(project, music, rate)
    lengths = _lengths(slots, target, beats)
    gap = median(b - a for a, b in zip(beats.frames, beats.frames[1:], strict=False)) \
        if beats and len(beats.frames) > 1 else None

    from opencine.speech.transcribe import load_transcripts

    trs = load_transcripts(project) if tpl.speech_db is not None else {}
    descs = None
    if any(x.slot.match for x in slots):
        from opencine.plan.match import describe

        descs = describe(project, index)
    used: dict[str, list[tuple[int, int]]] = {}
    for x in slots:  # locked footage is taken first, so others avoid it
        if x.locked is not None and "scene" in x.locked["source"] and x.clip:
            used.setdefault(x.clip, []).append((x.offset + x.locked["in"], x.offset + x.locked["out"]))
    items, events, rows, cuts, start = [], [], [], [], 0
    for k, (x, length) in enumerate(zip(slots, lengths, strict=True)):
        x.length = length
        s = x.slot
        prev = slots[k - 1] if k else None
        new_chapter = s.chapter is not None and (prev is None or prev.slot.chapter != s.chapter)
        if x.locked is not None:
            item = dict(x.locked)
        else:
            tr_frames, tr_kind = 0, "cut"
            out = prev.slot.transition_out if prev is not None else None
            if out is None and prev is not None and new_chapter and prev.slot.chapter is not None:
                out = tpl.chapter_transition  # the template marks the change of chapter
            if out not in (None, "cut"):
                tr_kind, secs = TRANSITION[out]
                tr_frames = frames_for(secs, rate)
                if gap:
                    tr_frames = min(tr_frames, gap // 2)
                # the previous item's free tail is its cut-to-cut length
                tr_frames = max(0, min(tr_frames, lengths[k - 1] - 1, length - 1))
            if x.avail is not None:  # the overlap must fit in the media too
                tr_frames = max(0, min(tr_frames, int(x.avail / x.speed) - length))
            out_len = length + tr_frames  # the overlap starts before the cut; cut times hold
            need = max(1, min(round(out_len * x.speed), x.avail or out_len))
            keys = None
            if s.speed == "ramp_slow_at_impact" and x.kind != "image":
                a, b, keys, note = _ramp_window(x, out_len, used, rate)
                x.notes.append(note)
                if x.kind == "scene":
                    used.setdefault(x.clip, []).append((x.offset + a, x.offset + b))
                i0, i1 = x.offset * (x.kind == "gen") + a, x.offset * (x.kind == "gen") + b
            elif x.kind == "scene":
                bonus = _match_bonus(project, index, prev, items[-1] if items else None, x, descs)
                a = _window(x, need, used, rate, bonus)
                if bonus is not None:
                    x.notes.append(f"match {prev.slot.match.kind} {bonus(a) / MATCH_WEIGHT:.2f} "
                                   f"from slot {prev.slot.slot}")
                used.setdefault(x.clip, []).append((x.offset + a, x.offset + a + need))
                i0, i1 = a, a + need
            elif x.kind == "gen":
                i0, i1 = x.offset, x.offset + need
            else:
                i0, i1 = 0, out_len
            item = {"source": x.source, "in": i0, "out": i1, "slot": s.slot,
                    "note": f"solver: {s.purpose}"[:200]}
            if keys:
                item["ramp"] = keys
            elif x.speed != 1.0:
                item["speed"] = x.speed
            if s.crop is not None and x.kind == "scene":
                item["crop"] = s.crop.model_dump()
            elif x.kind == "scene" and s.reframe != "center":
                from opencine.plan.reframe import reframe

                crop = reframe(project, x.clip, x.offset, i0, i1, out_len, speed=x.speed,
                               ramp=[_K(**k) for k in keys] if keys else None)
                if crop is not None:
                    item["crop"] = crop
                    x.notes.append("reframed: crop follows the subject" if isinstance(crop, list)
                                   else "reframed: crop on the subject")
            if tr_frames:
                item["transition_in"] = {"kind": tr_kind, "frames": tr_frames}
            clip_info = index.clip(x.clip) if x.kind == "scene" else None
            text = None
            if s.text:
                style, anim = tpl.text_for(s.role)
                text = {"content": s.text, "position": "auto", "style": style, "anim": anim}
            elif tpl.chapter_time and new_chapter and clip_info is not None and clip_info.shot_at:
                text = {"content": "{time}", "position": "auto", "style": "time", "anim": "type"}
            if text:
                if tr_frames:
                    text["start"] = tr_frames
                item["text"] = text
            if x.kind == "scene":
                speaks = tpl.speech_db is not None and _speaks(
                    trs, x.clip, x.offset + i0, x.offset + i1, rate)
                item["gain"] = tpl.speech_db if speaks else tpl.footage_db
            elif x.edit is not None and x.edit["source"].get("scene"):
                # the footage window's own sound, from the moment the picture shows
                w = x.edit["source"]
                item["sound"] = {"scene": w["scene"],
                                 "in": to_frame(float(w.get("in_s", 0)), rate, "nearest") + i0}
                item["gain"] = tpl.footage_db
                if x.edit["mode"] == "restyle":
                    item["color"] = "none"  # its colours are the new look, not a drift
                x.notes.append(f"{x.edit['mode']}: plays {w['scene']}'s sound in step")
            elif x.kind == "gen" and tpl.gen_db is not None:
                item["gain"] = tpl.gen_db  # the clip's own soundtrack is the scene's sound
        items.append(item)
        cuts.append(start)
        if s.sfx and s.sfx.lower() != "none":
            at = start + length if k < len(slots) - 1 else start
            events.append(("slot", max(0, at - frames_for(0.1, rate)), s.sfx, s.slot))
        rows.append(x)
        start += length
    events += _auto_sfx(project, tpl, slots, items, cuts, rate)
    sfx, sfx_notes = _place_sfx(project, events, tpl.sfx_db, rate, start)

    plan: dict[str, Any] = {"rate": str(rate), "deliverable": sp.deliverable or tpl.deliverable,
                            "items": items}
    if sp.letterbox:
        plan["letterbox"] = sp.letterbox
    if music:
        from opencine.music.library import track

        plan["music"] = {"file": track(project, music).relative_to(project.root).as_posix(),
                         "gain": tpl.music_db, "fade_out": frames_for(MUSIC_FADE_S, rate)}
    if sfx:
        plan["sfx"] = sfx
    if tpl.look:
        plan["look"] = {"name": tpl.look}
    built = EditPlan.model_validate(plan)
    starts = built.starts()
    cuts = [starts[i] + (built.items[i].transition_in.frames if not built.items[i].is_cut else 0)
            for i in range(1, len(starts))]
    on_beat = [min((abs(c - b) for b in beats.frames), default=99) <= 2 for c in cuts] if beats else []
    total = sum(lengths)
    report = {
        "target_s": round(target / rate.fps, 2),
        "total_s": round(total / rate.fps, 2),
        "cuts_on_beat": f"{sum(on_beat)}/{len(cuts)}" if beats else "no music",
        "slots": [_row(x, it, st, rate) for x, it, st in zip(rows, built.items, starts, strict=True)],
        "notes": lock_notes + taste_notes + [f"slot {x.slot.slot}: {n}" for x in rows for n in x.notes]
        + sfx_notes,
    }
    return plan, report


def _row(x: _Slot, it: EditItem, start: int, rate: Rate) -> str:
    fps = rate.fps
    src = next(iter(x.source.values()))
    span = f"{it.in_ / fps:.2f}–{it.out / fps:.2f} s" if x.kind != "image" else "still"
    tag = "locked" if x.locked is not None else x.slot.role or ""
    return (f"#{x.slot.slot} {tag} · {src} {span} · {x.length / fps:.2f} s · "
            f"@{start / fps:.2f} s").replace(" ·  ·", " ·")


def solve_and_save(project: Project, version: int | None = None, *, lock: list[int] | None = None,
                   base: int | None = None, target_s: float | None = None) -> dict[str, Any]:
    """Solve a ShotPlan version (latest by default) and save the draft as a new EditPlan."""
    from opencine.plan.store import load_plan, save_plan, versions

    v, sp = load_plan(project, "shotplan", version)
    brief = load_plan(project, "brief")[1] if versions(project, "brief") else None
    plan, report = solve(project, sp, brief, lock=lock, base=base, target_s=target_s)
    saved = save_plan(project, "editplan", plan)
    _, stored = load_plan(project, "editplan", saved["version"])
    rec = {"solver": SOLVE_VERSION, "shotplan": v, "lock": sorted(lock or []),
           "items": {str(it.slot): _item_hash(it.model_dump(mode="json", by_alias=True))
                     for it in stored.items if it.slot is not None}, "report": report}
    record_path(project, saved["version"]).write_text(json.dumps(rec, indent=2, ensure_ascii=False), "utf-8")
    return {**saved, "shotplan": v, "report": report}

