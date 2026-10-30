"""Semantic checks for plans (pydantic already enforces types and ranges); with a project, sources
are resolved too (ids, files, in/out within media, crop aspect, punch-in headroom).

``craft_hints`` never block: editing-craft hints ``plan_save`` returns so the Director can fix
them before rendering (hook length, rhythm, jump cuts, text reading time, cuts inside words…).
"""

from __future__ import annotations

import json
import re
import wave
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from opencine.core.ffmpeg import probe_json
from opencine.core.project import Project
from opencine.core.time import Rate, fmt_ts, to_frame
from opencine.gen.source import edit_record
from opencine.plan.media import Media, Resolver, cover, project_path, zoom_keys
from opencine.plan.schema import (
    ClipSource,
    ColorMatch,
    CropBox,
    EditItem,
    EditPlan,
    GenSource,
    SceneSource,
)

if TYPE_CHECKING:
    from opencine.index.model import SceneIndex
    from opencine.plan.schema import Brief, ShotPlan

MAX_TOTAL_S = 180
MAX_MAGNIFY = 1.05  # output px per original source px before warning
SPEED_RANGE = (0.25, 4.0)

# craft hints (rules of thumb; the why is in the edit skill's editing.md)
HOOK_MAX_S = 2.0  # the first seconds decide whether people stay
SHOT_MIN_S, SHOT_MAX_S = 0.3, 5.0
SAME_LEN, SAME_RUN = 0.10, 3  # 3+ shots in a row within ±10 % of each other read as a metronome
JUMP_GAP_S = 1.0  # same clip, same framing, ≤ this much later: a jump cut
SOUND_SKIP_S = 0.5  # multicam: a smaller jump in the take's sound at a cut is a slip, not a trim
REPEAT_MIN_S = 0.2
TEXT_WPS, TEXT_MIN_S, TEXT_WORDS = 3.0, 1.0, 8  # reading speed, shortest card, one idea per card
GEN_SKIP_S = 0.3  # generated clips often wobble in their first frames
VO_GAP_S = {False: 2.0, True: 4.0}  # longest pause between voice-over lines: without / with music
VO_OVERLAP_S = 0.05
EMPHASIS_MAX = 0.15  # share of caption words in the accent colour
WORD_EDGE_S = 0.03
BEAT_TOL = 2  # frames
MASK_MIN_FRAMES = 8
# a film (film project or ShotPlan.template "film", skill film) is paced like one: a long
# opening is fine, a shot may be a whole H3 clip, the timeline runs to minutes
FILM_SHOT_MAX_S = 15.5
FILM_MAX_TOTAL_S = 600


def is_film(project: Project | None = None, template: str | None = None) -> bool:
    return bool(getattr(project, "film", False)) or (template or "").strip().lower() == "film"


class Issue(BaseModel):
    level: Literal["error", "warning", "hint"]
    path: str
    message: str


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]


def validate(plan: EditPlan, project: Project | None = None) -> list[Issue]:
    issues: list[Issue] = []

    def add(level: Literal["error", "warning"], path: str, msg: str) -> None:
        issues.append(Issue(level=level, path=path, message=msg))

    if not plan.items:
        add("error", "items", "plan has no items")
        return issues
    rate = plan.fps
    for i, it in enumerate(plan.items):
        _check_item(plan, i, it, add)
    _check_timeline(plan, rate, add, FILM_MAX_TOTAL_S if is_film(project) else MAX_TOTAL_S)
    if project is not None:
        _check_project(plan, project, add)
    return issues


def _check_item(plan: EditPlan, i: int, it: EditItem, add) -> None:
    p = f"items[{i}]"
    if it.out <= it.in_:
        add("error", f"{p}.out", f"out ({it.out}) must be greater than in ({it.in_})")
    if not SPEED_RANGE[0] <= it.speed <= SPEED_RANGE[1]:
        add("warning", f"{p}.speed", f"speed {it.speed} is outside {SPEED_RANGE}; motion will look odd")
    if it.ramp is not None:
        ats = [k.at for k in it.ramp]
        if not it.ramp:
            add("error", f"{p}.ramp", "empty ramp; use null for a constant speed")
        elif any(b <= a for a, b in zip(ats, ats[1:], strict=False)):
            add("error", f"{p}.ramp", f"ramp 'at' values must strictly increase: {ats}")
        elif ats[-1] > it.out - it.in_:
            add("warning", f"{p}.ramp", f"ramp key at {ats[-1]} is past the item's {it.out - it.in_} "
                                        "source frames (keys count source frames from `in`)")
        if any(not SPEED_RANGE[0] <= k.speed <= SPEED_RANGE[1] for k in it.ramp):
            add("warning", f"{p}.ramp", f"ramp speeds {[k.speed for k in it.ramp]} leave {SPEED_RANGE}")
        if it.speed != 1:
            add("warning", f"{p}.speed", "speed is ignored when a ramp is given")
    if abs(it.gain or 0) > 12:
        add("warning", f"{p}.gain", f"gain {it.gain} dB is extreme")
    if (it.audio_lead or it.audio_tail) and (it.gain is None or (it.ramp and it.sound is None)):
        add("warning", f"{p}.audio_lead" if it.audio_lead else f"{p}.audio_tail",
            "J/L-cut on an item whose sound is muted (gain null or a ramp) does nothing")
    if it.sound is not None and it.gain is None:
        add("warning", f"{p}.sound", "gain null mutes the sound; give a gain (0 = as recorded)")
    if i == 0 and it.audio_lead:
        add("warning", f"{p}.audio_lead", "the first item has nothing to lead under")

    tr, n = it.transition_in, it.length
    if i == 0 and not it.is_cut:
        add("warning", f"{p}.transition_in", "first item has nothing to transition from; treated as cut")
    elif tr.kind != "cut" and tr.frames == 0:
        add("warning", f"{p}.transition_in.frames", f"{tr.kind} with 0 frames is a hard cut")
    elif tr.kind == "cut" and tr.frames:
        add("warning", f"{p}.transition_in.frames", "frames are ignored for a cut")
    elif i and not it.is_cut:
        if tr.kind == "mask" and tr.frames < MASK_MIN_FRAMES:
            add("warning", f"{p}.transition_in.frames", f"a mask wipe over {tr.frames} frames is too "
                                                        f"quick to read; give it ≥ {MASK_MIN_FRAMES}")
        prev = plan.items[i - 1]
        prev_free = prev.length - (0 if (i == 1 or prev.is_cut) else prev.transition_in.frames)
        if tr.frames >= n:
            add("error", f"{p}.transition_in.frames", f"transition ({tr.frames}) must be shorter than this item ({n} frames)")
        if tr.frames > prev_free:
            add("error", f"{p}.transition_in.frames",
                f"transition ({tr.frames}) is longer than the free tail of item {i - 1} ({prev_free} frames)")

    boxes = [it.crop] if isinstance(it.crop, CropBox) else (it.crop or [])
    for k, b in enumerate(boxes):
        bp = f"{p}.crop" if isinstance(it.crop, CropBox) else f"{p}.crop[{k}]"
        if b.x + b.w > 1 + 1e-9 or b.y + b.h > 1 + 1e-9:
            add("error", bp, f"crop box ({b.x}, {b.y}, {b.w}, {b.h}) goes outside the source frame")
    if isinstance(it.crop, list):
        _check_keys(f"{p}.crop", [k.at for k in it.crop], n, add)
        if not it.crop:
            add("warning", f"{p}.crop", "empty keyframe list; use null for the default crop")
    if it.zoom is not None:
        _check_keys(f"{p}.zoom", [k.at for k in it.zoom], n, add)

    if it.text is not None:
        t, tp = it.text, f"{p}.text"
        if not t.content.strip():
            add("warning", f"{tp}.content", "empty text")
        end = n if t.end is None else t.end
        if t.start >= n:
            add("error", f"{tp}.start", f"text starts at {t.start}, after the item ends ({n})")
        elif end <= t.start:
            add("error", tp, f"text end ({end}) must be after start ({t.start})")
        elif end > n:
            add("warning", f"{tp}.end", f"text end {end} is past the item end ({n}); clipped")


def _check_keys(path: str, ats: list[int], n: int, add) -> None:
    if any(b <= a for a, b in zip(ats, ats[1:], strict=False)):
        add("error", path, f"keyframe 'at' values must strictly increase: {ats}")
    if ats and ats[-1] >= n:
        add("warning", path, f"keyframe at {ats[-1]} is past the item end ({n})")


def _check_timeline(plan: EditPlan, rate: Rate, add, max_s: float = MAX_TOTAL_S) -> None:
    total = plan.total_frames
    secs = float(Fraction(total) / rate)
    if secs > max_s:
        add("warning", "items", f"timeline is {secs:.1f} s (> {max_s:g} s)")
    elif secs < 1:
        add("warning", "items", f"timeline is only {secs:.2f} s")
    if plan.music is not None:
        m = plan.music
        if m.fade_in + m.fade_out > total:
            add("warning", "music", f"fade_in + fade_out ({m.fade_in + m.fade_out}) exceed the timeline ({total} frames)")
        if abs(m.gain) > 12:
            add("warning", "music.gain", f"gain {m.gain} dB is extreme")
    for k, s in enumerate(plan.sfx):
        if s.at >= total:
            add("error", f"sfx[{k}].at", f"sfx at frame {s.at} is past the timeline end ({total})")
        if abs(s.gain) > 12:
            add("warning", f"sfx[{k}].gain", f"gain {s.gain} dB is extreme")


def _unknown(project: Project, e: KeyError) -> str:
    why = f"unknown id: {e.args[0] if e.args else e}"
    if not project.meta.sources:  # an empty index: a film, or footage not sent yet
        why += "; the project has no videos (nothing ingested): play generated clips or stills"
    return why


def _check_project(plan: EditPlan, project: Project, add) -> None:
    rate = plan.fps
    if rate != project.rate:
        add("warning", "rate", f"plan rate {rate} differs from project rate {project.rate}; frames will be resampled")
    res = Resolver(project, rate)
    out_w, out_h = plan.size
    for i, it in enumerate(plan.items):
        p = f"items[{i}]"
        if it.sound is not None:
            _check_sound(p, it, res, add)
        try:
            m = res.media(it)
        except FileNotFoundError as e:
            add("error", f"{p}.source", f"file not found: {e.args[0] if e.args else e}")
            continue
        except KeyError as e:
            add("error", f"{p}.source", _unknown(project, e))
            continue
        except ValueError as e:
            add("error", f"{p}.source", str(e))
            continue
        _check_media(p, it, m, out_w, out_h, add)
        if isinstance(it.color, ColorMatch):
            sid = it.color.match.strip().upper().partition("@")[0]
            try:
                res.index.scene(sid)
            except KeyError:
                add("error", f"{p}.color.match", f"unknown scene {sid!r}")

    total_s = Fraction(plan.total_frames) / rate
    files = ([("music.file", plan.music.file)] if plan.music else []) + [
        (f"sfx[{k}].file", s.file) for k, s in enumerate(plan.sfx)]
    for path, f in files:
        fp = project_path(project, f)
        if not fp.is_file():
            add("error", path, f"file not found: {fp}")
        elif path == "music.file":
            dur = Fraction(str(probe_json(fp)["format"].get("duration", "0")))
            if dur - Fraction(plan.music.offset) / rate < total_s:
                add("warning", path, f"music ({float(dur):.1f} s from offset) is shorter than the timeline ({float(total_s):.1f} s); the end is silent")


def _check_sound(p: str, it: EditItem, res: Resolver, add) -> None:
    try:
        m = res.sound(it)
    except FileNotFoundError as e:
        add("error", f"{p}.sound", f"file not found: {e.args[0] if e.args else e}")
        return
    except KeyError as e:
        add("error", f"{p}.sound", _unknown(res.project, e))
        return
    end = it.sound.in_ + it.length
    if m.hard_frames is not None and end > m.hard_frames:
        add("error", f"{p}.sound", f"the sound runs to frame {end}, past the end of the media "
                                   f"({m.hard_frames} frames)")
    elif m.soft_frames is not None and end > m.soft_frames:
        add("warning", f"{p}.sound", f"the sound runs to frame {end}, past the scene end "
                                     f"({m.soft_frames} frames)")
    if m.audio is None:
        add("warning", f"{p}.sound", "that footage has no audio track; the item is silent")


def _check_media(p: str, it: EditItem, m: Media, out_w: int, out_h: int, add) -> None:
    if it.layout == "fit_blur":  # framed as a 9:16 column of the output
        out_w = max(2, round(out_h * 9 / 16 / 2) * 2)
    if m.hard_frames is not None and it.out > m.hard_frames:
        add("error", f"{p}.out", f"out ({it.out}) is past the end of the media ({m.hard_frames} frames)")
    elif m.soft_frames is not None and it.out > m.soft_frames:
        what = "scene" if isinstance(it.source, SceneSource) else "source"
        add("warning", f"{p}.out", f"out ({it.out}) runs past the {what} end ({m.soft_frames} frames)")

    fw, fh = m.full_width, m.full_height  # final renders decode the original at the deliverable size
    boxes = [it.crop] if isinstance(it.crop, CropBox) else (it.crop or [None])
    aspect = out_w / out_h
    for b in boxes:
        if b is not None:
            box_aspect = (b.w * fw) / (b.h * fh)
            if abs(box_aspect / aspect - 1) > 0.02:
                add("warning", f"{p}.crop", f"crop aspect {box_aspect:.3f} ≠ deliverable {aspect:.3f}; it will be centre-trimmed")
                break
    if it.text is not None and m.shot_at is None and _PLACEHOLDER.search(it.text.content):
        add("warning", f"{p}.text.content", "{time}/{date} need a clip with a shooting time "
                                            "(L0 'shot'); they render empty here")
    magnify, window = _magnify(it, m, out_w, out_h)
    if magnify > MAX_MAGNIFY:
        add("warning", f"{p}.zoom" if it.zoom else f"{p}.transition_in" if it.transition_in.kind == "zoom_out" else f"{p}.crop",
            f"upscales the source {magnify:.2f}× (window {window:.0f} px → {out_w} px); expect softness")


def _at(keys: list[tuple[int, float]], t: int) -> float:
    """Linear interpolation of ``(at, value)`` keys at ``t``, held outside them (as rendered)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (a, va), (b, vb) in zip(keys, keys[1:], strict=False):
        if t < b:
            return va + (vb - va) * (t - a) / (b - a)
    return keys[-1][1]


def _magnify(it: EditItem, m: Media, out_w: int, out_h: int) -> tuple[float, float]:
    """Worst ``(output px per original px, source window px)`` over the item, crop and zoom
    combined. The covered crop width and the zoom are each linear between keys, so their ratio
    is monotonic between consecutive keys of either: checking every key time is exact."""
    def width(b: CropBox | None) -> float:
        box = None if b is None else (b.x, b.y, b.w, b.h)
        return cover(box, m.full_width, m.full_height, out_w, out_h)[2]

    if isinstance(it.crop, list) and it.crop:
        crop = [(k.at, width(k)) for k in it.crop]
    else:
        crop = [(0, width(it.crop if isinstance(it.crop, CropBox) else None))]
    zoom = [(a, sc) for a, sc, _, _ in zoom_keys(it, m.kind)]
    times = sorted({a for a, _ in crop} | {a for a, _ in zoom})
    window = min(_at(crop, t) / _at(zoom, t) for t in times)
    return out_w / window, window



# ---- craft hints (never block) -------------------------------------------------------
def _edit_sound_hint(i: int, it: EditItem, rec: dict, rate: Rate, hint) -> None:
    """A video-to-video clip runs in step with its footage window: its item should play that
    window's sound from the same moment (``sound``), or the picture drifts from what is heard."""
    src = rec["source"]
    if not src.get("scene"):
        return
    want = to_frame(float(src.get("in_s", 0)), rate, "nearest") + it.in_
    at = f'{{"scene": "{src["scene"]}", "in_s": {float(want / rate):.2f}}}'
    if it.sound is None:  # the clip has no sound of its own
        hint(f"items[{i}].sound", f"{rec['mode']} clip of {src['scene']} is silent; play the "
                                  f"footage in step: sound {at}")
    elif it.sound.scene != src["scene"] or abs(it.sound.in_ - want) > 1:
        hint(f"items[{i}].sound", f"{rec['mode']} clip and its sound are out of step; the picture "
                                  f"at in shows {src['scene']}@{float(want / rate):.2f}: sound {at}")


def craft_hints(plan: EditPlan, project: Project | None = None,
                brief: Brief | None = None) -> list[Issue]:
    """Editing-craft hints for a structurally valid plan; all ``level="hint"``."""
    issues: list[Issue] = []

    def hint(path: str, msg: str) -> None:
        issues.append(Issue(level="hint", path=path, message=msg))

    if not plan.items:
        return issues
    rate = plan.fps
    secs = [float(Fraction(it.length) / rate) for it in plan.items]
    film = is_film(project)
    shot_max = FILM_SHOT_MAX_S if film else SHOT_MAX_S
    if secs[0] > HOOK_MAX_S and not film:
        hint("items[0]", f"first shot lasts {secs[0]:.1f} s; open on the strongest moment and cut "
                         f"by ~{HOOK_MAX_S:g} s (the first seconds decide whether people stay)")
    for i, s in enumerate(secs):
        if s < SHOT_MIN_S:
            hint(f"items[{i}]", f"{s:.2f} s is a flash; fine for a burst, otherwise ≥ {SHOT_MIN_S:g} s")
        elif s > shot_max:
            hint(f"items[{i}]", f"{s:.1f} s shot; trim, split it with an insert or speed it up")
    i = 0
    while i < len(secs):
        j = i
        while j + 1 < len(secs) and abs(secs[j + 1] - secs[i]) <= SAME_LEN * secs[i]:
            j += 1
        if j - i + 1 >= SAME_RUN:
            hint(f"items[{i}]", f"items {i}–{j} all last ~{secs[i]:.1f} s: vary the rhythm "
                                "(a phrase of short shots, then a held one)")
        i = j + 1

    index = _index(project)
    spans = [_span(it, index, rate) for it in plan.items]
    for i in range(1, len(plan.items)):
        a, b, it, prev = spans[i - 1], spans[i], plan.items[i], plan.items[i - 1]
        if (a and b and a[0] == b[0] and it.is_cut and it.crop == prev.crop and not it.zoom
                and not prev.zoom and 0 <= b[1] - a[2] < JUMP_GAP_S * rate):
            if b[1] == a[2]:
                hint(f"items[{i}]", f"continues items[{i - 1}] with the same framing: an invisible "
                                    "cut; merge the items, or change size/angle/crop")
            else:
                hint(f"items[{i}]", f"jump cut: same shot as items[{i - 1}] "
                                    f"{float((b[1] - a[2]) / rate):.2f} s later, same framing; change "
                                    "crop/size, insert another shot, or make it a deliberate punch-in")
    # the footage frames each item is heard from: its `sound`, else its own source
    heard = [_sound_span(it, index, rate) if it.sound else span for it, span in zip(plan.items, spans, strict=True)]
    for i in range(1, len(plan.items)):
        it, prev = plan.items[i], plan.items[i - 1]
        a, b = heard[i - 1], heard[i]
        real = all(x.sound is not None or (x.speed == 1 and not x.ramp) for x in (it, prev))
        if not ((it.sound or prev.sound) and it.is_cut and real and a and b and a[0] == b[0]
                and it.gain is not None and prev.gain is not None):
            continue
        d = b[1] - a[2]
        if d < 0 or 0 < d < SOUND_SKIP_S * rate:
            s = float(abs(d) / rate)
            hint(f"items[{i}].sound" if it.sound else f"items[{i}].in",
                 f"the take's sound {'repeats' if d < 0 else 'skips'} {s:.2f} s at this cut; start "
                 f"it {s:.2f} s {'later' if d < 0 else 'earlier'} to run on without a gap, or trim "
                 "more on purpose")
    if project is not None and index is not None:
        _subject_hints(plan, project, index, rate, hint)
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a and b and a[0] == b[0] and min(a[2], b[2]) - max(a[1], b[1]) > REPEAT_MIN_S * rate:
                hint(f"items[{j}]", f"repeats footage already used in items[{i}]")
                break

    for i, it in enumerate(plan.items):
        if it.text is not None and it.text.content.strip():
            words = len(it.text.content.split())
            n = it.length
            shown = float(Fraction(min(n, it.text.end if it.text.end is not None else n) - it.text.start) / rate)
            need = max(TEXT_MIN_S, words / TEXT_WPS)
            if shown < need:
                hint(f"items[{i}].text", f"{words} words on screen {shown:.1f} s; give it ≥ {need:.1f} s "
                                         "or cut words")
            if words > TEXT_WORDS:
                hint(f"items[{i}].text", f"{words} words in one card; keep one idea (≤ {TEXT_WORDS} words)")
        if it.audio_lead and i and it.audio_lead > plan.items[i - 1].length:
            hint(f"items[{i}].audio_lead", f"J-cut of {it.audio_lead} frames is longer than the "
                                           f"previous shot ({plan.items[i - 1].length}); its sound "
                                           "also plays over the shot before")
        if it.audio_tail and i + 1 < len(plan.items) and it.audio_tail > plan.items[i + 1].length:
            hint(f"items[{i}].audio_tail", f"L-cut of {it.audio_tail} frames is longer than the "
                                           f"next shot ({plan.items[i + 1].length})")
        rec = edit_record(project, it.source.gen) if isinstance(it.source, GenSource) else None
        if rec is not None:
            _edit_sound_hint(i, it, rec, rate, hint)
        elif isinstance(it.source, GenSource) and it.in_ < round(GEN_SKIP_S * rate):
            hint(f"items[{i}].in", f"generated clip from its first {GEN_SKIP_S:g} s, which often "
                                   "wobble; start later")

    total = float(Fraction(plan.total_frames) / rate)
    if brief is not None:
        lo, hi = brief.duration_s
        if not lo <= total <= hi:
            hint("items", f"timeline is {total:.1f} s; the Brief asks for {lo:g}–{hi:g} s")
        elif brief.target_s and abs(total - brief.target_s) > max(2.0, 0.1 * brief.target_s):
            hint("items", f"timeline is {total:.1f} s; the Brief targets {brief.target_s:g} s")
    if project is not None:
        if index is not None:
            _word_hints(plan, project, heard, rate, hint)
        _beat_hints(plan, project, rate, hint)
        _colour_hints(plan, project, hint)
        _voice_hints(plan, project, rate, hint)
        _caption_hints(plan, project, rate, hint)
        _text_place_hints(plan, project, hint)
    return issues


def _text_place_hints(plan: EditPlan, project: Project, hint) -> None:
    """Item text that hides the tracked subject for ``COVER_HINT`` of its time or more."""
    from opencine.render.textplace import COVER_HINT, layout

    for p in layout(project, plan):
        if not p.hidden or p.hidden[p.band] < COVER_HINT:
            continue
        what = {"face": "face", "motion": "moving subject", "saliency": "subject"}.get(p.subject, "subject")
        pct = f"{p.hidden[p.band]:.0%} of the {p.shown_s:.1f} s it shows"
        free = [b for b in sorted(p.hidden, key=p.hidden.__getitem__)
                if b not in p.blocked and p.hidden[b] < p.hidden[p.band]]
        if p.auto or not free:
            hint(f"items[{p.item}].text", f"hides the {what} {pct} wherever it goes (auto → {p.band}): "
                                          "fewer words or a smaller size, or put it on a shot with room")
        else:
            hint(f"items[{p.item}].text", f'hides the {what} {pct} at "{p.band}"; "{free[0]}" hides '
                                          f'{p.hidden[free[0]]:.0%}: use position "auto"')


def _caption_hints(plan: EditPlan, project: Project, rate: Rate, hint) -> None:
    """Captions with nothing to show, item text in their band while they run, and emphasis on
    too many words (more than ``EMPHASIS_MAX`` of them stops standing out)."""
    from opencine.plan.captions import cues as caption_cues
    from opencine.plan.captions import emphasis_mask, missing_words

    cap = plan.captions
    if cap is None:
        return
    for k in missing_words(project, plan):
        hint(f"sfx[{k}]", "voice-over has no word times: run tts again for it")
    cues = caption_cues(project, plan)
    if not cues:
        hint("captions", "captions are on but no voice-over sfx has word times: add voice-over (tts) "
                         "or drop captions")
        return
    starts = plan.starts()
    for i, it in enumerate(plan.items):
        if it.text is None or it.text.position != cap.position:
            continue
        a = Fraction(starts[i] + it.text.start) / rate
        b = Fraction(starts[i] + (it.text.end if it.text.end is not None else it.length)) / rate
        if any(c.start < b and a < c.end for c in cues):
            hint(f"items[{i}].text", f"sits in the captions band ({cap.position}) while captions run: "
                                     "move it to another position")
    words = [w.text for c in cues for w in c.words]
    hit = sum(sum(emphasis_mask([w.text for w in c.words], cap.emphasis)) for c in cues)
    if words and hit / len(words) > EMPHASIS_MAX:
        hint("captions.emphasis", f"{hit} of {len(words)} words emphasised ({hit / len(words):.0%}); "
                                  f"keep it to ≤ {EMPHASIS_MAX:.0%} so they stand out")


def _voice_hints(plan: EditPlan, project: Project, rate: Rate, hint) -> None:
    """Voice-over that reads as broken up: lines that overlap, and a pause between lines longer
    than ``VO_GAP_S`` (music under it fills a longer one)."""
    spans = []
    for k, s in enumerate(plan.sfx):
        if not s.file.replace("\\", "/").startswith("audio/voice/"):
            continue
        try:
            with wave.open(str(project.root / s.file)) as w:
                dur = w.getnframes() / w.getframerate()
        except (OSError, EOFError, wave.Error):
            continue
        a = float(Fraction(s.at) / rate)
        spans.append((a, a + dur, k))
    if len(spans) < 2:
        return
    spans.sort()
    covered, end = 0.0, 0.0
    for a, b, _ in spans:
        covered += max(0.0, b - max(a, end))
        end = max(end, b)
    pairs = list(zip(spans, spans[1:], strict=False))
    for (_, b1, k1), (a2, _, k2) in pairs:
        if a2 < b1 - VO_OVERLAP_S:
            hint(f"sfx[{k2}]", f"voice-over overlaps sfx[{k1}] by {b1 - a2:.1f} s")
    gap, at = max((a2 - b1, b1) for (_, b1, _), (a2, _, _) in pairs)
    if gap > VO_GAP_S[plan.music is not None]:
        total = float(Fraction(plan.total_frames) / rate)
        bare = "" if plan.music else ", with no music under it"
        hint("sfx", f"voice-over covers {covered / total:.0%} of the timeline ({covered:.1f}/{total:.1f} s) "
                    f"and stops for {gap:.1f} s at {fmt_ts(at)}–{fmt_ts(at + gap)}{bare}: it reads as "
                    "separate lines; write one script per chapter and place the lines back to back")


def _colour_hints(plan: EditPlan, project: Project, hint) -> None:
    """Cuts whose two sides still differ a lot in colour after the automatic matching."""
    from opencine.render.colour import neighbour_gaps

    try:
        res = Resolver(project, plan.fps)
        medias = [res.media(it) for it in plan.items]
        gaps = neighbour_gaps(project, plan, medias)
    except (OSError, KeyError, ValueError):
        return
    for k, de in gaps:
        hint(f"items[{k}].color", f"colour jumps ΔE {de:g} from items[{k - 1}] (beyond the automatic "
                                  "match): intended? else color: {match: <the other scene>} or pick "
                                  "a closer shot")


def shotplan_hints(sp: ShotPlan, brief: Brief | None = None) -> list[Issue]:
    """Craft hints for a ShotPlan's intent fields (role, shot, cut_on) against the Brief."""
    issues: list[Issue] = []

    def hint(path: str, msg: str) -> None:
        issues.append(Issue(level="hint", path=path, message=msg))

    slots = sp.slots
    if not slots:
        return issues
    roles = [s.role for s in slots]
    film = is_film(template=sp.template)  # a film may open on a long establishing shot
    if any(roles) and roles[0] not in (None, "hook") and not film:
        hint("slots[0].role", f"the first slot is {roles[0]!r}; open on the hook (the first 1–2 s decide)")
    if slots[0].dur_s[0] > HOOK_MAX_S and not film:
        hint("slots[0].dur_s", f"the opening slot lasts ≥ {slots[0].dur_s[0]:g} s; cut by ~{HOOK_MAX_S:g} s")
    for i in range(1, len(slots)):
        a, b = slots[i - 1], slots[i]
        fa, fb = a.source.footage, b.source.footage
        if a.shot and a.shot == b.shot and fa and fb and fa.scene == fb.scene:
            hint(f"slots[{i}]", f"same scene and size ({b.shot}) as slots[{i - 1}]: jump-cut risk; "
                                "change size/angle or put another shot between")
    i = 0
    while i < len(slots):
        j = i
        while j + 1 < len(slots) and slots[i].shot and slots[j + 1].shot == slots[i].shot:
            j += 1
        if j - i + 1 >= SAME_RUN:
            hint(f"slots[{i}]", f"slots {i}–{j} are all {slots[i].shot}: vary the shot size")
        i = j + 1
    if any(roles) and "payoff" not in roles and not (brief and brief.ending == "cta"):
        hint("slots", "no payoff slot: show the result and hold it")
    lo, hi = sum(s.dur_s[0] for s in slots), sum(s.dur_s[1] for s in slots)
    if brief is not None:
        if brief.target_s and not lo <= brief.target_s <= hi:
            hint("slots", f"slots add up to {lo:g}–{hi:g} s; the Brief targets {brief.target_s:g} s")
        elif not brief.target_s and (hi < brief.duration_s[0] or lo > brief.duration_s[1]):
            hint("slots", f"slots add up to {lo:g}–{hi:g} s; the Brief asks for "
                          f"{brief.duration_s[0]:g}–{brief.duration_s[1]:g} s")
        if brief.ending == "loop" and roles[-1] != "loop":
            hint(f"slots[{len(slots) - 1}].role", "the Brief ends on a loop: make the last slot "
                                                  "role 'loop' (back to the opening frame)")
    if brief is not None and brief.story:
        _chapter_hints(slots, brief, hint)
    if any(s.cut_on == "beat" for s in slots) and not (sp.music or (brief and brief.music)):
        hint("music", "slots cut on the beat but no music is named: music_add a track and set "
                      "ShotPlan.music")
    _prompt_hints(slots, brief, hint)
    return issues


def _prompt_hints(slots, brief: Brief | None, hint) -> None:
    """``opencine.gen.lint`` on generated slots' H3 prompts (Context-IR / Ref2VA only: a short note
    is not a prompt yet), against the slot's longest length and the film bible."""
    from opencine.gen.lint import video_hints

    for i, s in enumerate(slots):
        g = s.source.generate
        if not (g and g.prompt and g.prompt.lstrip().startswith(
                ("integrated_multimodal_description:", "subject_definitions:"))):
            continue
        for text in video_hints(g.prompt, s.dur_s[1], first=bool(g.ref),
                                notes=brief.notes if brief else None):
            hint(f"slots[{i}].source.generate.prompt", text)


def _chapter_hints(slots, brief: Brief, hint) -> None:
    """Slots out of story order, unknown chapters, chapters without slots."""
    order = {c.name.strip().lower(): k for k, c in enumerate(brief.story)}
    last, seen = -1, set()
    for i, s in enumerate(slots):
        if s.chapter is None:
            continue
        k = order.get(s.chapter.strip().lower())
        if k is None:
            hint(f"slots[{i}].chapter", f"chapter {s.chapter!r} is not in Brief.story "
                                        f"({', '.join(c.name for c in brief.story)})")
            continue
        seen.add(k)
        if k < last:
            hint(f"slots[{i}].chapter", f"goes back to chapter {s.chapter!r} after "
                                        f"{brief.story[last].name!r}: keep chapters in story order "
                                        "(or make it a deliberate flashback)")
        last = max(last, k)
    if any(s.chapter for s in slots):
        empty = [c.name for k, c in enumerate(brief.story) if k not in seen]
        if empty:
            hint("slots", f"no slot for chapter(s) {', '.join(empty)}")


def _brief_story(project: Project) -> list | None:
    from opencine.plan.store import load_plan, versions

    try:
        return load_plan(project, "brief")[1].story if versions(project, "brief") else None
    except (FileNotFoundError, ValueError):
        return None


SLOTS_RANGE = (10, 20)  # a 30–60 s short is usually 10–20 shots
TRANSFORM_MAX_S = 15.0  # a gen_edit window (opencine.gen.source.MAX_WINDOW_S)


def validate_shotplan(sp: ShotPlan, project: Project | None = None) -> list[Issue]:
    """Errors that make a ShotPlan unusable (duplicate slot ids, a slot with no or two sources,
    a scene/ref/file that does not resolve) and warnings; with ``project`` refs are resolved."""
    from opencine.index.sheets import resolve_ref

    issues: list[Issue] = []

    def add(level: Literal["error", "warning"], path: str, msg: str) -> None:
        issues.append(Issue(level=level, path=path, message=msg))

    if not sp.slots:
        add("warning", "slots", "no slots")
        return issues
    if not SLOTS_RANGE[0] <= len(sp.slots) <= SLOTS_RANGE[1] and not is_film(project, sp.template):
        add("warning", "slots", f"{len(sp.slots)} slots; a 30–60 s short usually has "
                                f"{SLOTS_RANGE[0]}–{SLOTS_RANGE[1]}")
    from opencine.plan.templates import TEMPLATES

    if sp.template and sp.template.strip().lower() not in TEMPLATES:
        add("warning", "template", f"unknown template {sp.template!r}; plan_solve uses its defaults "
                                   f"(known: {', '.join(TEMPLATES)})")
    seen: dict[int, int] = {}
    for i, s in enumerate(sp.slots):
        if s.slot in seen:
            add("error", f"slots[{i}].slot", f"slot {s.slot} is also slots[{seen[s.slot]}]; ids must be unique")
        seen.setdefault(s.slot, i)
        lo, hi = s.dur_s
        if lo <= 0 or hi < lo:
            add("error", f"slots[{i}].dur_s", f"dur_s must be [min, max] with 0 < min ≤ max, got {list(s.dur_s)}")
        given = [s.source.footage, s.source.generate, s.source.transform]
        if sum(x is not None for x in given) != 1:
            add("error", f"slots[{i}].source", "give exactly one of footage, generate or transform")
        t = s.source.transform
        if t is not None and not 0 < t.out_s - t.in_s <= TRANSFORM_MAX_S + 1e-6:
            add("error", f"slots[{i}].source.transform",
                f"window {t.in_s:g}–{t.out_s:g} s: out_s after in_s, at most {TRANSFORM_MAX_S:g} s")
        if s.crop is not None and (s.crop.x + s.crop.w > 1 + 1e-9 or s.crop.y + s.crop.h > 1 + 1e-9):
            add("error", f"slots[{i}].crop", "crop box goes outside the source frame")
    if project is None:
        return issues
    index = _index(project)
    story = {c.name.strip().lower(): c for c in _brief_story(project) or []}
    for i, s in enumerate(sp.slots):
        f, g, t = s.source.footage, s.source.generate, s.source.transform
        if t is not None:
            if index is None:
                add("error", f"slots[{i}].source.transform", "the project is not ingested yet")
                continue
            try:
                sc = index.scene(t.scene.strip().upper())
            except KeyError:
                add("error", f"slots[{i}].source.transform.scene", f"unknown scene {t.scene!r}")
                continue
            if t.out_s > sc.duration_s + 1e-6:
                add("error", f"slots[{i}].source.transform.out_s",
                    f"{t.out_s:g} s is past the end of {sc.id} ({sc.duration_s:.2f} s)")
        elif f is not None:
            if index is None:
                add("error", f"slots[{i}].source.footage", "the project is not ingested yet")
                continue
            try:
                sc = index.scene(f.scene.strip().upper())
            except KeyError:
                add("error", f"slots[{i}].source.footage.scene", f"unknown scene {f.scene!r}")
                continue
            ch = story.get((s.chapter or "").strip().lower())
            if ch is not None and ch.clips and sc.clip not in {c.strip().upper() for c in ch.clips}:
                add("warning", f"slots[{i}].chapter", f"{sc.id} is in {sc.clip}, not in chapter "
                                                      f"{ch.name!r} ({', '.join(ch.clips)})")
            if f.at is not None and f.at >= sc.duration_s:
                add("error", f"slots[{i}].source.footage.at",
                    f"at {f.at:g} s is past the end of {sc.id} ({sc.duration_s:.2f} s); "
                    "it is seconds into the scene")
            elif s.dur_s[0] * _speed_factor(s.speed) > sc.duration_s + 1e-6:
                add("warning", f"slots[{i}].dur_s", f"{sc.id} lasts {sc.duration_s:.2f} s, too short "
                                                    f"for the slot's minimum {s.dur_s[0]:g} s"
                                                    + (f" at speed {s.speed}" if s.speed not in (None, "normal") else ""))
        elif g is not None and g.ref:
            ref = g.ref.strip()
            if _FRAME_REF.match(ref):
                if index is None:
                    add("error", f"slots[{i}].source.generate.ref", "the project is not ingested yet")
                    continue
                try:
                    resolve_ref(index, ref)
                except (KeyError, ValueError) as e:
                    add("error", f"slots[{i}].source.generate.ref", str(e.args[0] if e.args else e))
            elif not project_path(project, ref).is_file():
                add("error", f"slots[{i}].source.generate.ref",
                    f"{ref!r} is neither a frame ref (S03@2.5, C01@12.4) nor a file")
        elif g is not None:
            add("warning", f"slots[{i}].source.generate.ref", "no ref yet: generation needs a frame ref or an image")
    if sp.music:
        from opencine.music.library import track

        try:
            track(project, sp.music)
        except FileNotFoundError as e:
            add("error", "music", str(e))
    return issues


_PLACEHOLDER = re.compile(r"\{(time|date)\}")
_FRAME_REF = re.compile(r"^[SC]\d+(@[\d:.,]+)?$", re.IGNORECASE)


def _speed_factor(speed: str | None) -> float:
    """Source seconds per output second for a slot ``speed`` (as ``plan_solve`` maps it)."""
    return {"fast": 1.5, "slow": 0.5, "ramp_slow_at_impact": 0.5}.get(speed or "normal", 1.0)


def _index(project: Project | None) -> SceneIndex | None:
    if project is None:
        return None
    from opencine.index.model import load_index

    try:
        return load_index(project)
    except FileNotFoundError:
        return None


def _span(it: EditItem, index: SceneIndex | None, rate: Rate) -> tuple[str, int, int] | None:
    """``(clip, in, out)`` in plan-rate frames from the clip start (from the scene start for a
    scene when there is no index); None for generated clips and stills."""
    src = it.source
    if isinstance(src, ClipSource):
        return src.clip, it.in_, it.out
    if not isinstance(src, SceneSource):
        return None
    if index is None:
        return f"scene:{src.scene}", it.in_, it.out
    try:
        sc = index.scene(src.scene)
    except KeyError:
        return None
    off = to_frame(Fraction(sc.start_frame) / Rate(index.rate), rate, "nearest")
    return sc.clip, off + it.in_, off + it.out


def _sound_span(it: EditItem, index: SceneIndex | None, rate: Rate) -> tuple[str, int, int] | None:
    """``_span`` of the footage an item's ``sound`` plays (in real time, for the item's length)."""
    s = it.sound
    return _span(it.model_copy(update={"source": s.source, "in_": s.in_, "out": s.in_ + it.length}),
                 index, rate)


SUBJECT_OUT_S = 0.3


def _subject_hints(plan: EditPlan, project: Project, index: SceneIndex, rate: Rate, hint) -> None:
    """Cropped footage whose tracked subject leaves the crop for more than ``SUBJECT_OUT_S``."""
    from opencine.plan.reframe import outside_s

    for i, it in enumerate(plan.items):
        if it.crop is None or not isinstance(it.source, SceneSource):
            continue
        try:
            sc = index.scene(it.source.scene)
        except KeyError:
            continue
        out = outside_s(project, sc.clip, sc.start_frame, it.in_, it.out, it.crop, rate,
                        deliverable=plan.size)
        if out > SUBJECT_OUT_S:
            hint(f"items[{i}].crop", f"the subject is outside the crop for ~{out:.1f} s; move the "
                                     "crop, or let plan_solve track it (reframe)")


def _word_hints(plan: EditPlan, project: Project, spans, rate: Rate, hint) -> None:
    """Audible items whose in/out falls inside a spoken word."""
    from opencine.speech.transcribe import load_transcripts

    trs = load_transcripts(project)
    for i, (it, span) in enumerate(zip(plan.items, spans, strict=True)):
        if span is None or it.gain is None or span[0] not in trs:
            continue
        words = [w for seg in trs[span[0]].get("segments") or [] for w in seg.get("words") or []
                 if w.get("start") is not None and w.get("end") is not None]
        for edge, f in (("in", span[1]), ("out", span[2])):
            t = float(Fraction(f) / rate)
            cut = next((w for w in words if w["start"] + WORD_EDGE_S < t < w["end"] - WORD_EDGE_S), None)
            if cut:
                hint(f"items[{i}].{'sound' if it.sound else edge}", f"cuts the word {cut['word']!r} "
                                           f"({cut['start']:.2f}–{cut['end']:.2f} s in {span[0]}); "
                                           "move to a word boundary or mute the item")


def _beat_hints(plan: EditPlan, project: Project, rate: Rate, hint) -> None:
    """Cuts more than ``BEAT_TOL`` frames off the music's beats, when its grid exists
    (``music_add`` / ``opencine beats``)."""
    if plan.music is None:
        return
    from opencine.music.beats import find_beats

    side = find_beats(project_path(project, plan.music.file))
    if side is None:
        return
    beats = [to_frame(b["t"], rate, "nearest") - plan.music.offset
             for b in json.loads(side.read_text("utf-8"))]
    if not beats:
        return
    off = []
    for i, start in enumerate(plan.starts()[1:], 1):
        d = min((start - b for b in beats), key=abs)
        if abs(d) > BEAT_TOL:
            off.append(f"items[{i}] {d:+d}f")
    if off:
        hint("music", f"{len(off)}/{len(plan.items) - 1} cuts are > {BEAT_TOL} frames off a beat: "
                      + ", ".join(off[:8]) + ("…" if len(off) > 8 else ""))
