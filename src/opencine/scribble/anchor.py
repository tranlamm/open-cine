"""Resolve an item's ``at`` to timeline seconds.

- a number: timeline seconds (as in ``review_sheet`` and ``sfx.at_s``);
- ``"S03@2.1"`` / ``"C01@4.0"`` (``mm:ss.f`` and a decimal comma accepted): that footage moment
  wherever the edit first shows it, through speed/ramp, so it survives plan changes;
- ``"say:crispy"`` / ``"say:so crispy#2"``: when the voice-over says the word or phrase (``#n`` =
  n-th time), ``SAY_LEAD_S`` early so the pen moves as the word lands.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING

from opencine.core.time import parse_ts
from opencine.plan.schema import ClipSource, SceneSource

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.schema import EditPlan

SAY_LEAD_S = 0.1
MOMENT_RE = re.compile(r"^\s*([SC]\d+)\s*@\s*([\d:.,]+)\s*$", re.IGNORECASE)
SAY_RE = re.compile(r"^\s*say\s*:\s*(.+?)\s*(?:#(\d+))?\s*$", re.IGNORECASE | re.DOTALL)


def _clip_s(project: Project, ref: str, seconds: Fraction) -> tuple[str, Fraction]:
    """``(clip id, seconds from the clip start)`` of a scene or clip moment."""
    from opencine.index.model import load_index

    idx = load_index(project)
    if ref.startswith("C"):
        idx.clip(ref)  # unknown clip → KeyError
        return ref, seconds
    sc = idx.scene(ref)
    return sc.clip, Fraction(sc.start_frame) / Fraction(idx.rate) + seconds


def moment_s(project: Project, plan: EditPlan, ref: str, seconds: Fraction) -> float:
    """Timeline seconds where the edit shows ``ref`` (scene/clip id) at ``seconds`` into it."""
    from opencine.index.model import load_index
    from opencine.plan.ramp import time_at

    idx = load_index(project)
    clip, want = _clip_s(project, ref, seconds)
    rate = plan.fps
    shown = []
    for n, (start, it) in enumerate(zip(plan.starts(), plan.items, strict=True), 1):
        src = it.source
        if isinstance(src, SceneSource):
            sc = idx.scene(src.scene)
            if sc.clip != clip:
                continue
            base, name = Fraction(sc.start_frame) / Fraction(idx.rate), src.scene
        elif isinstance(src, ClipSource) and src.clip == clip:
            base, name = Fraction(0), src.clip
        else:
            continue
        a, b = base + Fraction(it.in_) / rate, base + Fraction(it.out) / rate
        shown.append(f"#{n} {name} {float(a - base):.2f}–{float(b - base):.2f} s")
        if not a <= want < b:
            continue
        src_frames = float((want - a) * rate)  # source frames past the item's in, at plan rate
        out = time_at(it.ramp, src_frames) if it.ramp else src_frames / it.speed
        return round(float((start + min(it.length - 1, round(out))) / rate), 3)
    where = "; ".join(shown) if shown else "none"
    raise ValueError(f"{ref}@{float(seconds):g} is not on screen in this EditPlan (footage of {clip} "
                     f"shown: {where})")


def say_s(project: Project, plan: EditPlan, phrase: str, nth: int = 1) -> float:
    """Timeline seconds the voice-over starts saying ``phrase`` (its ``nth`` time), minus the lead."""
    from opencine.plan.captions import _bare, line_words

    toks = [b for b in map(_bare, phrase.split()) if b]
    words = [w for _, ws in line_words(project, plan) for w in ws]
    bare = [_bare(w.text) for w in words]
    hits = [i for i in range(len(bare) - len(toks) + 1) if toks and bare[i:i + len(toks)] == toks]
    if len(hits) < nth:
        if not words:
            raise ValueError("say: needs voice-over lines with word timing (tts → sfx audio/voice/…)")
        said = " ".join(w.text for w in words)
        raise ValueError(f"say:{phrase} — the voice-over says it {len(hits)}×; it says: {said[:300]}")
    return round(max(0.0, float(words[hits[nth - 1]].start) - SAY_LEAD_S), 3)


def timeline_s(project: Project, plan: EditPlan, at: float | int | str) -> float:
    """Timeline seconds of an item's ``at`` (see the module doc)."""
    if isinstance(at, bool):
        raise ValueError(f"at: expected seconds or a moment, got {at!r}")
    if isinstance(at, int | float):
        if at < 0:
            raise ValueError(f"at must be ≥ 0, got {at}")
        return float(at)
    text = str(at)
    if m := SAY_RE.match(text):
        return say_s(project, plan, m.group(1), int(m.group(2) or 1))
    if m := MOMENT_RE.match(text):
        return moment_s(project, plan, m.group(1).upper(), parse_ts(m.group(2).replace(",", ".")))
    try:
        return timeline_s(project, plan, float(text.replace(",", ".")))
    except ValueError:
        pass
    raise ValueError(f"at: timeline seconds, 'S03@2.1', 'C01@4.0' or 'say:<word>[#n]'; got {at!r}")

