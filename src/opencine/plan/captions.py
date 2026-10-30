"""Voice captions: each voice-over line's words (``audio/voice/<id>.words.json``) grouped into cues.

``chunk`` closes at punctuation, a pause ≥ ``PAUSE_S`` or ``max_chars``, never on a function word;
``karaoke``/``line`` show a sentence per cue, at most ``MAX_LINES`` lines. A cue starts ``LEAD_S``
before its first word and lasts to the next cue (gap < ``JOIN_S``) or ``HOLD_S`` past its last word.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.schema import Captions, EditPlan

PAUSE_S = 0.25
LEAD_S = 0.05
JOIN_S = 0.6
HOLD_S = 0.3
MIN_S = 0.35
CHUNK_CHARS, LINE_CHARS, MAX_LINES = 15, 42, 2
SENTENCE_END = (".", "!", "?", "…")
PUNCT = (*SENTENCE_END, ",", ";", ":")
# function words a chunk should not end on ("squeeze the | lemon" reads badly)
FUNCTION_WORDS = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "by", "from",
                  "into", "onto", "about", "over", "under", "and", "or", "but", "nor", "so", "yet",
                  "as", "if", "than", "that", "is", "are", "was", "were", "be", "been", "am",
                  "will", "would", "can", "could", "should", "shall", "may", "might", "must",
                  "has", "have", "had", "do", "does", "did", "my", "your", "his", "her", "its",
                  "our", "their", "this", "these", "those"}


@dataclass
class Word:
    text: str
    start: Fraction  # timeline seconds
    end: Fraction


@dataclass
class Cue:
    start: Fraction
    end: Fraction
    words: list[Word]
    lines: list[int] = field(default_factory=list)  # word index where each line after the first starts

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _bare(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def emphasis_mask(texts: list[str], phrases: list[str]) -> list[bool]:
    """Which of ``texts`` (words in order) belong to one of ``phrases`` (case-insensitive, whole
    words; a phrase may span several words)."""
    bare = [_bare(t) for t in texts]
    mask = [False] * len(texts)
    for phrase in phrases:
        toks = [b for b in map(_bare, phrase.split()) if b]
        for i in range(len(bare) - len(toks) + 1):
            if toks and bare[i:i + len(toks)] == toks:
                mask[i:i + len(toks)] = [True] * len(toks)
    return mask


def line_words(project: Project, plan: EditPlan) -> list[tuple[str, list[Word]]]:
    """``(line id, words)`` for every voice-over sfx whose words file exists, in timeline order."""
    out = []
    for s in plan.sfx:
        f = s.file.replace("\\", "/")
        if not f.startswith("audio/voice/") or not f.endswith(".wav"):
            continue
        path = project.root / f[:-4]
        path = path.with_name(path.name + ".words.json")
        if not path.is_file():
            continue
        t0 = Fraction(s.at) / plan.fps
        data = json.loads(path.read_text("utf-8"))
        words = [Word(w["word"], t0 + Fraction(str(w["start"])), t0 + Fraction(str(w["end"])))
                 for w in data.get("words", [])]
        if words:
            out.append((path.name.removesuffix(".words.json"), words))
    return sorted(out, key=lambda lw: lw[1][0].start)


def missing_words(project: Project, plan: EditPlan) -> list[int]:
    """Indexes of voice-over sfx without a words file."""
    have = {lid for lid, _ in line_words(project, plan)}
    return [k for k, s in enumerate(plan.sfx)
            if s.file.replace("\\", "/").startswith("audio/voice/")
            and s.file.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".wav") not in have]


def _chunks(words: list[Word], max_chars: int, keep: set[str]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []

    def width(ws: list[Word]) -> int:
        return len(" ".join(w.text for w in ws))

    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if cur and width([*cur, w]) > max_chars and not (
                _bare(cur[-1].text) in FUNCTION_WORDS and width([*cur, w]) <= max_chars + 4):
            groups.append(cur)
            cur = []
        cur.append(w)
        ends = w.text.endswith(PUNCT) or nxt is None or nxt.start - w.end >= PAUSE_S
        if ends and (w.text.endswith(PUNCT) or _bare(w.text) not in FUNCTION_WORDS or nxt is None):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    merged: list[list[Word]] = []
    for g in groups:  # a lone word joins the chunk before it when both fit
        lone = len(g) == 1 and _bare(g[0].text) not in keep
        if merged and lone and not merged[-1][-1].text.endswith(SENTENCE_END) \
                and width([*merged[-1], *g]) <= max_chars + 4:
            merged[-1] = [*merged[-1], *g]
        else:
            merged.append(g)
    return merged


def _sentences(words: list[Word], max_chars: int) -> list[tuple[list[Word], list[int]]]:
    """Sentences wrapped to ``max_chars`` a line, ``MAX_LINES`` lines a cue."""
    out: list[tuple[list[Word], list[int]]] = []
    cur: list[Word] = []
    breaks: list[int] = []
    line = 0
    for w in words:
        add = len(w.text) + (1 if line else 0)
        if line and line + add > max_chars:
            if len(breaks) + 1 >= MAX_LINES:
                out.append((cur, breaks))
                cur, breaks, line, add = [], [], 0, len(w.text)
            else:
                breaks.append(len(cur))
                line, add = 0, len(w.text)
        cur.append(w)
        line += add
        if w.text.endswith(SENTENCE_END):
            out.append((cur, breaks))
            cur, breaks, line = [], [], 0
    if cur:
        out.append((cur, breaks))
    return out


def _by_breaks(words: list[Word], spec: str) -> list[list[Word]] | None:
    """The user's own chunks, when their words are exactly the line's."""
    parts = [p.split() for p in spec.split("|") if p.strip()]
    if [_bare(t) for p in parts for t in p] != [_bare(w.text) for w in words]:
        return None
    out, k = [], 0
    for p in parts:
        out.append(words[k:k + len(p)])
        k += len(p)
    return out


def cues(project: Project, plan: EditPlan) -> list[Cue]:
    """What is on screen when, for ``plan.captions`` (empty without captions or voice)."""
    cap: Captions | None = plan.captions
    if cap is None:
        return []
    keep = {b for k in cap.emphasis for b in map(_bare, k.split()) if b}
    raw: list[tuple[list[Word], list[int]]] = []
    for lid, words in line_words(project, plan):
        if cap.mode == "chunk":
            groups = (_by_breaks(words, cap.breaks[lid]) if lid in cap.breaks else None) \
                or _chunks(words, cap.max_chars or CHUNK_CHARS, keep)
            raw += [(g, []) for g in groups]
        else:
            raw += _sentences(words, cap.max_chars or LINE_CHARS)
    out: list[Cue] = []
    for k, (ws, br) in enumerate(raw):
        start = max(ws[0].start - Fraction(str(LEAD_S)), out[-1].end if out else Fraction(0))
        nxt = raw[k + 1][0][0].start - Fraction(str(LEAD_S)) if k + 1 < len(raw) else None
        end = ws[-1].end + Fraction(str(HOLD_S))
        if nxt is not None and nxt - ws[-1].end < Fraction(str(JOIN_S)):
            end = nxt
        end = max(end, start + Fraction(str(MIN_S)))
        if out and start < out[-1].end:
            start = out[-1].end
        out.append(Cue(start, end, ws, br))
    return out
