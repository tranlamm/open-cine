"""Word timings of a voice-over line, for captions that follow the voice.

Words keep the script's spelling; times come from STT on the synthesised file (script as context),
matched in order with ``difflib``; missed words are interpolated between matched neighbours.
Without STT, times are estimated: silent gaps anchor inner punctuation when the counts match,
otherwise the voiced span is split by word length with room for pauses.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FRAME_S = 0.02  # energy window for the voiced span
VOICED_DB = -40.0  # dB below the file's peak = silence
GAP_S = 0.12  # min silence inside speech that counts as a pause
PAUSE_WEIGHT = {",": 2, ";": 2, ":": 2, ".": 4, "!": 4, "?": 4, "…": 4}  # pause length in characters


def _norm(word: str) -> str:
    w = unicodedata.normalize("NFC", word).lower()
    return re.sub(r"[^\w]", "", w)


def script_words(text: str) -> list[str]:
    return [w for w in text.split() if _norm(w)]


def _loudness(wav: Path) -> tuple[np.ndarray, float]:
    """Per-``FRAME_S`` loud/quiet mask and the file length."""
    x, sr = sf.read(str(wav), always_2d=True)
    mono = np.abs(x).mean(axis=1)
    hop = max(1, int(sr * FRAME_S))
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0, bool), len(mono) / sr
    rms = np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return (rms > rms.max() * 10 ** (VOICED_DB / 20)) if rms.max() > 0 else np.zeros(n, bool), len(mono) / sr


def voiced_span(wav: Path) -> tuple[float, float]:
    """``(start, end)`` seconds of the part louder than ``VOICED_DB`` under the peak."""
    loud, length = _loudness(wav)
    idx = np.flatnonzero(loud)
    if idx.size == 0:
        return 0.0, length
    return float(idx[0] * FRAME_S), float((idx[-1] + 1) * FRAME_S)


def pauses(wav: Path) -> list[tuple[float, float]]:
    """Silent gaps of ``GAP_S`` or more inside the voiced span."""
    loud, _ = _loudness(wav)
    idx = np.flatnonzero(loud)
    out, run = [], None
    for k in range(int(idx[0]), int(idx[-1]) + 1) if idx.size else []:
        if not loud[k]:
            run = k if run is None else run
        elif run is not None:
            if (k - run) * FRAME_S >= GAP_S:
                out.append((run * FRAME_S, k * FRAME_S))
            run = None
    return out


def estimate(words: list[str], span: tuple[float, float]) -> list[dict[str, Any]]:
    """Split ``span`` by word length (+1 per gap), leaving a pause after punctuation."""
    a, b = span
    weights = [len(_norm(w)) + 1 for w in words]
    gaps = [PAUSE_WEIGHT.get(w[-1], 0) if k + 1 < len(words) else 0 for k, w in enumerate(words)]
    unit = (b - a) / (sum(weights) + sum(gaps) or 1)
    out, t = [], a
    for w, k, g in zip(words, weights, gaps, strict=True):
        out.append({"word": w, "start": round(float(t), 3), "end": round(float(t + k * unit), 3)})
        t += (k + g) * unit
    return out


def estimate_from_pauses(words: list[str], wav: Path) -> list[dict[str, Any]]:
    """``estimate`` per phrase, using pauses as anchors when they match the script's inner
    punctuation one to one; else ``estimate`` over the voiced span."""
    span = voiced_span(wav)
    marks = [k for k, w in enumerate(words[:-1]) if w[-1] in PAUSE_WEIGHT]
    gaps = pauses(wav)
    if not marks or len(gaps) != len(marks):
        return estimate(words, span)
    out: list[dict[str, Any]] = []
    starts = [span[0], *(g[1] for g in gaps)]
    ends = [*(g[0] for g in gaps), span[1]]
    bounds = [0, *(k + 1 for k in marks), len(words)]
    for (a, b), (i, j) in zip(zip(starts, ends, strict=True), zip(bounds, bounds[1:], strict=False),
                              strict=True):
        out += estimate(words[i:j], (a, b))
    return out


def align(words: list[str], recognised: list[dict[str, Any]], span: tuple[float, float]) -> list[dict[str, Any]]:
    """Time script ``words`` from recognised ``{word, start, end}`` (entries without times are
    not anchors)."""
    timed = [r for r in recognised if r.get("start") is not None and r.get("end") is not None]
    out: list[dict[str, Any]] = [{"word": w, "start": None, "end": None} for w in words]
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in words], b=[_norm(r["word"]) for r in timed],
                                 autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            r = timed[blk.b + k]
            out[blk.a + k].update(start=float(r["start"]), end=float(r["end"]))
    # fill gaps between anchors (or the voiced span's edges) by word length
    i = 0
    while i < len(out):
        if out[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j]["start"] is None:
            j += 1
        lo = out[i - 1]["end"] if i > 0 else span[0]
        hi = out[j]["start"] if j < len(out) else span[1]
        for k, e in enumerate(estimate(words[i:j], (lo, max(lo, hi)))):
            out[i + k].update(start=e["start"], end=e["end"])
        i = j
    return [{"word": e["word"], "start": round(e["start"], 3), "end": round(max(e["end"], e["start"]), 3)}
            for e in out]


def time_words(wav: Path, text: str, *, stt: Any = None, language: str | None = None) -> dict[str, Any]:
    """``{"timing": "stt"|"estimated", "words": [{word, start, end}]}`` for one line."""
    words = script_words(text)
    span = voiced_span(wav)
    if stt is not None:
        try:
            res = stt.transcribe(wav, language=language, context=text)
            recognised = [w for seg in res.get("segments") or [] for w in seg.get("words") or []]
            if recognised:
                return {"timing": "stt", "words": align(words, recognised, span)}
        except Exception:  # STT down or refused: estimate rather than fail
            pass
    return {"timing": "estimated", "words": estimate_from_pauses(words, wav)}
