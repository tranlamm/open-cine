"""Per-clip transcripts with word timestamps (``index/transcripts/C01.json``).

The clip's audio is sent as ≤ ``CHUNK_S`` 16 kHz mono FLAC pieces (the gateway caps body size and
duration) with ``timeline_offset_s``, so all times are **clip seconds**. Cached by clip sha256 +
language, so re-ingesting only transcribes new clips.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opencine.core.cache import cache_key
from opencine.core.config import get_settings
from opencine.core.ffmpeg import run
from opencine.core.ledger import Ledger
from opencine.core.time import fmt_ts
from opencine.index.steps import StepCache, place
from opencine.speech.client import SpeechError, SttClient

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.model import ClipInfo, SceneIndex

VERSION = 1
CHUNK_S = 600  # 10 min of 16 kHz mono FLAC ≈ 10–20 MB, well under the gateway's 64 MB / 1800 s
PARAMS = {"ar": 16000, "ac": 1, "codec": "flac", "chunk_s": CHUNK_S}
PARALLEL = 2  # gateway default concurrency (STS_BATCH_MAX_CONCURRENCY)
L0_SPEECH_CHARS = 160  # per scene line in L0


def transcripts_dir(project: Project) -> Path:
    return project.index_dir / "transcripts"


def _chunks(wav: Path, duration_s: float, work: Path) -> list[tuple[float, Path]]:
    out = []
    start = 0.0
    while start < max(duration_s, 0.01):
        piece = work / f"{int(start):06d}.flac"
        run(["-y", "-ss", f"{start:.3f}", "-t", str(CHUNK_S), "-i", str(wav), "-ac", "1",
             "-ar", "16000", "-c:a", "flac", str(piece)])
        out.append((start, piece))
        start += CHUNK_S
    return out


def _merge(parts: list[dict[str, Any]], clip: str) -> dict[str, Any]:
    segments = [seg for p in parts for seg in p.get("segments") or []]
    for seg in segments:  # keep only the fields the Director uses
        seg["words"] = [{k: w.get(k) for k in ("word", "start", "end", "score")}
                        for w in seg.get("words") or []]
    return {
        "clip": clip,
        "language": next((p.get("language") for p in parts if p.get("language")), None),
        "text": " ".join(t for p in parts if (t := (p.get("text") or "").strip())),
        "segments": [{k: s.get(k) for k in ("start", "end", "text", "confidence", "words")}
                     for s in segments],
    }


def transcribe_clip(project: Project, clip: ClipInfo, steps: StepCache, client: SttClient,
                    language: str | None) -> dict[str, Any]:
    if not clip.audio:
        return {"clip": clip.id, "language": None, "text": "", "segments": [], "no_audio": True}
    key = cache_key("stt", clip.sha256, language, PARAMS, VERSION)

    def make() -> dict[str, Any]:
        work = steps.dir("stt_chunks", key)
        parts = [client.transcribe(piece, language=language, offset_s=start)
                 for start, piece in _chunks(project.root / clip.audio, clip.duration_s, work)]
        shutil.rmtree(work, ignore_errors=True)  # keep only the JSON
        Ledger(project).record("stt", "transcribe", round(clip.duration_s, 1),
                               quota_note="self-hosted STT, audio seconds", clip=clip.id)
        return _merge(parts, clip.id)

    data = steps.json("stt", key, make)
    src = steps.cache.path("stt", key, ".json")
    place(src, transcripts_dir(project) / f"{clip.id}.json")
    return data


def transcribe_all(project: Project, index: SceneIndex, steps: StepCache | None = None,
                   client: SttClient | None = None, language: str | None = None) -> dict[str, Any]:
    """Transcribe every clip with audio (cached); raises ``SpeechUnavailable`` up front when the
    gateway is down. A duplicate clip (``same_as``) reuses its kept copy's transcript."""
    steps = steps or StepCache(project.cache_dir)
    client = client or SttClient.from_settings()
    client.require_up()
    language = language or get_settings().speech_language
    with ThreadPoolExecutor(PARALLEL) as pool:
        results = list(pool.map(lambda c: transcribe_clip(project, c, steps, client, language),
                                [c for c in index.clips if not c.same_as]))
    out = {r["clip"]: {"words": sum(len(s["words"]) for s in r["segments"]),
                       "segments": len(r["segments"]), "language": r["language"]} for r in results}
    d = transcripts_dir(project)
    for c in index.clips:
        if c.same_as and (kept := d / f"{c.same_as}.json").exists():
            place(kept, d / f"{c.id}.json")
            out[c.id] = {"same_as": c.same_as}
    return {"clips": out}


def load_transcripts(project: Project) -> dict[str, dict[str, Any]]:
    d = transcripts_dir(project)
    return {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(d.glob("C*.json"))} \
        if d.exists() else {}


def words_between(tr: dict[str, Any], start_s: float, end_s: float) -> list[dict[str, Any]]:
    """Words whose start lies in [start_s, end_s); unaligned words follow their segment."""
    out = []
    for seg in tr.get("segments") or []:
        if seg["end"] <= start_s or seg["start"] >= end_s:
            continue
        words = seg.get("words") or [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]
        for w in words:
            t = w["start"] if w.get("start") is not None else seg["start"]
            if start_s <= t < end_s:
                out.append(w)
    return out


def speech_line(tr: dict[str, Any] | None, start_s: float, end_s: float) -> str | None:
    """The L0 line under a scene: what is said in it, truncated."""
    if not tr:
        return None
    text = " ".join(w["word"] for w in words_between(tr, start_s, end_s)).strip()
    if not text:
        return None
    if len(text) > L0_SPEECH_CHARS:
        text = text[:L0_SPEECH_CHARS - 1] + "…"
    return f'  speech: "{text}"'


def transcript_text(project: Project, ref: str | None = None, words: bool = False) -> str:
    """Transcript as timed lines for a clip (``C01``), a scene (``S03``) or the whole project.

    Times use the EditPlan source origin: clip seconds, or **seconds into the scene** for a scene
    (as ``in_s``/``out_s`` and ``S03@t`` take; scene words also carry their frame,
    ``word@1.20/f36``). ``words`` adds ``word@start`` lines."""
    from opencine.core.time import Rate, to_frame
    from opencine.index.model import load_index

    index = load_index(project)
    trs = load_transcripts(project)
    if not trs:
        raise SpeechError(f"project {project.pid} has no transcript yet; run transcribe(pid)")
    scene = None
    if ref and ref.upper().startswith("S"):
        scene = index.scene(ref.upper())
        ranges = [(scene.clip, scene.start_s, scene.end_s)]
    elif ref:
        c = index.clip(ref.upper())
        ranges = [(c.id, 0.0, c.duration_s)]
    else:
        ranges = [(c.id, 0.0, c.duration_s) for c in index.clips]
    rate = Rate(index.rate)
    lines = []
    for clip, a, b in ranges:
        tr = trs.get(clip)
        head = f"## {clip}"
        if scene is not None:
            head += f" · {scene.id} · seconds from the start of {scene.id} (for in_s/out_s; /f = frame)"
        lines.append(head + (f" · {tr.get('language')}" if tr and tr.get("language") else ""))
        if not tr or not tr.get("segments"):
            lines.append("(no speech)")
            continue
        zero = a if scene is not None else 0.0  # origin of the printed times

        def ts(t: float, zero=zero, a=a, b=b) -> str:
            return fmt_ts(min(max(t, a), b) - zero, 2)

        for seg in tr["segments"]:
            if seg["end"] <= a or seg["start"] >= b:
                continue
            lines.append(f"[{ts(seg['start'])}–{ts(seg['end'])}] {seg['text'].strip()}")
            if not words:
                continue
            out = []
            for w in seg.get("words") or []:
                if w.get("start") is None:
                    out.append(w["word"])
                elif scene is None:
                    out.append(f"{w['word']}@{w['start']:.2f}")
                elif a <= w["start"] < b:
                    rel = w["start"] - zero
                    out.append(f"{w['word']}@{rel:.2f}/f{to_frame(round(rel, 3), rate, 'nearest')}")
            lines.append("  " + " ".join(out))
    return "\n".join(lines) + "\n"
