"""Voice-over lines → ``audio/voice/<id>.wav`` (48 kHz mono, used as ``sfx`` in an EditPlan) plus
``<id>.words.json`` word timings for voice captions.

Uncached lines go in batch requests (≤ ``BATCH_MAX`` lines / ``BATCH_MAX_CHARS`` chars, the
gateway's ``LTS_MAX_BATCH_ITEMS`` / ``LTS_MAX_BATCH_TOTAL_CHARS``), or one ``/speech/file`` call.
Cached by text + language, so editing one line re-synthesises only that line.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from opencine.core.cache import cache_key
from opencine.core.config import get_settings
from opencine.core.ledger import Ledger
from opencine.core.time import to_frame
from opencine.index.steps import StepCache, place
from opencine.speech.client import TtsClient
from opencine.speech.words import time_words

if TYPE_CHECKING:
    from opencine.core.project import Project

VERSION = 1
BATCH_MAX = 32
BATCH_MAX_CHARS = 20000
SAMPLE_RATE = 48000
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class VoiceLine(BaseModel):
    id: str = Field(description="Line id → audio/voice/<id>.wav, e.g. 'v01'.")
    text: str = Field(min_length=1, max_length=5000, description="What to say (whole clauses read best).")
    language: str | None = Field(None, description="'en', 'vi'…; default OPENCINE_SPEECH_LANGUAGE.")


def voice_dir(project: Project) -> Path:
    return project.audio_dir / "voice"


def words_path(project: Project, line_id: str) -> Path:
    return voice_dir(project) / f"{line_id}.words.json"


def _batches(reqs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive requests within the gateway's per-batch line and character limits."""
    out: list[list[dict[str, Any]]] = []
    chars = 0
    for r in reqs:
        if not out or len(out[-1]) == BATCH_MAX or chars + len(r["text"]) > BATCH_MAX_CHARS:
            out.append([])
            chars = 0
        out[-1].append(r)
        chars += len(r["text"])
    return out


def _duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def synthesize(project: Project, lines: list[VoiceLine | dict[str, Any]],
               client: TtsClient | None = None, steps: StepCache | None = None) -> dict[str, Any]:
    items = [VoiceLine.model_validate(x) for x in lines]
    if bad := [i.id for i in items if not ID_RE.match(i.id)]:
        raise ValueError(f"bad line id(s) {bad}: letters, digits, - and _ only")
    if len({i.id for i in items}) != len(items):
        raise ValueError("line ids must be unique")
    steps = steps or StepCache(project.cache_dir)
    lang = get_settings().speech_language
    keys = {i.id: cache_key("tts", i.text, i.language or lang, SAMPLE_RATE, VERSION) for i in items}
    missing = [i for i in items if not steps.cache.hit("tts", keys[i.id], ".wav")]
    if missing:
        client = client or TtsClient.from_settings()
        reqs = [{"text": i.text, "language": i.language or lang, "sample_rate_hz": SAMPLE_RATE}
                for i in missing]
        wavs = [client.speech(reqs[0])] if len(reqs) == 1 else [
            w for batch in _batches(reqs) for w in client.batch(batch)]
        for item, data in zip(missing, wavs, strict=True):
            steps.file("tts", keys[item.id], ".wav", lambda out, d=data: out.write_bytes(d))
        Ledger(project).record("tts", "speech", sum(len(i.text) for i in missing),
                               quota_note="self-hosted TTS, characters", lines=len(missing))
    out = []
    stt = None
    for i in items:
        path = place(steps.cache.path("tts", keys[i.id], ".wav"), voice_dir(project) / f"{i.id}.wav")
        dur = _duration(path)

        def make_words(i=i, path=path) -> dict[str, Any]:
            nonlocal stt
            if stt is None and get_settings().stt_url:
                from opencine.speech.client import SttClient

                stt = SttClient.from_settings()
            return time_words(path, i.text, stt=stt, language=i.language or lang)

        # cache only STT timings; estimates are redone next time, when STT may be back
        wkey = cache_key("tts-words", keys[i.id])
        if hit := steps.cache.hit("tts-words", wkey, ".json"):
            words = json.loads(hit.read_text("utf-8"))
            place(hit, words_path(project, i.id))
        elif (words := make_words())["timing"] == "stt":
            steps.json("tts-words", wkey, lambda w=words: w)
            place(steps.cache.path("tts-words", wkey, ".json"), words_path(project, i.id))
        else:
            words_path(project, i.id).unlink(missing_ok=True)  # never write through a cache link
            words_path(project, i.id).write_text(json.dumps(words, ensure_ascii=False), "utf-8")
        out.append({"id": i.id, "file": path.relative_to(project.root).as_posix(),
                    "duration_s": round(dur, 3), "frames": to_frame(dur, project.rate, "ceil"),
                    "words": len(words["words"]), "timing": words["timing"]})
    return {"lines": out, "synthesized": len(missing), "cached": len(items) - len(missing),
            "hint": 'add to the EditPlan as sfx: {"file": <file>, "at": <timeline frame>, "gain": 0}'}
