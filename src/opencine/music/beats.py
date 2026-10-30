"""Beat detection (librosa beat tracker, snapped to onset starts) and cut snapping.

``<file name>.beats.json`` (e.g. ``song.mp3.beats.json``) is a list of ``{"t": seconds,
"strength": 0–1}``, strength = onset-envelope peak relative to the track's strongest beat.
Extra keys (``downbeat``, ``bar``, …) are allowed and round-trip.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, TypeAdapter

from opencine.core.ffmpeg import run
from opencine.core.time import Rate, to_frame

SR = 22050
HOP = 128  # ~6 ms resolution


class Beat(BaseModel):
    model_config = ConfigDict(extra="allow")  # future keys (downbeat, bar, …) round-trip

    t: float
    strength: float
    downbeat: bool | None = None  # first beat of a bar, when the tracker knows
    bar: int | None = None


_Beats = TypeAdapter(list[Beat])


def _load_mono(path: Path) -> np.ndarray:
    """Decode any audio/video file to mono float at ``SR`` via ffmpeg."""
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        run(["-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_f32le", str(wav)])
        y, _ = sf.read(wav, dtype="float32")
    return y


def detect_beats(audio: str | Path) -> list[Beat]:
    y = _load_mono(Path(audio))
    if y.size < SR // 2 or not np.any(y):
        return []
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP)
    if len(frames) == 0:
        return []
    # the tracker lands on envelope peaks (tens of ms late): snap to a nearby backtracked onset
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True)
    near = round(0.06 * SR / HOP)
    win = round(0.05 * SR / HOP)
    out = []
    for f in frames:
        strength = float(env[max(0, f - win): f + win + 1].max())
        if len(onsets):
            o = int(onsets[np.argmin(np.abs(onsets - f))])
            f = o if abs(o - f) <= near else f
        out.append((float(librosa.frames_to_time(f, sr=SR, hop_length=HOP)), strength))
    top = max(s for _, s in out) or 1.0
    return [Beat(t=round(t, 4), strength=round(s / top, 3)) for t, s in out]


def beats_file(audio: str | Path) -> Path:
    """Path of ``audio``'s beat grid: ``<file name>.beats.json`` next to it."""
    audio = Path(audio)
    return audio.with_name(audio.name + ".beats.json")


def find_beats(audio: str | Path) -> Path | None:
    """The existing beat grid of ``audio``: ``beats_file``, else legacy ``<stem>.beats.json``."""
    audio = Path(audio)
    for p in (beats_file(audio), audio.with_name(audio.stem + ".beats.json")):
        if p.is_file():
            return p
    return None


def write_beats(audio: str | Path, out: str | Path | None = None) -> Path:
    """Detect beats and write the grid (default: ``beats_file(audio)``)."""
    audio = Path(audio)
    out = Path(out) if out else beats_file(audio)
    out.write_text(json.dumps([b.model_dump(exclude_none=True) for b in detect_beats(audio)], indent=1),
                   "utf-8")
    return out


def load_beats(path: str | Path) -> list[Beat]:
    return _Beats.validate_json(Path(path).read_text("utf-8"))


def snap(t: float, beats: list[Beat], rate: Rate, tol_frames: int = 2) -> int:
    """Frame (at ``rate``) for a cut at ``t`` s: the nearest beat's frame if within
    ``±tol_frames``, else ``t``'s nearest frame."""
    f = to_frame(t, rate, "nearest")
    best = min((to_frame(b.t, rate, "nearest") for b in beats), key=lambda bf: abs(bf - f), default=None)
    return best if best is not None and abs(best - f) <= tol_frames else f
