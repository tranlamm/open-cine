"""A project's music tracks: ``music/<name>`` plus beat grid ``music/<name>.beats.json``, which the
Director and ``plan_save``'s craft hints cut against.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from opencine.core.cache import sha256_file
from opencine.core.ffmpeg import probe_json
from opencine.core.time import to_frame
from opencine.music.beats import Beat, beats_file, find_beats, load_beats, write_beats

if TYPE_CHECKING:
    from opencine.core.project import Project

STRONG_MIN = 0.6  # minimum strength of a strong beat (0–1, relative to the track's strongest)
FIRST = 8  # beats listed in the summary
STRONG_FRAMES = 32


def music_dir(project: Project) -> Path:
    return project.root / "music"


def beats_path(track: Path) -> Path:
    """The grid to read: ``<name>.beats.json`` (or legacy ``<stem>.beats.json``)."""
    return find_beats(track) or beats_file(track)


def _safe_name(name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    return (re.sub(r"[^\w.-]+", "_", stem).strip("._") or "music") + ext


def add_music(project: Project, path: str | Path) -> dict[str, Any]:
    """Copy a track (any file with an audio stream) into ``music/`` and detect its beats.

    Identical content is not copied twice; a different file with a taken name gets ``-2``, ``-3``…"""
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(src)
    info = probe_json(src)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise ValueError(f"{src.name} has no audio stream")
    d = music_dir(project)
    d.mkdir(parents=True, exist_ok=True)
    sha = sha256_file(src)
    name = _safe_name(src.name)
    dst, k = d / name, 1
    while dst.exists() and sha256_file(dst) != sha:
        k += 1
        dst = d / f"{Path(name).stem}-{k}{Path(name).suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
    if not beats_file(dst).exists():
        write_beats(dst, beats_file(dst))
    return summary(project, dst.name)


def track(project: Project, name: str) -> Path:
    """``name`` (``x.mp3`` or ``music/x.mp3``) → the track file in ``music/``."""
    p = music_dir(project) / Path(name).name
    if not p.is_file():
        have = sorted(f.name for f in music_dir(project).glob("*") if not f.name.endswith(".beats.json")) \
            if music_dir(project).exists() else []
        raise FileNotFoundError(f"no track {name!r} in music/; have {have or 'none (music_add first)'}")
    return p


def strong_threshold(beats: list[Beat]) -> float:
    """Strong-beat threshold: halfway from the median strength to the top (≥ ``STRONG_MIN``)."""
    return max(STRONG_MIN, (median(b.strength for b in beats) + 1) / 2) if beats else 1.0


def _bpm(beats: list[Beat]) -> float | None:
    if len(beats) < 2 or beats[-1].t <= beats[0].t:
        return None
    return round(60 * (len(beats) - 1) / (beats[-1].t - beats[0].t), 1)


def summary(project: Project, name: str) -> dict[str, Any]:
    p = track(project, name)
    beats = load_beats(beats_path(p))
    rate = project.rate
    dur = float(probe_json(p)["format"].get("duration") or 0)
    thr = strong_threshold(beats)
    strong = [b for b in beats if b.strength >= thr]
    return {
        "file": p.relative_to(project.root).as_posix(),  # EditPlan music.file
        "duration_s": round(dur, 2),
        "bpm": _bpm(beats),
        "beats": len(beats),
        "first_beats_s": [round(b.t, 2) for b in beats[:FIRST]],
        "strong_frames": [to_frame(b.t, rate, "nearest") for b in strong[:STRONG_FRAMES]],
        "rate": str(rate),
    }


def beats_text(project: Project, name: str) -> str:
    """The grid as compact text, 8 beats a line: ``1.50/f45*`` (seconds/frame, * = strong,
    ``|`` before a downbeat when marked)."""
    p = track(project, name)
    beats = load_beats(beats_path(p))
    rate = project.rate
    bpm, thr = _bpm(beats), strong_threshold(beats)
    head = f"# music/{p.name} · {bpm or '?'} BPM · {len(beats)} beat · seconds/frame @{rate} fps (* = strong)"
    cells = []
    for b in beats:
        cell = f"{b.t:.2f}" + f"/f{to_frame(b.t, rate, 'nearest')}"
        cell += "*" if b.strength >= thr else ""
        cells.append(("|" if b.downbeat else "") + cell)
    lines = [" ".join(cells[i:i + 8]) for i in range(0, len(cells), 8)]
    return "\n".join([head, *(lines or ["(no beats)"])]) + "\n"
