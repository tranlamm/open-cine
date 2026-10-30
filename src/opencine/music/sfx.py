"""Sound effects: a small deterministic, rights-free library synthesised with numpy, plus real files.

Lookup: ``<project>/sfx/<name>.*`` → ``OPENCINE_SFX_DIR/<name>.*`` → synthesised
``<project>/sfx/_synth/<name>.wav`` (written on first use). Stereo 48 kHz WAV peaking at −1 dBFS;
plans set the level with ``sfx.gain``. ``import_pack`` adds packs (Kenney CC0, Sonniss GDC, …) as
``OPENCINE_SFX_DIR/<pack>/<name>.wav`` with tags in ``index.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from opencine.core.config import get_settings

if TYPE_CHECKING:
    from opencine.core.project import Project

SR = 48000
EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
PEAK_DB = -1.0
SYNTH_DIR = "sfx/_synth"
DURATION = {"whoosh": 0.6, "swish": 0.3, "pop": 0.12, "click": 0.03, "tick": 0.04,
            "riser": 1.5, "impact": 1.0, "slice": 0.25, "scribble": 0.45}
NAMES = tuple(DURATION)
VERSION = 1  # bump when a recipe changes so synthesised files are rewritten
INDEX = "index.json"
TAG_NOISE = {"wav", "mp3", "ogg", "flac", "sfx", "sound", "sounds", "the", "and", "with", "for",
             "audio", "fx", "kenney", "sonniss", "gdc"}


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{name}:{VERSION}".encode()))


def _t(name: str) -> np.ndarray:
    return np.arange(round(DURATION[name] * SR)) / SR


def _one_pole(x: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Low-pass with a time-varying cutoff (Hz per sample)."""
    a = np.exp(-2 * np.pi * np.clip(cutoff, 20, SR / 2.2) / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, (v, k) in enumerate(zip(x, a, strict=True)):
        acc = (1 - k) * v + k * acc
        y[i] = acc
    return y


def _band(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Band of ``x`` between two moving cutoffs (difference of two low-passes)."""
    return _one_pole(x, hi) - _one_pole(x, lo)


def _bell(t: np.ndarray, peak: float) -> np.ndarray:
    p = t / t[-1]
    up = np.sin(np.pi / 2 * np.clip(p / peak, 0, 1)) ** 2
    down = np.cos(np.pi / 2 * np.clip((p - peak) / (1 - peak), 0, 1)) ** 2
    return np.where(p < peak, up, down)


def _sweep(t: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Phase of an exponential sweep f0 → f1 over ``t``."""
    k = np.log(f1 / f0) / t[-1]
    return 2 * np.pi * f0 * (np.exp(k * t) - 1) / k


def _whoosh(name: str, lo: float, hi: float, peak: float) -> np.ndarray:
    t = _t(name)
    noise = _rng(name).standard_normal(t.size)
    env = _bell(t, peak)
    centre = lo + (hi - lo) * env  # brighter as it passes by
    return _band(noise, centre * 0.5, centre * 1.6) * env


def synth(name: str) -> np.ndarray:
    """Stereo float32 ``(n, 2)`` of a library sound, peak-normalised to ``PEAK_DB``."""
    if name not in DURATION:
        raise KeyError(f"no synthesised sfx {name!r}; have {', '.join(NAMES)}")
    t = _t(name)
    rng = _rng(name)
    pan = np.full(t.size, 0.5)
    if name == "whoosh":
        x = _whoosh(name, 300, 2600, 0.55)
        pan = np.clip(t / t[-1], 0.15, 0.85)  # passes left → right
    elif name == "swish":
        x = _whoosh(name, 900, 5000, 0.4)
        pan = np.clip(t / t[-1], 0.25, 0.75)
    elif name == "pop":
        env = np.minimum(1, t / 0.002) * np.exp(-t / 0.03)
        x = np.sin(_sweep(t, 900, 250)) * env + 0.2 * rng.standard_normal(t.size) * np.exp(-t / 0.002)
    elif name == "click":
        n = rng.standard_normal(t.size)
        x = (n - _one_pole(n, np.full(t.size, 2500.0))) * np.exp(-t / 0.004)
    elif name == "tick":
        x = np.sin(2 * np.pi * 2400 * t) * np.minimum(1, t / 0.001) * np.exp(-t / 0.008)
    elif name == "riser":
        p = t / t[-1]
        noise = rng.standard_normal(t.size)
        x = _band(noise, 150 + 2000 * p ** 2, 400 + 9000 * p ** 2) + 0.4 * np.sin(_sweep(t, 200, 1200))
        x *= p ** 2.5 * np.minimum(1, (t[-1] - t) / 0.01)
    elif name == "impact":
        body = np.sin(_sweep(t, 90, 45)) * np.exp(-t / 0.25)
        hit = _one_pole(rng.standard_normal(t.size), np.full(t.size, 800.0)) * np.exp(-t / 0.06)
        x = np.tanh(2.5 * (body + 1.5 * hit)) * np.minimum(1, t / 0.001)
    elif name == "scribble":  # marker on paper: band noise in back-and-forth strokes
        strokes = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4.5 * t + 0.4))
        x = _band(rng.standard_normal(t.size), np.full(t.size, 1400.0), np.full(t.size, 5200.0))
        x *= strokes * _bell(t, 0.2)
    else:  # slice: a bright scrape and a short metallic ring
        n = rng.standard_normal(t.size)
        scrape = (n - _one_pole(n, np.full(t.size, 3000.0))) * np.exp(-t / 0.04)
        x = scrape + 0.5 * np.sin(2 * np.pi * 5500 * t) * np.exp(-t / 0.08)
    st = np.stack([x * np.sqrt(1 - pan), x * np.sqrt(pan)], axis=1)
    return (st / max(1e-9, float(np.abs(st).max())) * 10 ** (PEAK_DB / 20)).astype(np.float32)


def write(name: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, synth(name), SR, subtype="PCM_16")
    tmp.replace(path)
    return path


def write_set(directory: Path, names: tuple[str, ...] = NAMES, *, force: bool = False) -> list[Path]:
    """Write the library (``<name>.wav``) into ``directory``; existing files are kept unless
    ``force``."""
    out = []
    for n in names:
        p = Path(directory) / f"{n}.wav"
        out.append(p if p.exists() and not force else write(n, p))
    return out


def _real(d: Path, name: str) -> Path | None:
    return next((p for ext in EXTS if (p := d / f"{name}{ext}").is_file()), None)


def find(project: Project, name: str, *, synthesise: bool = True) -> Path | None:
    """The file a plan should use for ``name`` (see the module doc), or None."""
    for d in [project.root / "sfx", *([Path(get_settings().sfx_dir)] if get_settings().sfx_dir else [])]:
        if p := _real(d, name):
            return p
    if name not in DURATION or not synthesise:
        return None
    p = project.root / SYNTH_DIR / f"{name}.wav"
    stamp = p.with_suffix(".v")
    if not p.exists() or not stamp.exists() or stamp.read_text() != str(VERSION):
        write(name, p)
        stamp.write_text(str(VERSION))
    return p


def duration(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except (RuntimeError, sf.LibsndfileError):
        return 0.5


def load_index(d: Path) -> list[dict[str, Any]]:
    """Sounds imported into library ``d`` (``index.json``, written by ``import_pack``)."""
    try:
        return json.loads((d / INDEX).read_text("utf-8"))["sounds"]
    except (OSError, ValueError, KeyError):
        return []


def _tags(*parts: str) -> list[str]:
    split = (re.sub(r"([a-z])([A-Z])", r"\1 \2", part) for part in parts)  # impactMetal → impact metal
    words = (w for part in split for w in re.findall(r"[a-z]+", part.lower()))
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in TAG_NOISE))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "sound"


def import_pack(src: Path, pack: str, *, license: str, source: str | None = None,
                dest: Path | None = None) -> dict[str, Any]:
    """Import every sound of ``src`` (folder or ``.zip``) into ``<dest>/<pack>/`` as 48 kHz stereo
    WAV peaking at ``PEAK_DB`` (same gain scale as the synthesised set), tagged from folder and file
    names, and rewrite ``<dest>/index.json``. ``dest`` defaults to ``OPENCINE_SFX_DIR``; re-importing
    a pack replaces it."""
    from opencine.core.ffmpeg import run

    dest = Path(dest or get_settings().sfx_dir or "")
    if not str(dest):
        raise ValueError("set OPENCINE_SFX_DIR (or pass dest) for the library")
    pack = _slug(pack)
    out_dir = dest / pack
    sounds: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(src)
        if root.suffix.lower() == ".zip":
            with zipfile.ZipFile(root) as z:
                z.extractall(tmp)
            root = Path(tmp)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
        if not files:
            raise ValueError(f"no sounds ({', '.join(EXTS)}) in {src}")
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        for f in files:
            rel = f.relative_to(root)
            stem = name = _slug(rel.stem)
            for k in range(2, len(files) + 2):  # de-duplicate names that slug the same
                if not (out_dir / f"{name}.wav").exists():
                    break
                name = f"{stem}-{k}"
            wav = Path(tmp) / "decoded.wav"
            run(["-y", "-i", str(f), "-ar", str(SR), "-ac", "2", "-c:a", "pcm_f32le", str(wav)])
            x, _ = sf.read(wav, dtype="float32", always_2d=True)
            if not x.size or not (peak := float(np.abs(x).max())):
                continue  # silent file
            sf.write(out_dir / f"{name}.wav", x / peak * 10 ** (PEAK_DB / 20), SR, subtype="PCM_16")
            sounds.append({"name": f"{pack}/{name}", "file": f"{pack}/{name}.wav",
                           "tags": _tags(pack, rel.with_suffix("").as_posix()),
                           "duration_s": round(len(x) / SR, 2), "license": license,
                           **({"source": source} if source else {})})
    kept = [s for s in load_index(dest) if not s["name"].startswith(f"{pack}/")]
    (dest / INDEX).write_text(json.dumps({"sounds": kept + sounds}, indent=1, ensure_ascii=False), "utf-8")
    return {"pack": pack, "sounds": len(sounds), "dir": str(out_dir), "index": str(dest / INDEX)}


def library(project: Project, query: str | None = None, limit: int | None = None) -> dict[str, Any]:
    """Sound names a plan can use: project ``sfx/``, ``OPENCINE_SFX_DIR`` (imported index, else flat
    files), the synthesised set. ``query``: words that must all match a name or tag; ``limit`` caps
    the list (``total`` counts all matches)."""
    seen: dict[str, dict[str, Any]] = {}

    def add(entry: dict[str, Any]) -> None:
        seen.setdefault(entry["name"], entry)

    def flat(d: Path, source: str) -> None:
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in EXTS:
                    add({"name": p.stem, "source": source, "file": str(p), "tags": _tags(p.stem)})

    d = Path(get_settings().sfx_dir) if get_settings().sfx_dir else None
    index = load_index(d) if d else []
    flat(project.root / "sfx", "project")
    if d and not index:
        flat(d, "OPENCINE_SFX_DIR")  # a plain folder of <name>.wav overrides synthesised names
    for n in NAMES:  # before imported packs, so a capped unfiltered list still shows them
        add({"name": n, "source": "synth", "file": f"{SYNTH_DIR}/{n}.wav", "tags": [n],
             "duration_s": DURATION[n]})
    for s in index:
        add({**s, "source": "OPENCINE_SFX_DIR", "file": str(d / s["file"])})
    words = (query or "").lower().split()
    hits = [s for s in seen.values()
            if all(w in s["name"].lower() or any(w in t for t in s.get("tags", [])) for w in words)]
    return {"total": len(hits), "sfx": hits[:limit] if limit else hits}
