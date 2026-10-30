"""Generated background music: an instrumental from ACE-Step 1.5 on the private ZeroGPU Space
(``OPENCINE_HF_SPACE_MUSIC``), saved to ``music/`` and beat-tracked like ``music_add``.

Cached by caption, length, BPM, key and seed (no seed = one derived from the inputs, so a repeat
call returns the same track). ACE-Step is MIT (code and weights); output is free for commercial use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencine.core.cache import cache_key
from opencine.core.ledger import Ledger
from opencine.music.library import add_music, music_dir

if TYPE_CHECKING:
    from opencine.core.project import Project

VERSION = 1
MAX_SEED = 2**31 - 1
OCTAVE_TOLERANCE = 0.08  # detected/asked BPM within 8 % of ×2 or ×½ = tracker locked an octave off


def generate(project: Project, caption: str, seconds: float, bpm: int | None = None,
             key: str | None = None, seed: int | None = None, provider: Any = None) -> dict[str, Any]:
    """Make (or reuse) a track: ``music_add``'s summary plus model, seed, the model's ``meta`` and a
    ``warning`` when the detected BPM is an octave off."""
    from opencine.gen.providers.hf_space import HfSpaceProvider

    inputs = (caption.strip(), round(float(seconds), 1), bpm, (key or "").strip())
    if seed is None:
        seed = int(cache_key("music-seed", *inputs)[:8], 16) % MAX_SEED
    name = f"gen-{cache_key('music', *inputs, seed, VERSION)[:12]}.wav"
    out = music_dir(project) / name
    meta: dict[str, Any] = {}
    if not out.exists():
        prov = provider or HfSpaceProvider()
        res = prov.music(inputs[0], inputs[1], out=out, bpm=bpm, key=key, seed=seed)
        meta = res.extra.get("meta", {})
        Ledger(project).record(prov.name, "gen_music", res.quota_used, usd_est=res.usd_est,
                               quota_note=f"{res.quota_used:g} {res.quota_unit}",
                               model=res.model, path=f"music/{name}", seconds=inputs[1])
    summary = add_music(project, out)
    found = summary.get("bpm")
    warning = None
    if bpm and found:
        ratio = float(found) / bpm
        if any(abs(ratio - r) <= OCTAVE_TOLERANCE * r for r in (2.0, 0.5)):
            warning = (f"beat tracker found {found} BPM for a {bpm} BPM request (an octave off): "
                       "cut on every other beat, or regenerate with another seed")
    return {**summary, "model": "ACE-Step-1.5", "seed": seed, "caption": inputs[0],
            "meta": meta, **({"warning": warning} if warning else {})}
