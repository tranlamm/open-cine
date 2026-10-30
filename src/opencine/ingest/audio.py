"""Source audio → 48 kHz stereo PCM WAV (the renderer's and beat tracker's input).

WAV sample 0 is file time 0, like proxy frame 0: an audio stream that starts late (or early)
in the container is padded (or trimmed) so the WAV stays in sync with the proxy.
"""

from __future__ import annotations

from pathlib import Path

from opencine.core.ffmpeg import run

SAMPLE_RATE = 48_000
ALIGN = "aresample=async=1:first_pts=0"  # pad/trim so the first sample sits at file time 0
PARAMS = {"ar": SAMPLE_RATE, "ac": 2, "codec": "pcm_s16le", "af": ALIGN}  # cache key part


def extract_audio(src: str | Path, out: str | Path) -> Path:
    run(["-y", "-i", str(src), "-map", "0:a:0", "-vn", "-af", ALIGN,
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)])
    return Path(out)
