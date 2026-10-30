"""Encode presets, segment encode, and the final concat (stream copy) + audio mux."""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from opencine.core.ffmpeg import X264_BT709, run

Preset = Literal["preview", "final"]
PREVIEW_SIZE = (720, 1280)  # every portrait preview, whatever the deliverable (landscape: turned)
# native AAC at 192k overshoots the true peak by up to ~4 dB on some material (measured
# −1.5 → +2.6 dBTP); 256k+ stays within 0.1 dB. 320k is what platforms take without complaint.
AAC_BITRATE = "320k"
AUDIO_RATE = 48000


@dataclass(frozen=True)
class EncodePreset:
    """Encoder settings only; the size comes from ``output_size``."""

    x264: tuple[str, ...]
    master: bool  # decode the original sources (final) instead of the 720p proxies

    @property
    def args(self) -> list[str]:
        # identical params (and BT.709 VUI) for every segment: the concat demuxer stream-copies
        # them and keeps the first segment's SPS for all
        return ["-c:v", "libx264", *self.x264, "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-x264-params", X264_BT709, "-video_track_timescale", "90000"]


PRESETS: dict[str, EncodePreset] = {
    "preview": EncodePreset(("-preset", "veryfast", "-crf", "23"), master=False),
    "final": EncodePreset(("-preset", "medium", "-crf", "18"), master=True),
}


def output_size(preset: Preset, deliverable: tuple[int, int]) -> tuple[int, int]:
    """Final renders at the plan's deliverable; previews at 720×1280 (1280×720 for landscape)."""
    if preset == "final":
        return deliverable
    return PREVIEW_SIZE if deliverable[1] >= deliverable[0] else PREVIEW_SIZE[::-1]


def encode_segment(inputs: list[str], graph: str, out_label: str, frames: int,
                   preset: EncodePreset, out: Path) -> None:
    """The graph goes in a file next to ``out`` (``-/filter_complex``): keyed crop/zoom graphs
    outgrow the 32 767-character Windows command line."""
    tmp = out.with_suffix(".tmp.mp4")
    script = out.with_suffix(".filtergraph")
    script.write_text(graph, "utf-8")
    run(["-y", "-loglevel", "error", *inputs, "-/filter_complex", str(script), "-map", f"[{out_label}]",
         "-an", *preset.args, "-frames:v", str(frames), str(tmp)])
    os.replace(tmp, out)


def concat_entry(p: Path, frames: int, rate: Fraction) -> str:
    # exact duration: without it the demuxer places the next segment at the container
    # duration, rounded to whole ms, and the video drifts against the audio cut by cut
    path = p.resolve().as_posix().replace("'", "'\\''")
    return f"file '{path}'\nduration {float(Fraction(frames) / rate):.9f}"


def mux(segments: list[tuple[Path, int]], rate: Fraction, audio: Path, list_path: Path,
        out: Path) -> None:
    """Concat ``(segment, frames)`` by stream copy and mux the normalised audio as AAC."""
    list_path.write_text("\n".join(concat_entry(p, n, rate) for p, n in segments) + "\n", "utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    run(["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path),
         "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", AAC_BITRATE, "-ar", str(AUDIO_RATE), "-movflags", "+faststart",
         str(tmp)])
    os.replace(tmp, out)
