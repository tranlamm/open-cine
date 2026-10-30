"""Thin wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from opencine.core.config import get_settings
from opencine.core.proc import NO_WINDOW

# Every picture we write is SDR BT.709, limited range. swscale converts RGB/untagged input with
# BT.601 coefficients unless told otherwise, and ffmpeg 9 drops CLI colour options on filtered
# RGB input, so the conversion is explicit (``TO_BT709`` on a ``scale``), the frames are tagged
# (``BT709_TAGS``) and so is the H.264 VUI (``X264_BT709``) — identical in every segment.
TO_BT709 = "out_color_matrix=bt709:out_range=tv"
BT709_TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
X264_BT709 = "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"


class FFmpegError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        super().__init__(f"{Path(cmd[0]).name} exited {returncode}:\n{tail}")
        self.cmd, self.returncode, self.stderr = cmd, returncode, stderr


def run(args: list[str], *, binary: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with ``args`` (no leading binary). Raises ``FFmpegError`` on failure."""
    cmd = [binary or get_settings().ffmpeg, "-hide_banner", "-nostdin", *args]
    return _check(cmd, _launch(cmd))


def _launch(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Start the binary; failing to start it (missing, or a command line over the OS limit —
    WinError 206 surfaces as FileNotFoundError) is an ``FFmpegError`` like any other failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              **NO_WINDOW)
    except OSError as e:
        raise FFmpegError(cmd, -1, f"could not start ({sum(map(len, cmd))} chars): {e}") from e


def _check(cmd: list[str], proc: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return proc


def probe_json(path: str | Path, *extra: str) -> dict[str, Any]:
    """``ffprobe -show_format -show_streams`` as a dict."""
    cmd = [get_settings().ffprobe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", *extra, str(path)]
    return json.loads(_check(cmd, _launch(cmd)).stdout)
