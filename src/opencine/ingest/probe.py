"""ffprobe → the facts ingest needs: codec, rates, VFR, rotation, HDR, size, audio, and when
the clip was shot (container tags, else a date-time in the file name)."""

from __future__ import annotations

import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from statistics import median

from pydantic import BaseModel

from opencine.core.ffmpeg import probe_json

HDR_TRANSFERS = {"arib-std-b67": "hlg", "smpte2084": "pq"}
VFR_PACKETS = 600  # packets inspected for timestamp jitter
VFR_JITTER = 0.10  # (max Δpts − min Δpts − 1 tick) / median Δpts above this ⇒ VFR
VFR_RATE_GAP = 0.02  # |nominal − avg| / nominal above this ⇒ VFR
LOW_RES_SHORT_SIDE = 720  # below this a 1080×1920 final upscales ≥ 1.5× (e.g. chat-app compressed)
PARAMS = {"shot_at": 1}  # cache key part of the probe step
# QuickTime's creationdate is local wall-clock time with its offset; creation_time is UTC
SHOT_TAGS = ("com.apple.quicktime.creationdate", "creation_time")
SHOT_MIN_YEAR = 2000  # earlier dates are encoder placeholders (1904, 1970)
_NAME_TIME = re.compile(r"(?<!\d)(20\d{2})[-_.]?([01]\d)[-_.]?([0-3]\d)[ _T.-]?([0-2]\d)[-_.:h]?([0-5]\d)[-_.:m]?([0-5]\d)(?:\.?\d{3})?(?!\d)")


def low_res(width: int, height: int) -> bool:
    """True when the source is too small for a 1080×1920 final without visible upscaling."""
    return min(width, height) < LOW_RES_SHORT_SIDE


class ProbeInfo(BaseModel):
    codec: str
    fps_nominal: str  # r_frame_rate
    fps_avg: str  # avg_frame_rate
    vfr: bool
    rotation: int  # clockwise degrees applied on display: 0 | 90 | 180 | 270
    hdr: str | None  # "hlg" | "pq" | None
    pix_fmt: str
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    width: int  # display orientation (after rotation)
    height: int
    duration_s: float
    has_audio: bool
    shot_at: str | None = None  # ISO local time the recording started (tags), None if unknown


def _frac(text: str | None) -> Fraction:
    try:
        return Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def stream_rotation(stream: dict) -> int:
    """Clockwise degrees a player turns the stream on display: 0 | 90 | 180 | 270."""
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:  # display matrix: counter-clockwise degrees
            return int(-float(sd["rotation"])) % 360
    return int(float(stream.get("tags", {}).get("rotate", 0))) % 360  # legacy tag: clockwise


def _pts_jitter(path: Path) -> float:
    data = probe_json(path, "-select_streams", "v:0", "-show_entries", "packet=pts",
                      "-read_intervals", f"%+#{VFR_PACKETS}")
    pts = sorted(int(p["pts"]) for p in data.get("packets", []) if p.get("pts") is not None)
    deltas = [b - a for a, b in zip(pts, pts[1:], strict=False) if b > a]
    if len(deltas) < 3:
        return 0.0
    # one tick of spread is timestamp rounding (e.g. 120 fps in a 1 ms timebase: 8/9 ms), not VFR
    return max(0, max(deltas) - min(deltas) - 1) / median(deltas)


def _iso(dt: datetime) -> str | None:
    if dt.year < SHOT_MIN_YEAR:
        return None
    return (dt if dt.tzinfo and dt.utcoffset() else dt.astimezone()).isoformat(timespec="seconds")


def shot_time(data: dict) -> str | None:
    """When the recording started, from ffprobe tags (format, then video stream): QuickTime's
    local ``creationdate`` keeps its own offset; a UTC ``creation_time`` is shown in this
    machine's zone. None when absent or a placeholder."""
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    for tags in (data.get("format", {}).get("tags") or {}, video.get("tags") or {}):
        low = {k.lower(): v for k, v in tags.items()}
        for key in SHOT_TAGS:
            if (v := low.get(key)) and isinstance(v, str):
                try:
                    if out := _iso(datetime.fromisoformat(v.strip())):
                        return out
                except ValueError:
                    continue
    return None


def shot_from_name(name: str) -> str | None:
    """A date-time in a camera file name (``VID_20261001_061233.mp4``, ``PXL_20261001_061233``,
    ``2026-10-01 06.12.33.mov``) as ISO local time, or None."""
    m = _NAME_TIME.search(name)
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups())).isoformat(timespec="seconds")
    except ValueError:
        return None


def probe(path: str | Path) -> ProbeInfo:
    path = Path(path)
    data = probe_json(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(f"{path.name}: no video stream")

    nominal, avg = _frac(video.get("r_frame_rate")), _frac(video.get("avg_frame_rate"))
    rate_gap = abs(nominal - avg) / nominal if nominal and avg else 0
    vfr = bool(rate_gap > VFR_RATE_GAP or _pts_jitter(path) > VFR_JITTER)

    rotation = stream_rotation(video)
    w, h = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        w, h = h, w

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    transfer = video.get("color_transfer")
    return ProbeInfo(
        codec=video.get("codec_name", "?"),
        fps_nominal=str(nominal),
        fps_avg=str(avg),
        vfr=vfr,
        rotation=rotation,
        hdr=HDR_TRANSFERS.get(transfer or ""),
        pix_fmt=video.get("pix_fmt", "?"),
        color_primaries=video.get("color_primaries"),
        color_transfer=transfer,
        color_space=video.get("color_space"),
        width=w,
        height=h,
        duration_s=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        shot_at=shot_time(data),
    )
