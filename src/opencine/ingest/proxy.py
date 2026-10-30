"""720p CFR edit proxy: rotation applied, HDR tone-mapped to SDR BT.709, short GOP, no audio.

Colour: every proxy is converted to BT.709 limited range explicitly and tagged so. A source
without a matrix tag is assumed BT.601 below 720p (and for generated media) and BT.709 at or
above it. HDR is tone-mapped *after* the downscale (fewer pixels) with error-diffusion dither,
so 10-bit gradients do not band in 8 bits.
"""

from __future__ import annotations

import math
from pathlib import Path

from opencine.core.ffmpeg import BT709_TAGS, TO_BT709, X264_BT709, run
from opencine.core.time import Rate
from opencine.ingest.probe import ProbeInfo

SHORT_SIDE = 720  # never upscaled
GOP_S = 0.5
CRF = 20
PRESET = "veryfast"
COLOUR = 2  # 2: explicit BT.709 conversion + dithered tone-map after the downscale
# what shapes the proxy's frames (size, rate, GOP) — and so the shot cuts found on it — as
# opposed to its colour: cut detection is keyed on this, so a colour change does not move cuts
GEOMETRY_PARAMS = {"short_side": SHORT_SIDE, "gop_s": GOP_S, "crf": CRF, "preset": PRESET}
PARAMS = {**GEOMETRY_PARAMS, "colour": COLOUR}  # cache key part of the proxy itself
HD_SHORT_SIDE = 720  # untagged sources at least this big are taken as BT.709, smaller as BT.601
_UNKNOWN = {None, "", "unknown", "unspecified", "reserved"}


def _tonemap(info: ProbeInfo) -> str:
    # tag frames explicitly so zscale always has a path, then linear → hable → BT.709, dithered
    # to the 8-bit output (zscale does not dither by default: smooth skies band)
    tags = (f"setparams=color_primaries={info.color_primaries or 'bt2020'}"
            f":color_trc={info.color_transfer}:colorspace={info.color_space or 'bt2020nc'}")
    return (f"{tags},zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv:d=error_diffusion")


def untagged_matrix(width: int, height: int, *, generated: bool = False) -> str:
    """The YCbCr matrix to assume for a stream without a matrix tag."""
    return "bt470bg" if generated or min(width, height) < HD_SHORT_SIDE else "bt709"


def input_tags(color_space: str | None, width: int, height: int, *, generated: bool = False) -> str:
    """``setparams`` naming the matrix of an untagged SDR stream ("" when it is tagged), so the
    BT.709 conversion downstream starts from the right coefficients."""
    if color_space not in _UNKNOWN:
        return ""
    return f"setparams=colorspace={untagged_matrix(width, height, generated=generated)}"


def _scale(short_side: int, colour: bool) -> str:
    s = f"trunc(min({short_side},%s)/2)*2"
    return (f"scale=w='if(gte(iw,ih),-2,{s % 'iw'})':h='if(gte(iw,ih),{s % 'ih'},-2)'"
            ":flags=bicubic" + (f":{TO_BT709}" if colour else ""))


def decode_filters(info: ProbeInfo, rate: Rate, short_side: int, *, fps: bool = True) -> tuple[str, str]:
    """``(prefilter, tonemap)`` that decode the original like the proxy, for the renderer.

    The prefilter normalises size (even, ``short_side``) and, with ``fps``, the frame rate; it
    leaves colour alone. ``tonemap`` (HDR only) is applied by the renderer after its crop and
    scale, so it runs on the output pixels rather than the whole 4K frame. Without ``fps`` the
    native frames are kept (speed changes resample them themselves)."""
    tags = "" if info.hdr else input_tags(info.color_space, info.width, info.height)
    chain = [tags] if tags else []
    chain += [_scale(short_side, colour=False), "setsar=1"] + ([f"fps={rate}"] if fps else [])
    return ",".join(chain), _tonemap(info) if info.hdr else ""


def proxy_filter(info: ProbeInfo, rate: Rate, short_side: int = SHORT_SIDE) -> str:
    """Video filter chain. ffmpeg autorotates before it, so ``iw/ih`` are display-oriented."""
    if info.hdr:
        chain = [_scale(short_side, colour=False), "setsar=1", f"fps={rate}", _tonemap(info)]
    else:
        tags = input_tags(info.color_space, info.width, info.height)
        chain = ([tags] if tags else []) + [_scale(short_side, colour=True), "setsar=1", f"fps={rate}"]
    return ",".join([*chain, "format=yuv420p", BT709_TAGS])


def make_proxy(src: str | Path, out: str | Path, info: ProbeInfo, rate: Rate) -> Path:
    gop = max(1, math.ceil(rate.fps * GOP_S))
    run([
        "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", proxy_filter(info, rate), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-x264-params", X264_BT709,
        "-movflags", "+faststart", str(out),
    ])
    return Path(out)
