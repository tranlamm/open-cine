"""Local GPU help for final renders: Real-ESRGAN and RIFE as ncnn/Vulkan binaries (no Python,
no CUDA; integrated GPUs work too).

Items that need it get a cached intermediate the segment decodes instead of the original: an
**upscale** (``realesrgan-ncnn-vulkan`` ×4, then Lanczos to 1:1 with the output) for a speed-1
video or still magnified ≥ ``UPSCALE_FROM``×, or an **interpolation** (``rife-ncnn-vulkan``) for
slow motion without enough native frames. Largest magnification first within
``OPENCINE_UPSCALE_MAX_S``; past it, without the binaries, or for HDR items the render falls back
to Lanczos + CAS / ``minterpolate`` and says so in its notes. ``opencine tools fetch esrgan|rife``
downloads the pinned Windows releases into ``OPENCINE_TOOLS_DIR``.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opencine.core.cache import cache_key
from opencine.core.config import get_settings
from opencine.core.ffmpeg import run
from opencine.core.proc import NO_WINDOW

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.media import Media
    from opencine.plan.schema import EditPlan

VERSION = 1
UPSCALE_FROM = 1.4
ESRGAN_SCALE = 4
LEAD = 3  # as the segment's seek (segments.LEAD)
INTERMEDIATE = ["-c:v", "libx264", "-preset", "medium", "-crf", "10", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv"]
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd,format=yuv420p"
# pinned releases: no published digests, so the size is pinned and the sha256 is recorded on
# the first download (OPENCINE_TOOLS_DIR/<tool>.sha256) and checked on every later one
RELEASES = {
    "esrgan": {"url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                      "realesrgan-ncnn-vulkan-20220424-windows.zip",
               "size": 45474481, "exe": "realesrgan-ncnn-vulkan"},
    "rife": {"url": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
                    "rife-ncnn-vulkan-20221029-windows.zip",
             "size": 431540241, "exe": "rife-ncnn-vulkan"},
}


class ToolError(RuntimeError):
    pass


# ---- binaries ------------------------------------------------------------------------
def binary(tool: str) -> str | None:
    """The configured binary (``OPENCINE_ESRGAN_BIN`` / ``OPENCINE_RIFE_BIN``), else one fetched into
    ``OPENCINE_TOOLS_DIR``; None when neither exists."""
    s = get_settings()
    configured = s.esrgan_bin if tool == "esrgan" else s.rife_bin
    if configured:
        return configured if (Path(configured).is_file() or shutil.which(configured)) else None
    exe = RELEASES[tool]["exe"] + (".exe" if sys.platform == "win32" else "")
    found = sorted(s.tools_dir.glob(f"{tool}/**/{exe}"))
    return str(found[0]) if found else None


def _command(path: str) -> list[str]:
    if path.endswith(".py"):  # a Python stand-in for the binary
        return [sys.executable, path]
    return [shutil.which(path) or path]


def _run_tool(path: str, args: list[str]) -> None:
    cwd = Path(path).parent if Path(path).is_file() else None  # models/ sit next to the binary
    proc = subprocess.run([*_command(path), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=cwd, **NO_WINDOW)
    if proc.returncode != 0:
        raise ToolError(f"{Path(path).name} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")


def fetch(tool: str) -> dict[str, Any]:
    """Download and unpack the pinned release of ``tool`` (Windows zip) into ``OPENCINE_TOOLS_DIR``."""
    import urllib.request

    rel = RELEASES[tool]
    root = get_settings().tools_dir
    dest = root / tool
    zpath = root / Path(rel["url"]).name
    root.mkdir(parents=True, exist_ok=True)
    if not zpath.exists() or zpath.stat().st_size != rel["size"]:
        tmp = zpath.with_suffix(".part")
        with urllib.request.urlopen(rel["url"]) as r, tmp.open("wb") as f:  # noqa: S310 (pinned https)
            shutil.copyfileobj(r, f)
        tmp.replace(zpath)
    if zpath.stat().st_size != rel["size"]:
        raise ToolError(f"{zpath.name}: {zpath.stat().st_size} bytes, expected {rel['size']}")
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    pin = root / f"{tool}.sha256"
    if pin.exists() and pin.read_text().strip() != sha:
        raise ToolError(f"{zpath.name}: sha256 {sha} differs from the one recorded in {pin}")
    pin.write_text(sha)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return {"tool": tool, "binary": binary(tool), "sha256": sha}


# ---- frames in and out -------------------------------------------------------------------
def _extract(m: Media, a: Fraction, n: int, out_dir: Path, rate: Fraction | None) -> None:
    """``n`` frames of ``m.path`` from file time ``a`` as RGB PNGs. ``rate``: on its frame grid
    exactly as a segment decodes them (seek-relative timestamps, the same tie shift); None:
    native frames."""
    if rate is not None:
        chain = [m.prefilter, "settb=AVTB,setpts=PTS-0.001/TB", f"fps={rate}:round=up"]
    else:
        chain = [m.native_filter if m.native_filter is not None else m.prefilter]
    vf = ",".join(x for x in (*chain, "format=rgb24") if x)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["-y", "-loglevel", "error", *(["-ss", f"{float(a):.6f}"] if a > 0 else []), "-i", str(m.path),
         "-vf", vf, "-frames:v", str(n), "-fps_mode", "passthrough", str(out_dir / "%08d.png")])


def _assemble(frames: Path, fps: Fraction, out: Path, size: tuple[int, int] | None = None) -> None:
    vf = (f"scale={size[0]}:{size[1]}:flags=lanczos," if size else "") + TO_YUV
    tmp = out.with_suffix(".tmp.mp4")
    run(["-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%08d.png"),
         "-vf", vf, *INTERMEDIATE, str(tmp)])
    tmp.replace(out)


def _identity(m: Media) -> list:
    st = m.path.stat()
    return [str(m.path.resolve()), st.st_size, st.st_mtime_ns]


def _even(v: float) -> int:
    return max(2, math.ceil(v / 2) * 2)


# ---- the two jobs ---------------------------------------------------------------------------
def upscale(project: Project, m: Media, in_: int, out_: int, rate: Fraction, factor: float,
            exe: str) -> Media:
    """``m`` (decoded view) → an upscaled intermediate of source frames ``[in − LEAD, out + 2)``
    at ``rate`` (a still: the image), ``factor``× its size."""
    s = get_settings()
    size = (_even(m.width * factor), _even(m.height * factor))
    if m.kind == "image":
        key = cache_key("esrgan", _identity(m), size, s.esrgan_model, VERSION)
        dst = project.cache_dir / "ncnn" / f"{key}.png"
        if not dst.exists():
            with tempfile.TemporaryDirectory() as tmp:
                src, up = Path(tmp) / "in", Path(tmp) / "up"
                src.mkdir()
                up.mkdir()
                shutil.copyfile(m.path, src / ("00000001" + m.path.suffix.lower()))
                _run_tool(exe, ["-i", str(src), "-o", str(up), "-n", s.esrgan_model,
                                "-s", str(ESRGAN_SCALE), "-f", "png"])
                from PIL import Image

                got = next(up.glob("*.png"))
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(got) as im:
                    im.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(dst)
        return replace(m, path=dst, width=size[0], height=size[1])
    lead = min(LEAD, math.floor((m.start_s * rate) + in_))
    first = in_ - lead
    n = out_ - first + 2
    key = cache_key("esrgan", _identity(m), m.prefilter, str(m.start_s), first, n, str(rate), size,
                    s.esrgan_model, VERSION)
    dst = project.cache_dir / "ncnn" / f"{key}.mp4"
    if not dst.exists():
        with tempfile.TemporaryDirectory() as tmp:
            src, up = Path(tmp) / "in", Path(tmp) / "up"
            _extract(m, m.start_s + Fraction(first) / rate, n, src, rate)
            up.mkdir()
            _run_tool(exe, ["-i", str(src), "-o", str(up), "-n", s.esrgan_model,
                            "-s", str(ESRGAN_SCALE), "-f", "png"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            _assemble(up, Fraction(rate), dst, size)
    # frame `first` of the source is the file's frame 0: source frame 0 sits before the file
    return replace(m, path=dst, start_s=-Fraction(first) / rate, width=size[0], height=size[1],
                   prefilter="", native_filter="", fps=Fraction(rate), tonemap="")


def interpolate(project: Project, m: Media, in_: int, out_: int, rate: Fraction, slowest: Fraction,
                exe: str) -> Media:
    """``m`` → native frames of ``[in − LEAD, out + 2)`` (plan-rate frames), ×``k`` by RIFE so
    the slowest speed has a frame per output frame."""
    s = get_settings()
    src_fps = m.fps or Fraction(rate)
    k = max(2, math.ceil(Fraction(rate) / (slowest * src_fps) - Fraction(1, 1000)))
    lead = min(LEAD, math.floor((m.start_s * rate) + in_))
    a = m.start_s + Fraction(in_ - lead) / rate
    n = math.ceil(Fraction(out_ - in_ + lead + 2) / rate * src_fps) + 1
    key = cache_key("rife", _identity(m), m.native_filter, str(a), n, k, s.rife_model, VERSION)
    dst = project.cache_dir / "ncnn" / f"{key}.mp4"
    if not dst.exists():
        with tempfile.TemporaryDirectory() as tmp:
            src, mid = Path(tmp) / "in", Path(tmp) / "mid"
            _extract(m, a, n, src, None)
            mid.mkdir()
            got = len(list(src.glob("*.png")))
            _run_tool(exe, ["-i", str(src), "-o", str(mid), "-m", s.rife_model, "-n", str(got * k),
                            "-f", "%08d.png"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            _assemble(mid, src_fps * k, dst)
    return replace(m, path=dst, start_s=m.start_s - a, prefilter="", native_filter="",
                   fps=src_fps * k, tonemap="")


def enhance(project: Project, plan: EditPlan, medias: list[Media], decoded: list[Media],
            size: tuple[int, int]) -> tuple[list[Media], list[str]]:
    """Final renders: the decoded media with upscaled / interpolated intermediates swapped in
    where they help (see the module doc) + notes for the render result."""
    from opencine.plan import ramp as ramps
    from opencine.plan.validate import _magnify
    from opencine.render.segments import _wants_mci, item_size, window_box

    esr, rife = binary("esrgan"), binary("rife")
    out, notes = list(decoded), []
    if not (esr or rife):
        return out, notes
    rate = Fraction(plan.fps)
    jobs = []
    for i, (it, m, d) in enumerate(zip(plan.items, medias, decoded, strict=True)):
        if d.tonemap:
            continue  # HDR: tone-mapped after the crop, keep the native path
        sp = Fraction(str(ramps.speeds(it.ramp)[0] if it.ramp else it.speed)).limit_denominator(100)
        isize = item_size(it, *size)
        if rife and d.kind == "video" and sp < 1 and _wants_mci(d, sp, plan.fps, True,
                                                                window_box(it, d, *isize)):
            jobs.append((math.inf, i, "rife", sp))
            continue
        if esr and (d.kind == "image" or (sp == 1 and not it.ramp)):
            mag = _magnify(it, m, *isize)[0]
            if mag >= UPSCALE_FROM:
                jobs.append((mag, i, "esrgan", mag))
    cap = get_settings().upscale_max_s
    t0 = time.perf_counter()
    for _, i, tool, arg in sorted(jobs, key=lambda j: -j[0]):
        if time.perf_counter() - t0 > cap:
            notes.append(f"#{i + 1}: {tool} skipped (OPENCINE_UPSCALE_MAX_S {cap:g} s reached)")
            continue
        it, d = plan.items[i], decoded[i]
        try:
            if tool == "rife":
                out[i] = interpolate(project, d, it.in_, it.out, rate, arg, rife)
                notes.append(f"#{i + 1}: slow motion interpolated by RIFE")
            else:
                # enough pixels that the window meets the output 1:1 (arg is per original px)
                factor = min(float(ESRGAN_SCALE), arg * (medias[i].full_width or d.width) / d.width)
                out[i] = upscale(project, d, it.in_, it.out, rate, factor, esr)
                notes.append(f"#{i + 1}: upscaled ×{factor:.1f} by Real-ESRGAN")
        except (ToolError, OSError, RuntimeError) as e:
            notes.append(f"#{i + 1}: {tool} failed, CPU fallback ({e})")
    return out, notes

