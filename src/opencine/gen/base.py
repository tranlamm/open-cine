"""Provider interface for image/video generation, plus results, errors and media helpers.

The Director calls ``opencine.gen.service``; providers implement ``edit_image``/``video``/``quota``
(``edit_video`` when they can) and never pick another provider themselves.
"""

from __future__ import annotations

import math
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps
from pydantic import BaseModel

from opencine.core.ffmpeg import probe_json

# edit: video-to-video; enhance: a sharper, larger copy of a clip (same frames)
Kind = Literal["image", "video", "edit", "enhance", "face"]
# swap: the clip's person becomes a reference character (same motion, camera, background);
# restyle: the whole picture takes the look of a restyled first frame (same layout and motion)
EditMode = Literal["swap", "restyle"]


class GenResult(BaseModel):
    path: Path
    provider: str
    model: str
    seed: int | None = None
    quota_used: float = 0.0  # in ``quota_unit``
    quota_unit: str = "none"  # gpu_s (hf) | call (fake)
    usd_est: float = 0.0
    latency_s: float = 0.0
    extra: dict[str, Any] = {}


# ---- errors ---------------------------------------------------------------
class ProviderError(RuntimeError):
    """A generation call failed. ``retryable`` = the same call may succeed later.

    ``quota_used``/``quota_unit``/``usd_est``: what the failed call still spent (GPU time
    of an aborted task), for the project ledger.
    """

    retryable = False
    quota_used: float = 0.0
    quota_unit: str = "none"
    usd_est: float = 0.0

    def spent(self, quota_used: float, quota_unit: str, usd_est: float = 0.0) -> ProviderError:
        self.quota_used, self.quota_unit, self.usd_est = quota_used, quota_unit, usd_est
        return self


class QuotaExhausted(ProviderError):
    """Quota/credit is used up. ``reset_at`` is when it comes back (None = unknown/top-up)."""

    def __init__(self, message: str, reset_at: datetime | None = None):
        super().__init__(message)
        self.reset_at = reset_at


class TransientError(ProviderError):
    """Queue timeout, no GPU, network: retry the same call."""

    retryable = True


class AuthError(ProviderError):
    """Not logged in / token not recognised. The user must fix credentials."""


class ModerationRejected(ProviderError):
    """Input or prompt refused by moderation. Do not resubmit unchanged."""


# ---- interface ------------------------------------------------------------
class GenProvider(ABC):
    name: str
    max_parallel: int = 1
    ref_model: str | None = None  # the model choice a clip with refs runs when it names none

    @abstractmethod
    def edit_image(self, refs: list[Path], prompt: str, aspect: str = "9:16", *, out: Path,
                   seed: int | None = None, model: str | None = None) -> GenResult:
        """Edit/compose ``refs`` into one still at ``out``; a new camera angle is an edit
        written in the prompt. No ``refs``: text-to-image. ``model``: a provider model choice
        (hf only); None = the default."""

    @abstractmethod
    def video(self, first: Path | None, prompt: str, seconds: float = 5.0, aspect: str = "9:16",
              last: Path | None = None, *, out: Path, seed: int | None = None,
              canvas: str | None = None, model: str | None = None,
              refs: list[Path] | None = None) -> GenResult:
        """Image-to-video from ``first`` (first/last-frame when ``last`` is given; text-to-video
        when ``first`` is None) at ``out``.
        ``canvas``: provider quality tier (hf: fast | mid | full); None = the configured one.
        ``model``: a provider model choice (hf only); None = the configured one.
        ``refs``: reference pictures (a character, a place) the whole clip follows (hf Ref2VA);
        ``first`` is then the first frame too."""

    def edit_video(self, source: Path, refs: list[Path], prompt: str, mode: EditMode,
                   aspect: str = "9:16", *, out: Path, seed: int | None = None,
                   model: str | None = None, subject: str = "person",
                   control: str = "depth") -> GenResult:
        """Video-to-video on ``source`` (a footage window, already cut and framed): ``swap``
        puts the character of ``refs[0]`` in place of the person (``subject``), ``restyle``
        gives the clip the look of ``refs[0]`` (its first frame, restyled), holding the
        ``control`` structure (depth | canny). Same length, motion and camera. ``model``: a
        provider model choice; None = the mode's default. ``extra["covers_s"]``: seconds of the
        window the output's frames span (when less than all of it)."""
        raise ProviderError(f"{self.name}: video-to-video ({mode}) is not supported; use the "
                            "hf provider")

    def enhance(self, src: Path, *, out: Path, short: int | None = None) -> GenResult:
        """A sharper copy of the clip ``src`` at ``out`` with its short side ``short`` (None = the
        configured one): the same frames, timing, colours and sound."""
        raise ProviderError(f"{self.name}: enhance is not supported; use the hf provider")

    def face(self, crop: Path, refs: list[Path], prompt: str, strengths: list[float], *, out: Path,
             audio: Path | None = None, seed: int | None = None) -> GenResult:
        """The face pass on a square crop clip that follows one face (``opencine.gen.face``): redrawn
        at ``out`` (silent, same frames), each frame by its denoise ``strengths``, the identity
        from ``refs``, the mouth held to ``audio`` (32 kHz WAV) when given."""
        raise ProviderError(f"{self.name}: the face pass is not supported; use the hf provider")

    @abstractmethod
    def quota(self) -> dict[str, Any]:
        """``{provider, unit, left, reset_at, …}``."""

    def fingerprint(self, kind: Kind, model: str | None = None,
                    mode: EditMode | None = None) -> dict[str, Any]:
        """Settings that change the output of ``kind`` (model, steps, Space revision…): part
        of the cache key, so changing one regenerates instead of reusing a stale file.
        ``model``: a slot's model choice, when it has one; ``mode``: an edit's mode."""
        return {}

    def warm(self) -> dict[str, Any]:
        """Wake remote backends before a batch (no-op by default)."""
        return {}

    def prepare(self, kind: Kind, slots: list[Any]) -> dict[str, Any]:
        """Called once per batch with the slots that will really run (e.g. HF warm-up)."""
        return {}

    def estimate_usd(self, kind: Kind, seconds: float = 0.0) -> float:
        """Upper-bound USD for one call, for ``Ledger.check_cap``."""
        return 0.0


# ---- media helpers --------------------------------------------------------
def parse_aspect(aspect: str) -> Fraction:
    """``"9:16"`` → ``Fraction(9, 16)`` (width / height)."""
    w, _, h = aspect.partition(":")
    return Fraction(int(w), int(h))


def crop_to_aspect(src: Path, aspect: str, dst: Path) -> Path:
    """Centre-crop ``src`` to ``aspect`` and save as ``dst`` (EXIF rotation applied)."""
    target = parse_aspect(aspect)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if Fraction(w, h) > target:
            nw = round(h * target)
            box = ((w - nw) // 2, 0, (w - nw) // 2 + nw, h)
        else:
            nh = round(w / target)
            box = (0, (h - nh) // 2, w, (h - nh) // 2 + nh)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.crop(box).save(dst)
    return dst


def fit_canvas(src: Path, width: int, height: int, dst: Path,
               box: tuple[float, float, float, float] | None = None) -> Path:
    """``src`` (EXIF rotation applied) at exactly ``width×height``: the optional ``box``
    (fractions x, y, w, h) first, then its centred window of that aspect, Lanczos-resized.

    A model given exactly the pixels it works at neither stretches nor re-crops them.
    """
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if box is not None:
            w, h = im.size
            x0, y0 = min(max(box[0], 0.0), 1.0) * w, min(max(box[1], 0.0), 1.0) * h
            x1, y1 = min(x0 + box[2] * w, w), min(y0 + box[3] * h, h)
            im = im.crop((round(x0), round(y0), max(round(x1), round(x0) + 1),
                          max(round(y1), round(y0) + 1)))
        im = ImageOps.fit(im, (width, height), method=Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)
    return dst


def limit_pixels(src: Path, dst: Path, max_pixels: int = 1024 * 1024) -> Path:
    """``src`` (EXIF rotation applied, RGB) shrunk to at most ``max_pixels``, aspect kept.
    Never upscales: a reference image keeps its whole frame."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            k = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)
    return dst


def fit_video(src: Path, width: int, height: int, fps: float | Fraction | None, dst: Path,
              frames: int | None = None) -> Path:
    """``src`` at exactly ``width×height`` (centred cover, Lanczos), resampled to ``fps`` (None =
    keep) and cut to ``frames`` frames, BT.709 yuv420p, no audio: a video input given at the size
    and rate a model works at, like ``fit_canvas`` for stills."""
    from opencine.core import ffmpeg

    vf = [f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
          f"crop={width}:{height}", "setsar=1"]
    if fps is not None:
        vf.append(f"fps={Fraction(fps).numerator}/{Fraction(fps).denominator}")
    vf.append("format=yuv420p")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-y", "-i", str(src), "-vf", ",".join(vf), *(["-frames:v", str(frames)] if frames else []),
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                str(dst)])
    return dst


def save_output(src: str | Path, out: Path) -> Path:
    """Copy a downloaded file to ``out``; images are converted when the suffix differs."""
    src = Path(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    if out.suffix.lower() in image_exts and src.suffix.lower() != out.suffix.lower():
        with Image.open(src) as im:
            im.convert("RGB").save(out)
    else:
        shutil.copyfile(src, out)
    return out


def verify_video(path: Path, aspect: str | None = None) -> dict[str, Any]:
    """ffprobe check: one video stream, positive duration. Aspect mismatch → ``warning``."""
    info = probe_json(path)
    video = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    duration = float(info.get("format", {}).get("duration") or 0)
    if not video or duration <= 0:
        raise ProviderError(f"{path.name}: not a playable video (streams={len(video)}, {duration}s)")
    v = video[0]
    out = {"width": v.get("width"), "height": v.get("height"), "codec": v.get("codec_name"),
           "duration": round(duration, 3), "fps": v.get("avg_frame_rate")}
    if aspect and v.get("width") and v.get("height"):
        got = Fraction(v["width"], v["height"])
        if abs(float(got / parse_aspect(aspect)) - 1) > 0.03:
            out["warning"] = f"aspect {v['width']}x{v['height']} != {aspect}"
    return out
