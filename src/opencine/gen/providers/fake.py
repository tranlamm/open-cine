"""Offline dry-run provider: no GPU, no network.

Writes a small real PNG / H.264 MP4 that depends only on inputs, prompt and seed.
``quota_calls`` simulates quota exhaustion after N calls.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from opencine.core import ffmpeg
from opencine.gen.base import (
    GenProvider,
    GenResult,
    QuotaExhausted,
    crop_to_aspect,
    parse_aspect,
)
from opencine.gen.config import get_gen_settings

SHORT_SIDE = 288


def _size(aspect: str) -> tuple[int, int]:
    a = parse_aspect(aspect)
    if a < 1:
        return SHORT_SIDE, int(SHORT_SIDE / a) // 2 * 2
    return int(SHORT_SIDE * a) // 2 * 2, SHORT_SIDE


def _digest(*parts: object) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


class FakeProvider(GenProvider):
    name = "fake"
    max_parallel = 2

    def __init__(self, quota_calls: int | None = -1, reset_in_s: float = 3600.0):
        self.quota_calls = get_gen_settings().fake_quota_calls if quota_calls == -1 else quota_calls
        self.reset_in_s = reset_in_s
        self.calls = 0
        self._lock = threading.Lock()

    def _spend(self) -> None:
        with self._lock:
            if self.quota_calls is not None and self.calls >= self.quota_calls:
                reset = datetime.now() + timedelta(seconds=self.reset_in_s)
                raise QuotaExhausted(f"fake quota of {self.quota_calls} calls used up",
                                     reset_at=reset)
            self.calls += 1

    def _result(self, out: Path, seed: int | None, t0: float, model: str) -> GenResult:
        return GenResult(path=out, provider=self.name, model=model, seed=seed, quota_used=1,
                         quota_unit="call", latency_s=round(time.monotonic() - t0, 3))

    def edit_image(self, refs: list[Path], prompt: str, aspect: str = "9:16", *, out: Path,
                   seed: int | None = None, model: str | None = None) -> GenResult:
        t0 = time.monotonic()
        self._spend()
        seed = seed if seed is not None else int.from_bytes(_digest(prompt)[:2], "big")
        tint = tuple(_digest(prompt, seed)[:3])
        if not refs:  # text-to-image: the prompt's colour
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", _size(aspect), tint).save(out)
            return self._result(out, seed, t0, "fake-t2i")
        crop_to_aspect(refs[0], aspect, out)
        with Image.open(out) as im:
            base = im.convert("RGB").resize(_size(aspect))
        Image.blend(base, Image.new("RGB", base.size, tint), 0.35).save(out)
        return self._result(out, seed, t0, "fake-edit")

    def video(self, first: Path | None, prompt: str, seconds: float = 5.0, aspect: str = "9:16",
              last: Path | None = None, *, out: Path, seed: int | None = None,
              canvas: str | None = None, model: str | None = None,
              refs: list[Path] | None = None) -> GenResult:
        """No first frame: a tinted test pattern; first (else first ref): that still held;
        first + last: each still for half the clip."""
        t0 = time.monotonic()
        self._spend()
        seed = seed if seed is not None else int.from_bytes(_digest(prompt)[:2], "big")
        w, h = _size(aspect)
        fit = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
        out.parent.mkdir(parents=True, exist_ok=True)
        model_name = "fake-ref" if refs else None
        first = first if first is not None or not refs else refs[0]
        if first is None:  # text-to-video: a moving test pattern tinted by the prompt
            hue = _digest(prompt, seed)[0] * 360 // 256
            ffmpeg.run(["-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=16:d={seconds}",
                        "-vf", f"hue=h={hue},format=yuv420p", "-c:v", "libx264", "-preset",
                        "ultrafast", str(out)])
            return self._result(out, seed, t0, "fake-t2v")
        inputs = ["-loop", "1", "-framerate", "16", "-t", f"{seconds}", "-i", str(first)]
        if last is not None:
            half = seconds / 2
            inputs = ["-loop", "1", "-framerate", "16", "-t", f"{half}", "-i", str(first),
                      "-loop", "1", "-framerate", "16", "-t", f"{half}", "-i", str(last)]
            graph = f"[0:v]{fit}[a];[1:v]{fit}[b];[a][b]concat=n=2:v=1[v];[v]format=yuv420p[o]"
        else:
            graph = f"[0:v]{fit},format=yuv420p[o]"
        ffmpeg.run(["-y", *inputs, "-filter_complex", graph, "-map", "[o]", "-r", "16",
                    "-c:v", "libx264", "-preset", "ultrafast", str(out)])
        return self._result(out, seed, t0, model_name or ("fake-flf" if last else "fake-i2v"))

    def edit_video(self, source: Path, refs: list[Path], prompt: str, mode: str,
                   aspect: str = "9:16", *, out: Path, seed: int | None = None,
                   model: str | None = None, subject: str = "person",
                   control: str = "depth") -> GenResult:
        """The source hue-shifted by the seed (restyle also desaturated), same length, at 16 fps
        like a real model below the project rate."""
        t0 = time.monotonic()
        self._spend()
        seed = seed if seed is not None else int.from_bytes(_digest(prompt)[:2], "big")
        w, h = _size(aspect)
        look = f"hue=h={seed % 360}" + (":s=0.4" if mode == "restyle" else "")
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(["-y", "-i", str(source), "-vf",
                    f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,"
                    f"fps=16,{look},format=yuv420p", "-an", "-c:v", "libx264", "-preset",
                    "ultrafast", str(out)])
        return self._result(out, seed, t0, f"fake-{mode}")

    def enhance(self, src: Path, *, out: Path, short: int | None = None) -> GenResult:
        """Lanczos upscale to a short side of ``short`` (default 1080), audio copied."""
        t0 = time.monotonic()
        self._spend()
        short = short or 1080
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(["-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-vf",
                    f"scale='if(gt(iw,ih),-2,{short})':'if(gt(iw,ih),{short},-2)':flags=lanczos,"
                    "format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy",
                    str(out)])
        return self._result(out, None, t0, "fake-enhance")

    def face(self, crop: Path, refs: list[Path], prompt: str, strengths: list[float], *, out: Path,
             audio: Path | None = None, seed: int | None = None) -> GenResult:
        """The crop back, slightly sharpened, silent, same frames."""
        t0 = time.monotonic()
        self._spend()
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(["-y", "-i", str(crop), "-map", "0:v:0", "-vf", "unsharp=5:5:0.8,format=yuv420p",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)])
        return self._result(out, seed, t0, "fake-face")

    def quota(self) -> dict[str, Any]:
        left = None if self.quota_calls is None else max(self.quota_calls - self.calls, 0)
        return {"provider": self.name, "unit": "call", "used": self.calls, "left": left,
                "reset_at": None}
