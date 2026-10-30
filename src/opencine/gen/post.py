"""CPU post-process of a generated clip so it cuts against phone footage.

One ffmpeg pass from the Space's file (kept as ``*.raw.mp4``): drop ``trim_head`` frames, decode
with the clip's own matrix, optional Reinhard Lab colour match (one 33³ LUT for the whole clip,
so it cannot flicker), re-encode tagged BT.709, retime to the project rate.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from opencine.core.config import get_settings
from opencine.core.ffmpeg import FFmpegError, probe_json
from opencine.core.proc import NO_WINDOW
from opencine.core.time import Rate, to_frame
from opencine.gen.qc import (
    delta_e,
    fps_of,
    lab_stats,
    load_rgb,
    matrix_of,
    read_frames,
    video_stream,
)

Retime = Literal["conform", "interp", "none"]
POST_VERSION = 1
CONFORM_RANGE = (0.8, 1.3)  # project rate / clip rate where a speed change is acceptable
CRF = 12
LUT_SIZE = 33
MATCH_FRAMES = 8
MATCH_DE = (1.0, 20.0)  # below: nothing to fix; above: content changed, not a colour shift
MAX_GAIN = 1.25  # std ratio clamp per Lab channel


def reinhard_cube(src: tuple[np.ndarray, np.ndarray], dst: tuple[np.ndarray, np.ndarray],
                  path: Path, size: int = LUT_SIZE, max_gain: float = MAX_GAIN, *,
                  strength: float = 1.0, l_gain: float | None = None,
                  knee: tuple[float, float] | None = None) -> Path:
    """A ``.cube`` 3D LUT moving Lab mean/std ``src`` onto ``dst`` (per channel, std ratio
    clamped to ``[1/max_gain, max_gain]``; L's to ``[1/l_gain, l_gain]`` when given). Red varies
    fastest, as the format requires. ``strength`` blends the move in Lab; ``knee`` = (L, L)
    over which it fades out (smoothstep) so highlights keep their value instead of clipping."""
    (mu_s, sd_s), (mu_t, sd_t) = src, dst
    gain = np.clip(np.asarray(sd_t) / np.maximum(np.asarray(sd_s), 1e-3), 1 / max_gain, max_gain)
    if l_gain is not None:
        gain[0] = np.clip(gain[0], 1 / l_gain, l_gain)
    lin = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3)
    lab_in = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    lab = (lab_in - mu_s) * gain + mu_t
    weight = np.full(lab_in.shape[:-1] + (1,), strength, np.float32)
    if knee is not None:
        x = np.clip((lab_in[..., :1] - knee[0]) / (knee[1] - knee[0]), 0.0, 1.0)
        weight *= 1.0 - x * x * (3.0 - 2.0 * x)
    lab = (lab_in + weight * (lab - lab_in)).astype(np.float32)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0).reshape(-1, 3)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in out]
    path.write_text("\n".join(lines) + "\n", "ascii")
    return path


def _colour_match(raw: Path, targets: list[Path], stream: dict[str, Any], lut: Path,
                  trim_head: int) -> dict[str, Any]:
    """Measure the first frames (after the trim) against the first target close enough to be
    the same picture (ΔE ≤ ``MATCH_DE[1]``); write ``lut`` if the shift is worth fixing.
    → ``{"de_before", "applied", "gain", "target"}``."""
    frames = read_frames(raw, width=192, max_frames=trim_head + MATCH_FRAMES, stream=stream)
    frames = frames[trim_head:]
    if not len(frames):
        return {"applied": False, "reason": "no frames"}
    h, w = frames.shape[1:3]
    src = lab_stats(frames)
    info: dict[str, Any] = {"applied": False}
    for k, target in enumerate(targets):
        dst = lab_stats(load_rgb(target, (w, h)))
        de = delta_e(src[0], dst[0])
        info = {"de_before": round(de, 2), "applied": False,
                "target": "ref" if k < len(targets) - 1 else "still"}
        if de <= MATCH_DE[0]:
            return {**info, "reason": "within tolerance"}
        if de <= MATCH_DE[1]:
            reinhard_cube(src, dst, lut)
            gain = np.clip(dst[1] / np.maximum(src[1], 1e-3), 1 / MAX_GAIN, MAX_GAIN)
            return {**info, "applied": True, "gain": [round(float(x), 3) for x in gain]}
    return {**info, "reason": "too far (content changed?)"}


WINDOW_WIDTH = 256  # frames are measured at this width
WINDOW_KNEE = (75.0, 95.0)  # L where the window match fades out: whites stay white
WINDOW_L_GAIN = 1.1  # contrast barely moves; a/b keep MAX_GAIN
WINDOW_MIN_BG = 0.05  # background share below which the whole frame is measured


def match_window(clip: Path, window: np.ndarray, masks: list[np.ndarray] | None,
                 strength: float, work: Path, every: int = 1) -> dict[str, Any]:
    """Pull a video-to-video clip's colours onto its footage window (fixes a swap model's cast).

    Matches Lab mean/std of the background (outside the grown person ``masks``; whole frame
    without them) over every ``every``-th frame with one Reinhard LUT, ``strength`` blended, L
    gain held near 1 and faded over ``WINDOW_KNEE`` so highlights keep. ``window``: footage
    frames (RGB uint8), one per clip frame; ``masks`` belong to ``window[::every]``. ``clip`` is
    rewritten in place (CRF 10, BT.709); the Space's file stays as ``*.space.mp4``.
    → what was measured and done."""
    from opencine.gen.control import _grow

    v = video_stream(clip)
    got = read_frames(clip, width=WINDOW_WIDTH, max_frames=len(window), stream=v)[::every]
    ref = window[::every][: len(got)]
    got = got[: len(ref)]
    if not len(got):
        return {"applied": False, "reason": "no frames"}
    h, w = got.shape[1:3]
    ref = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in ref])
    keep = np.ones(got.shape[:3], bool)
    if masks:
        grow = max(1, round(0.03 * h))
        person = np.stack([_grow(cv2.resize(m.astype(np.uint8), (w, h),
                                            interpolation=cv2.INTER_NEAREST) > 0, grow)
                           for m in masks[: len(got)]])
        if (~person).mean() >= WINDOW_MIN_BG:
            keep[: len(person)] = ~person
    src, dst = lab_stats(got[keep]), lab_stats(ref[keep])
    info: dict[str, Any] = {"applied": False, "de_before": round(delta_e(src[0], dst[0]), 2),
                            "background": round(float(keep.mean()), 3), "masked": bool(masks),
                            "strength": strength}
    if info["de_before"] <= MATCH_DE[0]:
        return {**info, "reason": "within tolerance"}
    lut = reinhard_cube(src, dst, work / "window.cube", strength=strength, l_gain=WINDOW_L_GAIN,
                        knee=WINDOW_KNEE)
    space = clip.with_name(f"{clip.stem}.space{clip.suffix}")
    tmp = clip.with_name(f"{clip.stem}.tmp{clip.suffix}")
    matrix, rng = matrix_of(v)
    chain = [f"scale=in_color_matrix={matrix}:in_range={rng}"
             ":flags=bicubic+accurate_rnd+full_chroma_int", "format=gbrp", f"lut3d=file={lut.name}",
             "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd",
             "format=yuv420p",
             "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"]
    _run(["-y", "-i", str(clip.resolve()), "-vf", ",".join(chain), "-c:v", "libx264", "-preset",
          "medium", "-crf", "10", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc",
          "bt709", "-colorspace", "bt709", "-color_range", "tv", "-c:a", "copy",
          str(tmp.resolve())], cwd=lut.parent)
    os.replace(clip, space)
    os.replace(tmp, clip)
    after = lab_stats(read_frames(clip, width=WINDOW_WIDTH, max_frames=len(window))[::every]
                      [: len(ref)][keep])
    gain = np.clip(dst[1] / np.maximum(src[1], 1e-3), 1 / MAX_GAIN, MAX_GAIN)
    gain[0] = np.clip(gain[0], 1 / WINDOW_L_GAIN, WINDOW_L_GAIN)
    return {**info, "applied": True, "de_after": round(delta_e(after[0], dst[0]), 2),
            "gain": [round(float(x), 3) for x in gain], "space": space.name}


LETTERBOX_LEVEL = 20  # a bar row's brightest pixel stays below this in every frame
LETTERBOX_MIN = 4  # rows: a thinner dark edge is picture, not a bar


def letterbox_rows(frames: np.ndarray, level: int = LETTERBOX_LEVEL) -> tuple[int, int]:
    """``(top, bottom)`` rows of the black bars of footage ``frames`` (T, H, W, 3): the leading and
    trailing rows whose brightest pixel stays below ``level`` in every frame; 0 for a side without
    one (or thinner than ``LETTERBOX_MIN``)."""
    if not len(frames):
        return 0, 0
    dark = frames.max(axis=(0, 2, 3)) < level
    if dark.all():
        return 0, 0
    top, bottom = int(np.argmax(~dark)), int(np.argmax(~dark[::-1]))
    return (top if top >= LETTERBOX_MIN else 0), (bottom if bottom >= LETTERBOX_MIN else 0)


def restore_letterbox(clip: Path, top: int, bottom: int, height: int) -> dict[str, int]:
    """Paint the footage's black bars back over ``clip`` (a swap model sometimes draws picture into
    them: building, sky, pavement): ``top`` / ``bottom`` rows of a frame ``height`` tall, scaled
    to the clip's. In place (CRF 10, BT.709, sound copied) → the rows painted at its size."""
    h = int(video_stream(clip)["height"])
    t, b = round(top * h / height), round(bottom * h / height)
    boxes = ([f"drawbox=x=0:y=0:w=iw:h={t}:color=black:t=fill"] if t else []) + (
        [f"drawbox=x=0:y=ih-{b}:w=iw:h={b}:color=black:t=fill"] if b else [])
    if boxes:
        tmp = clip.with_name(f"{clip.stem}.bars{clip.suffix}")
        _run(["-y", "-i", str(clip.resolve()), "-vf", ",".join(boxes), "-c:v", "libx264", "-preset",
              "medium", "-crf", "10", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc",
              "bt709", "-colorspace", "bt709", "-color_range", "tv", "-c:a", "copy",
              str(tmp.resolve())], cwd=clip.parent)
        os.replace(tmp, clip)
    return {"top": t, "bottom": b}


def _run(args: list[str], cwd: Path) -> None:
    cmd = [get_settings().ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")


def frames_at(duration: float | Fraction | str, rate: Fraction) -> int:
    """Whole frames at ``rate`` in ``duration`` (floor, like the render)."""
    return to_frame(str(duration), Rate(rate), "floor")


def exact_duration(stream: dict[str, Any], fallback: str | None = None) -> Fraction:
    """A stream's duration from its integer ticks (``duration_ts`` × ``time_base``):
    ffprobe's ``duration`` is rounded to 6 decimals, so 22 frames at 30 fps would read
    0.733333 s and floor to 21 frames."""
    ts, tb = stream.get("duration_ts"), stream.get("time_base")
    if ts and tb:
        return Fraction(int(ts)) * Fraction(tb)
    return Fraction(str(stream.get("duration") or fallback or 0))


def frame_count(stream: dict[str, Any]) -> int:
    """Frames in a stream: its ``nb_frames``, else duration × rate."""
    if str(stream.get("nb_frames") or "").isdigit() and int(stream["nb_frames"]) > 0:
        return int(stream["nb_frames"])
    return max(1, round(float(stream.get("duration") or 0) * fps_of(stream)))


def post_clip(raw: Path, out: Path, *, rate: Fraction, retime: Retime = "conform",
              still: Path | None = None, trim_head: int = 0, crf: int = CRF,
              ref: Path | None = None, span: Fraction | None = None) -> dict[str, Any]:
    """``raw`` → ``out``; returns what was done and the final ``frames`` at ``rate``.

    ``retime``: ``conform`` plays every frame once at the project rate (24 → 30 fps = 1.25×
    faster, audio sped up; only within ``CONFORM_RANGE``), ``interp`` = motion-compensated
    ``minterpolate`` (keeps speed; slow, warps fast occlusions), ``none`` leaves it to the
    render. ``ref``: the footage frame ``still`` was made from (tried first as colour target).
    ``span``: seconds of footage the frames cover (video-to-video): frames are spread over it
    and the output cut to ``span`` × rate frames; conform does not apply."""
    rate = Fraction(rate)
    v = video_stream(raw)
    src_fps = fps_of(v)
    if span is not None:
        span = Fraction(span).limit_denominator(100000)
        # the rate the frames really have over the footage they cover
        src_fps = (Fraction(frame_count(v) - trim_head) / span).limit_denominator(1001)
        retime = "none" if retime == "conform" else retime
    matrix, rng = matrix_of(v)
    ratio = rate / src_fps
    applied: Retime = retime
    if retime == "conform" and not CONFORM_RANGE[0] <= ratio <= CONFORM_RANGE[1]:
        applied = "none"
    if retime != "none" and ratio == 1:
        applied = "none"
    info: dict[str, Any] = {"version": POST_VERSION, "src_fps": str(src_fps), "matrix_in": matrix,
                            "retime": applied, "trim_head": trim_head}
    if applied != retime:
        info["retime_asked"] = retime
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        lut = work / "match.cube"
        if still is not None:
            targets = ([ref] if ref is not None else []) + [still]
            info["colour"] = _colour_match(raw, targets, v, lut, trim_head)
        chain = [f"trim=start_frame={trim_head}", "setpts=PTS-STARTPTS"] if trim_head else []
        if span is not None:  # frame N at N / src_fps: CFR at the rate it really has
            chain += [f"settb={src_fps.denominator}/{src_fps.numerator}", "setpts=N",
                      f"fps={src_fps.numerator}/{src_fps.denominator}"]
            if applied == "interp":  # minterpolate stops at the last input frame: pad, cut below
                chain.append("tpad=stop_mode=clone:stop=2")
        chain += [f"scale=in_color_matrix={matrix}:in_range={rng}"
                  ":flags=bicubic+accurate_rnd+full_chroma_int", "format=gbrp"]
        if lut.exists():
            chain.append(f"lut3d=file={lut.name}")
        chain += ["scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd",
                  "format=yuv420p"]
        audio = ["-an"]
        afilters = ([f"atrim=start={float(trim_head / src_fps):.6f}", "asetpts=PTS-STARTPTS"]
                    if trim_head else [])
        if applied == "conform":
            # time base = one frame at the project rate, so every timestamp is exact
            chain += [f"settb={rate.denominator}/{rate.numerator}", "setpts=N"]
            afilters.append(f"atempo={float(ratio):.6f}")
        elif applied == "interp":
            chain.append(f"minterpolate=fps={rate.numerator}/{rate.denominator}:mi_mode=mci"
                         ":mc_mode=aobmc:me_mode=bidir:vsbmc=1")
        chain.append("setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")
        if v["has_audio"]:
            audio = (["-af", ",".join(afilters)] if afilters else []) + ["-c:a", "aac", "-b:a",
                                                                         "192k"]
        rate_args = ["-r", f"{rate.numerator}/{rate.denominator}"] if applied != "none" else []
        if span is not None and applied != "none":
            rate_args += ["-frames:v", str(frames_at(span, rate))]
        tmp_out = out.with_name(f"{out.stem}.tmp{out.suffix}")
        _run(["-y", "-i", str(raw.resolve()), "-vf", ",".join(chain), *rate_args,
              "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0",
              "-pix_fmt", "yuv420p",
              "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
              "-color_range", "tv", *audio, "-movflags", "+faststart", str(tmp_out.resolve())],
             cwd=work)
        os.replace(tmp_out, out)
    o = probe_json(out)
    ov = next(s for s in o["streams"] if s.get("codec_type") == "video")
    duration = exact_duration(ov, o["format"].get("duration"))
    info.update({"fps": str(fps_of(ov)), "width": ov.get("width"), "height": ov.get("height"),
                 "duration_s": round(float(duration), 3), "frames": frames_at(duration, rate),
                 "rate": str(Rate(rate))})
    return info
