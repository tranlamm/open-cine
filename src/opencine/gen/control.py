"""Inputs video-to-video models need besides the footage window, made locally (no quota).

- ``control_video``: what a restyle keeps (depth via Depth Anything V2 Small, near = white, or
  Canny edges) for the LTX Union-Control IC-LoRA.
- ``swap_inputs``: SCAIL-2 inputs: the person's mask per frame (Florence-2 on the first frame,
  EdgeTAM after), grown so a bulkier character fits; the character still and its mask on black
  (SCAIL-2 "Replacement": black = no background, white = background visible, pure colour = slot).

Frames go through ffmpeg as raw RGB; clips are written as H.264 at the given rate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from opencine.core.config import get_settings
from opencine.core.ffmpeg import FFmpegError
from opencine.core.proc import NO_WINDOW
from opencine.gen.base import ProviderError

Control = Literal["depth", "canny"]
CANNY = (100, 200)  # the LTX trainer's thresholds
SLOT_COLOUR = (0, 0, 255)  # SCAIL-2's first binding slot (pure blue; ≥ 225 per channel)
REF_GROW = 0.12  # the character still is searched in the person's box grown by this much


def read_rgb(video: Path, width: int, height: int, fps: float | None = None) -> np.ndarray:
    """Every frame of ``video`` as ``(n, height, width, 3)`` uint8 RGB (cover-scaled), resampled
    to ``fps`` when given."""
    vf = (("" if fps is None else f"fps={fps:g},")
          + f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height},format=rgb24")
    cmd = [get_settings().ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(video),
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-"]
    proc = subprocess.run(cmd, capture_output=True, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
    size = width * height * 3
    return np.frombuffer(proc.stdout[: len(proc.stdout) // size * size], np.uint8).reshape(
        -1, height, width, 3)


def write_rgb(frames: np.ndarray | list[np.ndarray], fps: float, dst: Path, crf: int = 4) -> Path:
    """Frames (RGB uint8, or grey) as an H.264 mp4 at ``fps``."""
    frames = [f if f.ndim == 3 else np.repeat(f[..., None], 3, axis=2) for f in frames]
    h, w = frames[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [get_settings().ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv", str(dst)]
    proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                          capture_output=True, **NO_WINDOW)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
    return dst


def pad_frames(frames: np.ndarray, n: int) -> np.ndarray:
    """``frames`` cut or padded (the last frame repeated) to exactly ``n``."""
    if len(frames) >= n:
        return frames[:n]
    return np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), axis=0)])


# ---- restyle: structure to keep ---------------------------------------------------------------
def canny(frames: np.ndarray) -> list[np.ndarray]:
    """White edges on black, every frame at the same thresholds (no flicker from per-frame ones)."""
    out = []
    for f in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        out.append(cv2.Canny(g, *CANNY))
    return out


def depth(frames: np.ndarray) -> list[np.ndarray]:
    """Grey depth maps (near = white) of a clip; needs the local depth model."""
    from opencine.vision.depth import Depth, grey

    model = Depth.load()
    if model is None:
        raise ProviderError("the depth control needs the local depth model (uv sync --extra "
                            "vision; OPENCINE_VISION_MODELS on); or use control='canny'")
    return grey([model.predict(f) for f in frames])


def control_video(frames: np.ndarray, kind: Control, fps: float, dst: Path) -> Path:
    """The control clip of ``frames`` (already at the size and count the model wants)."""
    maps = depth(frames) if kind == "depth" else canny(frames)
    return write_rgb(maps, fps, dst)


# ---- swap: who to replace ----------------------------------------------------------------------
def _vision():
    from opencine.vision.florence import Florence
    from opencine.vision.tam import Tam

    fl, tam = Florence.load(), Tam.load()
    if fl is None or tam is None:
        raise ProviderError("a character swap needs the local vision models (Florence-2, "
                            "EdgeTAM): uv sync --extra vision, OPENCINE_VISION_MODELS on")
    return fl, tam


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def person_masks(frames: np.ndarray, phrase: str = "person"
                 ) -> tuple[tuple[float, float, float, float], list[np.ndarray]]:
    """``(box on the first frame, mask per frame)`` of the largest ``phrase`` on the first frame,
    followed through every frame (empty where it is not visible)."""
    fl, tam = _vision()
    found = fl.find(fl.see(frames[0]), phrase)
    if not found:
        raise ProviderError(f"no {phrase!r} found on the window's first frame; name the person "
                            "to replace in `subject` (English, e.g. 'runner')")
    box = max(found, key=lambda f: f.area).box
    masks = [m if s > 0 else np.zeros_like(m) for m, s in tam.propagate(list(frames), box=box)]
    return box, masks


def character_mask(still: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The character's mask in a still framed like the window's first frame: the thing inside
    the person's first-frame ``box`` grown by ``REF_GROW`` (armor is wider than a body)."""
    _, tam = _vision()
    x0, y0, x1, y1 = box
    gx, gy = (x1 - x0) * REF_GROW, (y1 - y0) * REF_GROW
    grown = (max(0.0, x0 - gx), max(0.0, y0 - gy), min(1.0, x1 + gx), min(1.0, y1 + gy))
    mask, score = tam.propagate([still], box=grown)[0]
    if score <= 0 or mask.mean() < 0.002:
        raise ProviderError("the character was not found in the reference still where the person "
                            "stands on the first frame: make the still an edit of that frame "
                            "(same pose, position and size)")
    return mask


def grey_cutout(picture: np.ndarray, phrases: tuple[str, ...] = ("person", "character"),
                grey: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray | None:
    """The largest ``phrases`` match of a character picture on ``grey`` (the H3 swap LoRA's
    reference ground); None when nothing is found."""
    fl, tam = _vision()
    seen = fl.see(picture)
    for phrase in phrases:
        if found := fl.find(seen, phrase):
            box = max(found, key=lambda f: f.area).box
            mask, score = tam.propagate([picture], box=box)[0]
            if score > 0 and mask.mean() >= 0.01:
                return np.where(mask[..., None], picture, np.array(grey, np.uint8)).astype(np.uint8)
    return None


def swap_inputs(frames: np.ndarray, still: np.ndarray, fps: float, work: Path, *,
                phrase: str = "person", grow: float = 0.03, replace: bool = True
                ) -> tuple[Path, Path, Path, dict]:
    """SCAIL-2 inputs → ``(reference on black, its mask, mask video, info)``. ``still`` is at
    the frames' size; ``grow`` widens the person's masks by that fraction of the frame height
    so a bulkier character is not clipped to the old silhouette. Replacement: the reference
    mask on black, the clip's masks on white; ``replace=False`` (animation mode keeping the
    background through the masks) swaps those grounds."""
    box, masks = person_masks(frames, phrase)
    ref_mask = character_mask(still, box)
    h = frames.shape[1]
    px = round(grow * h)
    colour = np.array(SLOT_COLOUR, np.uint8)
    ref = np.where(ref_mask[..., None], still, 0).astype(np.uint8)
    ref_ground, clip_ground = (0, 255) if replace else (255, 0)
    ref_m = np.where(ref_mask[..., None], colour, ref_ground).astype(np.uint8)
    video = [np.where(_grow(m, px)[..., None], colour, clip_ground).astype(np.uint8)
             for m in masks]
    work.mkdir(parents=True, exist_ok=True)
    ref_path, mask_path = work / "ref_black.png", work / "ref_mask.png"
    cv2.imwrite(str(ref_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), cv2.cvtColor(ref_m, cv2.COLOR_RGB2BGR))
    seen = sum(bool(m.any()) for m in masks)
    info = {"box": [round(v, 4) for v in box], "frames_with_person": f"{seen}/{len(masks)}",
            "grow_px": px}
    return ref_path, mask_path, write_rgb(video, fps, work / "mask_video.mp4", crf=1), info
