"""Colour as 3D LUTs: an item's Lab shift and the plan's look baked into one 33³ ``.cube``
(a single fixed ``lut3d`` pass, so it cannot flicker). Looks are deterministic curves on display
RGB (0–1), mixed with the identity by ``strength``: clean, warm, cool, film, vivid, mono.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

LUT_SIZE = 33
LOOKS = ("clean", "warm", "cool", "film", "vivid", "mono")
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _s_curve(x: np.ndarray, amount: float) -> np.ndarray:
    return x + amount * (x * x * (3 - 2 * x) - x)


def _saturate(rgb: np.ndarray, s: float) -> np.ndarray:
    lum = (rgb @ LUMA)[..., None]
    return lum + (rgb - lum) * s


def look_rgb(name: str, rgb: np.ndarray) -> np.ndarray:
    """``rgb`` (…, 3) in 0–1 through look ``name`` at full strength."""
    if name == "clean":
        out = _saturate(_s_curve(rgb, 0.15), 1.05)
    elif name in ("warm", "cool"):
        k = 1 if name == "warm" else -1
        out = _s_curve(rgb * np.array([1 + 0.04 * k, 1.0, 1 - 0.05 * k], np.float32), 0.12)
    elif name == "film":
        out = 0.03 + 0.94 * rgb
        out = np.where(out > 0.75, 0.75 + (out - 0.75) * 0.8, out)  # highlight roll-off
        lum = (out @ LUMA)[..., None]
        shadow, high = np.clip(1 - lum * 2, 0, 1), np.clip(lum * 2 - 1, 0, 1)
        out = out + shadow * np.array([-0.02, 0.01, 0.03], np.float32) \
            + high * np.array([0.03, 0.01, -0.03], np.float32)
        out = _saturate(_s_curve(out, 0.1), 0.92)
    elif name == "vivid":
        out = _saturate(_s_curve(rgb, 0.25), 1.18)
    elif name == "mono":
        out = np.repeat(_s_curve(rgb @ LUMA, 0.15)[..., None], 3, axis=-1)
    else:
        raise ValueError(f"unknown look {name!r}; one of {', '.join(LOOKS)}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shift_rgb(rgb: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """``rgb`` (…, 3) moved by ``shift`` in CIELAB (L 0–100)."""
    shape = rgb.shape
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab)
    lab = lab + np.asarray(shift, np.float32)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0).reshape(shape)


def transform(rgb: np.ndarray, shift: np.ndarray | None, look: str | None,
              strength: float = 1.0) -> np.ndarray:
    out = rgb.astype(np.float32)
    if shift is not None and np.any(np.abs(shift) > 1e-6):
        out = shift_rgb(out, shift)
    if look:
        out = out + strength * (look_rgb(look, out) - out)
    return np.clip(out, 0.0, 1.0)


def cube_text(shift: np.ndarray | None, look: str | None, strength: float = 1.0,
              size: int = LUT_SIZE) -> str:
    """The ``.cube`` of ``transform`` (red varies fastest, as the format requires)."""
    lin = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = transform(grid, shift, look, strength)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in out]
    return "\n".join(lines) + "\n"


def write_cube(text: str, out_dir: Path) -> Path:
    """``<out_dir>/<hash>.cube`` (written once)."""
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    path = out_dir / f"{key}.cube"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, "ascii")
        tmp.replace(path)
    return path


def lut_filter(path: Path) -> str:
    """BT.709 limited YUV → RGB → ``lut3d`` → back, the path escaped like the ``ass`` filter's."""
    p = path.resolve().as_posix()
    if "'" in p:
        raise ValueError(f"LUT path must not contain a quote: {p}")
    p = p.replace(":", "\\:")
    return (f"scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int,format=gbrp,"
            f"lut3d=file='{p}':interp=tetrahedral,"
            "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd,format=yuv420p")
