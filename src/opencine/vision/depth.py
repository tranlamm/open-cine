"""Relative depth of a frame: Depth Anything V2 Small (Apache-2.0), for restyle control videos.

Preprocessing follows the DPT processor (short side 518, multiples of 14, ImageNet-normalised);
the output is relative inverse depth (near = large). Per-frame scale drifts, so ``grey``
normalises against the clip's smoothed range; near is white, as the LTX Union IC-LoRA expects.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from opencine.vision import models, runtime

SIDE = 518
MULTIPLE = 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LO, HI = 2.0, 98.0  # percentiles of a frame's depth mapped to black / white
SMOOTH = 0.3  # weight of the new frame in the running range (less flicker between frames)


def _size(w: int, h: int) -> tuple[int, int]:
    k = SIDE / min(w, h)
    return (max(MULTIPLE, round(w * k / MULTIPLE) * MULTIPLE),
            max(MULTIPLE, round(h * k / MULTIPLE) * MULTIPLE))


class Depth:
    def __init__(self, session: Any):
        self.s = session
        self.input = session.get_inputs()[0].name

    @classmethod
    def load(cls) -> Depth | None:
        d = models.fetch("depth")
        if d is None:
            return None
        return cls(runtime.session(d / models.MODELS["depth"]["files"][0]))

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Relative inverse depth (float32, near = large) at the frame's size."""
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, _size(w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        out = np.asarray(self.s.run(None, {self.input: x})[0], np.float32).reshape(x.shape[2:])
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def grey(depths: list[np.ndarray]) -> list[np.ndarray]:
    """``uint8`` maps of a clip's depths: each frame's ``LO``–``HI`` percentile range, smoothed
    over time, stretched to 0–255 (near = white)."""
    out, lo, hi = [], None, None
    for d in depths:
        a, b = np.percentile(d, [LO, HI])
        lo = a if lo is None else (1 - SMOOTH) * lo + SMOOTH * a
        hi = b if hi is None else (1 - SMOOTH) * hi + SMOOTH * b
        out.append(np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))
    return out
