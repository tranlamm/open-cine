"""Shot boundaries on the proxy (PySceneDetect ``AdaptiveDetector``) → scene spans.

Spans are half-open ``[start, end)`` proxy frames at the project rate. Shots shorter than
``MIN_SCENE_S`` are merged into a neighbour; scenes longer than ``MAX_SCENE_S`` are split at
their lowest-motion sample (at least ``SPLIT_MARGIN_S`` from either end).
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import AdaptiveDetector, detect

from opencine.core.time import Rate, frames_for

MIN_SCENE_S = 0.5
MAX_SCENE_S = 8.0
SPLIT_MARGIN_S = 2.0
ADAPTIVE_THRESHOLD = 3.0
PARAMS = {"min": MIN_SCENE_S, "max": MAX_SCENE_S, "margin": SPLIT_MARGIN_S,
          "adaptive": ADAPTIVE_THRESHOLD}  # cache key part

Span = tuple[int, int]


def detect_cuts(proxy: str | Path, rate: Rate) -> list[int]:
    """Proxy frames where a new shot starts (frame 0 excluded)."""
    detector = AdaptiveDetector(adaptive_threshold=ADAPTIVE_THRESHOLD,
                                min_scene_len=frames_for(MIN_SCENE_S, rate))
    return [start.frame_num for start, _ in detect(str(proxy), detector)[1:]]


def merge_short(cuts: list[int], n_frames: int, rate: Rate) -> list[Span]:
    """Shots between ``cuts``, with any shot < ``MIN_SCENE_S`` folded into its neighbour."""
    min_len = frames_for(MIN_SCENE_S, rate)
    edges = [0, *(c for c in sorted(set(cuts)) if 0 < c < n_frames), n_frames]
    spans: list[Span] = []
    for a, b in zip(edges, edges[1:], strict=False):
        if spans and (b - a < min_len or spans[-1][1] - spans[-1][0] < min_len):
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def split_long(span: Span, motion: list[tuple[int, float]], rate: Rate) -> list[Span]:
    """Recursively split ``span`` at the lowest-motion ``(frame, motion)`` sample."""
    a, b = span
    if b - a <= frames_for(MAX_SCENE_S, rate):
        return [span]
    margin = frames_for(SPLIT_MARGIN_S, rate)
    mid = (a + b) / 2
    candidates = [(m + 1e-9 * abs(f - mid), f) for f, m in motion if a + margin <= f <= b - margin]
    cut = min(candidates)[1] if candidates else round(mid)
    return split_long((a, cut), motion, rate) + split_long((cut, b), motion, rate)
