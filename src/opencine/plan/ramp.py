"""Speed ramps (``EditItem.ramp``): the time map from source to output frames.

Keys give the speed at source frames from the item's ``in`` (linear between, held outside). Source
offset ``s`` reaches output frame ``T(s) = ∫₀ˢ du / v(u)``; the item lasts ``round(T(out − in))``
frames, and ``expr`` hands the same ``T`` to FFmpeg so plan and render agree to the frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

EPS = 1e-9
TB_STEPS = 1000  # preview renders stamp ramped frames in 1/1000-frame steps (setpts truncates)


class Key(Protocol):
    at: int
    speed: float


Piece = tuple[float, float, float, float, float]  # s_a, s_b, v_a, v_b, T(s_a)


def _pieces(keys: Sequence[Key]) -> list[Piece]:
    """Linear speed pieces between consecutive keys, with ``T`` at each start (``T(0) = 0``)."""
    pts = [(float(k.at), float(k.speed)) for k in keys]
    out: list[Piece] = []
    # T at the first key: the speed is held before it
    t = pts[0][0] / pts[0][1]
    for (sa, va), (sb, vb) in zip(pts, pts[1:], strict=False):
        out.append((sa, sb, va, vb, t))
        t += _integral(sa, sb, va, vb, sb)
    out.append((pts[-1][0], math.inf, pts[-1][1], pts[-1][1], t))
    return out


def _integral(sa: float, sb: float, va: float, vb: float, s: float) -> float:
    """∫ from sa to s of du / v(u), v linear from (sa, va) to (sb, vb)."""
    if abs(vb - va) < EPS or sb - sa < EPS:
        return (s - sa) / va
    g = (vb - va) / (sb - sa)
    return math.log((va + g * (s - sa)) / va) / g


def time_at(keys: Sequence[Key], s: float) -> float:
    """``T(s)`` in output frames; before the first key (and below 0) the first speed holds."""
    first = keys[0]
    if s <= first.at:
        return s / first.speed
    for sa, sb, va, vb, t in _pieces(keys):
        if s < sb:
            return t + _integral(sa, sb, va, vb, s)
    raise AssertionError("unreachable")  # the last piece reaches infinity


def length(keys: Sequence[Key], n_src: int) -> int:
    """Output frames of a ramped item spanning ``n_src`` source frames (at least 1)."""
    return max(1, math.floor(time_at(keys, n_src) + 0.5))


def shown(keys: Sequence[Key], n_src: int, n_out: int | None = None) -> list[int]:
    """Source offset shown at each output frame: the last ``N < n_src`` stamped at or before
    ``j`` (``floor(T(N) · TB_STEPS) ≤ j · TB_STEPS``, as the preview render stamps it)."""
    n_out = length(keys, n_src) if n_out is None else n_out
    stamps = [math.floor(time_at(keys, f) * TB_STEPS + EPS) for f in range(n_src)]
    out, f = [], 0
    for j in range(n_out):
        while f + 1 < n_src and stamps[f + 1] <= j * TB_STEPS:
            f += 1
        out.append(f)
    return out


def source_for(keys: Sequence[Key], n_out: int, *, start: float = 0.0, limit: int | None = None) -> int:
    """Fewest source frames from ``start`` (an offset the keys count from) whose output reaches
    ``n_out`` frames, capped at ``limit``."""
    t0 = time_at(keys, start)
    lo, hi = 1, max(1, limit if limit is not None else 64 * n_out + 64)
    if time_at(keys, start + hi) - t0 < n_out - 0.5:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if time_at(keys, start + mid) - t0 >= n_out - 0.5:
            hi = mid
        else:
            lo = mid + 1
    return lo


def speeds(keys: Sequence[Key]) -> tuple[float, float]:
    """Slowest and fastest speed of the ramp."""
    v = [k.speed for k in keys]
    return min(v), max(v)


def _n(x: float) -> str:
    return f"{x:.9f}".rstrip("0").rstrip(".") or "0"


def expr(keys: Sequence[Key], var: str) -> str:
    """``T(var)`` as an FFmpeg expression (nested ``if`` over the pieces; ``log`` is natural)."""
    first = keys[0]
    v = f"({var})"
    tail = None
    for sa, sb, va, vb, t in reversed(_pieces(keys)):
        if math.isinf(sb) or abs(vb - va) < EPS or sb - sa < EPS:
            piece = f"{_n(t)}+({v}-{_n(sa)})/{_n(va)}"
        else:
            g = (vb - va) / (sb - sa)
            piece = f"{_n(t)}+log(({_n(va)}+{_n(g)}*({v}-{_n(sa)}))/{_n(va)})/{_n(g)}"
        tail = piece if tail is None else f"if(lt({v},{_n(sb)}),{piece},{tail})"
    return f"if(lt({v},{_n(first.at)}),{v}/{_n(first.speed)},{tail})"


def label(keys: Sequence[Key]) -> str:
    """``ramp 1→0.35→1`` for review tables."""
    out: list[str] = []
    for k in keys:
        s = f"{k.speed:g}"
        if not out or out[-1] != s:
            out.append(s)
    return "ramp " + "→".join(out)
