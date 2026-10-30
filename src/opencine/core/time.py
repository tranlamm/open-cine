"""Frame-exact time math.

Times inside the engine are integer frame indices plus a rational ``Rate``.
Frame ``f`` at rate ``r`` covers the half-open interval ``[f/r, (f+1)/r)`` seconds.

Rounding rules (explicit, never float):
- ``to_frame(seconds)``: the frame whose interval contains ``seconds`` (``floor``), the
  nearest frame start (``nearest``, ties go up), or ``ceil``.
- ``convert(f, src, dst)``: the ``dst`` frame containing the *centre* of ``f``. Centre
  mapping makes coarse→fine→coarse the identity and keeps any chain of conversions
  within one coarse frame of the original, so repeated conversions never drift.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal

Rounding = Literal["floor", "nearest", "ceil"]

_COMMON = {
    "23.976": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "59.94": Fraction(60000, 1001),
    "119.88": Fraction(120000, 1001),
}
NTSC_TOLERANCE = 1e-3  # a float rate this close to an NTSC rate means that rate


class Rate(Fraction):
    """Frames per second as an exact fraction (e.g. ``Rate("30000/1001")``)."""

    def __new__(cls, value: int | str | Fraction | float = 30, denominator: int | None = None):
        if denominator is not None:
            self = super().__new__(cls, value, denominator)
        else:
            if isinstance(value, str):
                value = _COMMON.get(value.strip(), value.strip())
            if isinstance(value, float):
                value = next((r for r in _COMMON.values() if abs(value - r) < NTSC_TOLERANCE),
                             Fraction(value).limit_denominator(1001))
            self = super().__new__(cls, value)
        if self <= 0:
            raise ValueError(f"rate must be positive, got {value!r}")
        return self

    @property
    def fps(self) -> float:
        return float(self)

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        return f"Rate({str(self)!r})"


def _round(x: Fraction, rounding: Rounding) -> int:
    if rounding == "floor":
        return math.floor(x)
    if rounding == "ceil":
        return math.ceil(x)
    return math.floor(x + Fraction(1, 2))


def _exact(seconds: float | Fraction | int | str) -> Fraction:
    # floats go through str() so 0.1 means 1/10, not its binary approximation
    return Fraction(str(seconds)) if isinstance(seconds, float) else Fraction(seconds)


def to_frame(seconds: float | Fraction | int | str, rate: Rate, rounding: Rounding = "floor") -> int:
    """Seconds → frame index."""
    return _round(_exact(seconds) * rate, rounding)


def to_seconds(frame: int, rate: Rate) -> Fraction:
    """Start time of ``frame`` in seconds (exact)."""
    return Fraction(frame) / rate


def convert(frame: int, src: Rate, dst: Rate) -> int:
    """Map a frame between rates by its centre (see module docstring)."""
    if src == dst:
        return frame
    return math.floor((Fraction(frame) + Fraction(1, 2)) * dst / src)


def frames_for(seconds: float | Fraction | int, rate: Rate) -> int:
    """Duration in seconds → whole number of frames (nearest)."""
    return to_frame(seconds, rate, "nearest")


def fmt_ts(seconds: float | Fraction, decimals: int = 1) -> str:
    """``72.44`` → ``"01:12.4"`` (mm:ss.f; hours roll into minutes)."""
    total = round(float(seconds), decimals)
    m, s = divmod(total, 60)
    width = 3 + decimals if decimals else 2
    return f"{int(m):02d}:{s:0{width}.{decimals}f}"


def parse_ts(text: str) -> Fraction:
    """``"01:12.4"`` / ``"72.4"`` / ``"1:02:03.5"`` → seconds."""
    total = Fraction(0)
    for part in text.strip().split(":"):
        total = total * 60 + Fraction(part)
    return total
