"""Reframe a wide shot to 9:16 following its subject (``index/subjects``: faces, then motion,
then saliency). The camera ignores moves inside a dead zone, eases like a critically damped spring
with a top speed, and emits crop keys every ``KEY_S`` with straight runs merged; faces get headroom.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from opencine.core.time import Rate, frames_for

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.schema import SpeedKey

KEY_S = 0.5
DEAD_ZONE = 0.12  # of the crop width: subject moves inside it do not move the camera
SPRING_HZ = 0.9  # how quickly the camera catches up
MAX_PAN = 0.45  # crop centre speed limit, frame widths per second
STRAIGHT = 0.004  # a key this close to the line through its neighbours is dropped
MARGIN_S = 0.5  # subject rows this far outside the item still steer it
WIDE = 1.15  # the crop must leave ≥ 13 % of the frame out before panning is worth it
HEADROOM = 0.12  # vertical pans: a face sits this much (of the crop height) above the centre
KIND_ORDER = ("face", "motion", "saliency")


def crop_fraction(sw: int, sh: int, out_w: int, out_h: int) -> tuple[float, float]:
    """``(w, h)`` of the largest crop with the deliverable's aspect, as fractions of the source."""
    aspect = out_w / out_h
    if sw / sh > aspect:
        return (sh * aspect) / sw, 1.0
    return 1.0, (sw / aspect) / sh


def _source_frames(n_out: int, in_: int, speed: float, ramp: list[SpeedKey] | None,
                   n_src: int) -> list[float]:
    if ramp:
        from opencine.plan.ramp import shown

        return [in_ + s for s in shown(ramp, n_src, n_out)]
    return [in_ + j * speed for j in range(n_out)]


def _targets(rows: list[dict], frames: list[float], axis: int, size: float) -> list[float] | None:
    """Subject centre along ``axis`` (0 = x, 1 = y) at each source frame, linear between rows
    and held outside; only the best kind found in the window is used, so the camera does not
    jump between a face and something moving."""
    if not rows:
        return None
    kind = next(k for k in KIND_ORDER if any(r["kind"] == k for r in rows))
    pts = []
    for r in rows:
        if r["kind"] != kind:
            continue
        x, y, w, h = r["box"]
        c = x + w / 2 if axis == 0 else y + h / 2 + (HEADROOM * size if kind == "face" else 0)
        pts.append((r["f"], c))
    out = []
    for f in frames:
        if f <= pts[0][0]:
            out.append(pts[0][1])
            continue
        for (a, va), (b, vb) in zip(pts, pts[1:], strict=False):
            if f < b:
                out.append(va + (vb - va) * (f - a) / (b - a))
                break
        else:
            out.append(pts[-1][1])
    return out


def follow(targets: list[float], size: float, rate: Rate) -> list[float]:
    """Crop centre per output frame following ``targets`` (fractions): a dead zone, then a
    critically damped spring toward the target with a top speed, kept inside the frame."""
    lo, hi = size / 2, 1 - size / 2
    dt = 1 / rate.fps
    omega = 2 * math.pi * SPRING_HZ
    dz = DEAD_ZONE * size
    c, v = min(hi, max(lo, targets[0])), 0.0  # centre, and its speed per second
    out = []
    for t in targets:
        err = t - c
        goal = c if abs(err) <= dz else t - math.copysign(dz, err)
        v += (omega * omega * (goal - c) - 2 * omega * v) * dt
        v = max(-MAX_PAN, min(MAX_PAN, v))
        c += v * dt
        if not lo <= c <= hi:
            c, v = min(hi, max(lo, c)), 0.0
        out.append(c)
    return out


def _keys(centres: list[float], size: float, rate: Rate) -> list[tuple[int, float]]:
    """Keyframes ``(at, left edge)`` every ``KEY_S`` plus the last frame, straight runs merged."""
    step = max(1, frames_for(KEY_S, rate))
    idx = sorted(set(range(0, len(centres), step)) | {len(centres) - 1})
    keys = [(j, round(centres[j] - size / 2, 4)) for j in idx]
    out = [keys[0]]
    for k in range(1, len(keys) - 1):
        (a, va), (j, vj), (b, vb) = out[-1], keys[k], keys[k + 1]
        if abs(vj - (va + (vb - va) * (j - a) / (b - a))) > STRAIGHT:
            out.append(keys[k])
    if len(keys) > 1:
        out.append(keys[-1])
    return out


def reframe(project: Project, clip: str, first: int, in_: int, out: int, n_out: int, *,
            speed: float = 1.0, ramp: list[SpeedKey] | None = None,
            deliverable: tuple[int, int] = (1080, 1920)) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Crop for an item showing clip frames ``first + [in_, out)`` over ``n_out`` output frames:
    a static box or keyframes (``EditItem.crop``), or None when the source has no room to pan
    or no subject was found."""
    from opencine.index.model import load_index, load_subjects

    info = load_index(project).clip(clip)
    wf, hf = crop_fraction(info.src_width, info.src_height, *deliverable)
    pan_x = wf * WIDE < 1
    pan_y = hf * WIDE < 1
    if not (pan_x or pan_y):
        return None
    rows = load_subjects(project, clip)
    if not rows:
        return None
    margin = frames_for(MARGIN_S, project.rate)
    a, b = first + in_ - margin, first + out + margin
    rows = [r for r in rows if a <= r["f"] < b]
    frames = [first + f for f in _source_frames(n_out, in_, speed, ramp, out - in_)]
    axis, size = (0, wf) if pan_x else (1, hf)
    targets = _targets(rows, frames, axis, size)
    if targets is None:
        return None
    keys = _keys(follow(targets, size, project.rate), size, project.rate)

    def box(edge: float) -> dict[str, float]:
        x, y = (edge, (1 - hf) / 2) if axis == 0 else ((1 - wf) / 2, edge)
        return {"x": round(max(0.0, x), 4), "y": round(max(0.0, y), 4), "w": round(wf, 4), "h": round(hf, 4)}

    if max(v for _, v in keys) - min(v for _, v in keys) <= STRAIGHT:
        return box(keys[0][1])
    return [{"at": j, **box(v)} for j, v in keys]


def _lerp(keys: list[tuple[int, float]], t: float) -> float:
    if t <= keys[0][0]:
        return keys[0][1]
    for (a, va), (b, vb) in zip(keys, keys[1:], strict=False):
        if t < b:
            return va + (vb - va) * (t - a) / (b - a)
    return keys[-1][1]


def outside_s(project: Project, clip: str, first: int, in_: int, out: int, crop: Any,
              rate: Rate, *, deliverable: tuple[int, int] = (1080, 1920)) -> float:
    """Seconds (source time, speed ignored) the tracked subject's centre is outside the crop."""
    from opencine.index.model import load_subjects
    from opencine.plan.schema import CropBox

    rows = [r for r in load_subjects(project, clip) or [] if first + in_ <= r["f"] < first + out]
    if not rows or crop is None:
        return 0.0
    kind = next(k for k in KIND_ORDER if any(r["kind"] == k for r in rows))
    step = 1 / 2  # rows every 0.5 s
    bad = 0
    for r in rows:
        if r["kind"] != kind:
            continue
        cx, cy = r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2
        if isinstance(crop, CropBox):
            x, y, w, h = crop.x, crop.y, crop.w, crop.h
        else:  # keyframes, linear between them (output frames ≈ source frames at speed 1)
            j = r["f"] - first - in_
            x, y, w, h = (_lerp([(k.at, getattr(k, a)) for k in crop], j) for a in "xywh")
        if not (x <= cx <= x + w and y <= cy <= y + h):
            bad += 1
    return bad * step
