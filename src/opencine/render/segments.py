"""Split an ``EditPlan`` into segments (runs between hard cuts) and build one FFmpeg
``filter_complex`` per segment.

A segment's cache key depends only on its own items, so editing one cut re-renders only the
segments touching it. Per item: frame-exact trim → speed / ramp → crop/zoom → colour LUT →
transitions (``xfade``, ``push``, mask wipe) → text (libass) → BT.709 tags. Animated crop/zoom is
sub-pixel (``perspective`` on a prescaled union of the windows; scale + crop alone judders in
2-px steps); magnified items are Lanczos-scaled and CAS-sharpened; HDR is tone-mapped after the
crop. Final renders retime native frames: speed-ups blur like a shutter (``tmix``), slow motion
is motion-interpolated when the window is small enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from opencine.core.cache import cache_key
from opencine.core.ffmpeg import BT709_TAGS, TO_BT709
from opencine.core.time import Rate
from opencine.plan import ramp as ramps
from opencine.plan.captions import Cue
from opencine.plan.media import Media, cover, moment, zoom_keys
from opencine.plan.schema import Captions, CropBox, EditItem, EditPlan, SpeedKey
from opencine.render.look import lut_filter
from opencine.render.text import ass_filter, ass_script, caption_events, fill, has_placeholder

RENDER_VERSION = 4
LEAD = 3  # plan frames decoded before `in`: the source frame showing at `in` may start earlier
TIE_S = Fraction(1, 1000)  # a source frame this close after a plan frame start still counts as on it

XFADE = {"dissolve": "fade", "flash": "fadewhite", "dip": "fadeblack", "zoom": "zoomin",
         "zoom_out": "fade"}  # zoom_out: the incoming item's own settle (media.zoom_keys)
# pushes (see push()): which way the picture moves, and whether it is smeared like a whip pan
PUSH = {"whip": ("left", True), "whip_left": ("right", True), "slide_left": ("left", False),
        "slide_right": ("right", False), "slide_up": ("up", False), "slide_down": ("down", False)}
SCALE_FLAGS = "lanczos"
CAS_FROM, CAS_MIN, CAS_MAX = 1.2, 0.2, 0.5  # magnification where sharpening starts; strengths
WHIP_SHUTTER = 0.5  # share of a frame's travel smeared into it (a 180° shutter)
SPEED_BLUR_FROM = 1.5  # final renders: speed-ups from here average the source frames they skip
MCI_MAX_PIXELS = 1920 * 1080  # motion-interpolate slow motion only when the shown window is this small
MCI_MAX_SPEED = Fraction(3, 4)  # … and this slow (milder slow motion repeats few frames)
MCI = "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"

Key = tuple[int, float]
Box = tuple[int, int, int, int]  # x, y, w, h in decoded px


@dataclass
class Segment:
    items: list[int]  # plan item indices
    start: int  # timeline frame (informational; not part of the key)
    frames: int
    key: str = ""
    inputs: list[str] = field(default_factory=list)
    graph: str = ""
    out: str = ""  # graph output label
    ass: str | None = None  # ASS script to write next to the output, if any


def split(plan: EditPlan) -> list[list[int]]:
    """Item indices grouped into segments; a new segment starts at every hard cut."""
    groups: list[list[int]] = []
    for i, it in enumerate(plan.items):
        if i == 0 or it.is_cut:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _f(v: float | Fraction) -> str:
    return f"{float(v):.6f}".rstrip("0").rstrip(".") or "0"


def _pw(keys: list[Key], var: str = "n") -> str:
    """Piecewise-linear expression in the frame number ``var``; held outside the keys."""
    if all(v == keys[0][1] for _, v in keys):
        return _f(keys[0][1])
    expr = _f(keys[-1][1])
    for (a, va), (b, vb) in reversed(list(zip(keys, keys[1:], strict=False))):
        expr = f"if(lt({var},{b}),{_f(va)}+({_f(vb - va)})*({var}-{a})/{b - a},{expr})"
    return f"if(lt({var},{keys[0][0]}),{_f(keys[0][1])},{expr})"


def _at(keys: list[Key], t: int) -> float:
    """``_pw`` evaluated in Python."""
    if t < keys[0][0]:
        return keys[0][1]
    for (a, va), (b, vb) in zip(keys, keys[1:], strict=False):
        if t < b:
            return va + (vb - va) * (t - a) / (b - a)
    return keys[-1][1]


def _keys(it: EditItem, m: Media, width: int, height: int):
    """Crop boxes (decoded px, output aspect) and zoom keys ``(at, scale, cx, cy)``."""
    sw, sh = m.width, m.height
    if isinstance(it.crop, list) and it.crop:
        crops = [(k.at, cover((k.x, k.y, k.w, k.h), sw, sh, width, height)) for k in it.crop]
    else:
        box = it.crop if isinstance(it.crop, CropBox) else None
        crops = [(0, cover(box and (box.x, box.y, box.w, box.h), sw, sh, width, height))]
    return crops, zoom_keys(it, m.kind)


def _window(crops, zooms, t: int, aspect: float) -> tuple[float, float, float, float]:
    """Source window ``(x, y, w, h)`` shown at output frame ``t``."""
    x, y, wd = (_at([(a, c[i]) for a, c in crops], t) for i in range(3))
    z, cx, cy = (_at([(k[0], k[i]) for k in zooms], t) for i in (1, 2, 3))
    w, hd = wd / z, wd * aspect
    h = hd / z
    return x + (wd - w) * cx, y + (hd - h) * cy, w, h


def _to_sdr(m: Media) -> list[str]:
    """Right after the scale that sets the size: HDR tone-map, then 8-bit 4:2:0 (forcing the
    BT.709 conversion to happen in that scale)."""
    return [*([m.tonemap] if m.tonemap else []), "format=yuv420p"]


def _sharpen(mag: float) -> list[str]:
    """CAS on luma for items magnified more than ``CAS_FROM``× (output px per decoded px)."""
    if mag <= CAS_FROM:
        return []
    return [f"cas=strength={_f(round(min(CAS_MAX, CAS_MIN + 0.25 * (mag - CAS_FROM)), 2))}:planes=1"]


def _even_up(v: float) -> int:
    return max(2, math.ceil(v / 2 - 1e-9) * 2)


def _static(crops, zooms) -> bool:
    return len({c for _, c in crops}) == 1 and len({z[1:] for z in zooms}) == 1


def _static_box(crops, zooms, sw: int, sh: int) -> Box:
    (x, y, w, h), (z, cx, cy) = crops[0][1], zooms[0][1:]
    zw, zh = w / z, h / z
    zx, zy = x + (w - zw) * cx, y + (h - zh) * cy
    cw, ch = min(sw, max(2, round(zw))), min(sh, max(2, round(zh)))
    return min(sw - cw, max(0, round(zx))), min(sh - ch, max(0, round(zy))), cw, ch


def _union(wins, sw: int, sh: int) -> Box:
    ux = max(0, math.floor(min(x for x, _, _, _ in wins)) // 2 * 2)
    uy = max(0, math.floor(min(y for _, y, _, _ in wins)) // 2 * 2)
    uw = min(sw - ux, _even_up(max(x + w for x, _, w, _ in wins) - ux))
    uh = min(sh - uy, _even_up(max(y + h for _, y, _, h in wins) - uy))
    return ux, uy, uw, uh


def source_window(it: EditItem, m: Media, width: int, height: int, t: int) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` in decoded px of the picture the item shows at its output frame ``t``
    (crop and zoom applied), framed at ``width×height``."""
    crops, zooms = _keys(it, m, width, height)
    return _window(crops, zooms, t, height / width)


def window_box(it: EditItem, m: Media, width: int, height: int) -> Box:
    """``(x, y, w, h)`` in decoded px holding every frame's source window of the item: all the
    picture the item ever shows (cropping to it first keeps per-frame work small)."""
    crops, zooms = _keys(it, m, width, height)
    if _static(crops, zooms):
        return _static_box(crops, zooms, m.width, m.height)
    return _union([_window(crops, zooms, t, height / width) for t in range(it.length)],
                  m.width, m.height)


def geometry(it: EditItem, m: Media, width: int, height: int, box: Box | None = None) -> str:
    """Crop box (static or keyed) + zoom keys → filters producing exactly ``width×height``
    BT.709 yuv420p (HDR media: through ``m.tonemap``). ``box``: the frames arrive already
    cropped to it (a ``window_box``), so offsets are taken relative to its corner."""
    sw, sh = m.width, m.height
    ox, oy = (box[0], box[1]) if box else (0, 0)
    crops, zooms = _keys(it, m, width, height)
    colour = "" if m.tonemap else f":{TO_BT709}"

    if _static(crops, zooms):
        cx_px, cy_px, cw, ch = _static_box(crops, zooms, sw, sh)
        return ",".join([f"crop={cw}:{ch}:{cx_px - ox}:{cy_px - oy}",
                         f"scale={width}:{height}:flags={SCALE_FLAGS}{colour}",
                         *_to_sdr(m), *_sharpen(width / cw)])

    # Union of every frame's window, cropped once (even, inside the frame) and prescaled so the
    # warp below stays within √(max/min) of 1:1; `perspective` then sends that frame to where the
    # window lands on the output (top-left W×H), at sub-pixel precision.
    aspect = height / width
    wins = [_window(crops, zooms, t, aspect) for t in range(it.length)]
    ux, uy, uw, uh = _union(wins, sw, sh)
    lo, hi = min(w for _, _, w, _ in wins), max(w for _, _, w, _ in wins)
    k = width / math.sqrt(lo * hi)
    pw, ph = max(width, _even_up(k * uw)), max(height, _even_up(k * uh))

    n = "(in-1)"  # perspective counts input frames from 1
    Z = _pw([(k_[0], k_[1]) for k_ in zooms], n)
    CX = _pw([(k_[0], k_[2]) for k_ in zooms], n)
    CY = _pw([(k_[0], k_[3]) for k_ in zooms], n)
    X, Y, Wd = (_pw([(a, c[i]) for a, c in crops], n) for i in range(3))
    # ld(0) zoom, ld(1) crop width, ld(2) output px per source px
    head = f"st(0,{Z});st(1,{Wd});st(2,{width}*ld(0)/ld(1));"
    xw = f"({X}+ld(1)*(1-1/ld(0))*{CX})"
    yw = f"({Y}+ld(1)*{_f(aspect)}*(1-1/ld(0))*{CY})"
    left, right = f"{head}-({xw}-{ux})*ld(2)", f"{head}({ux + uw}-{xw})*ld(2)"
    top, bottom = f"{head}-({yw}-{uy})*ld(2)", f"{head}({uy + uh}-{yw})*ld(2)"
    warp = (f"perspective=x0='{left}':y0='{top}':x1='{right}':y1='{top}':x2='{left}':y2='{bottom}'"
            f":x3='{right}':y3='{bottom}':interpolation=cubic:sense=destination:eval=frame")
    return ",".join([f"crop={uw}:{uh}:{ux - ox}:{uy - oy}", f"scale={pw}:{ph}:flags={SCALE_FLAGS}{colour}",
                     *_to_sdr(m), warp, f"crop={width}:{height}:0:0", *_sharpen(width / lo)])


def push(a: str, b: str, off: int, d: int, size: tuple[int, int], out: str, tag: str, rate: Rate,
         direction: str = "left", blur: bool = True) -> list[str]:
    """Push from ``a`` to ``b``: frames ``off…off+d-1`` of ``a`` and ``0…d-1`` of ``b`` side by
    side (stacked for up/down), the picture moving ``direction`` with smoothstep easing (frame 0
    is clean ``a``, like ``xfade``). A whip (``blur``) box-blurs each frame along the motion by
    ``WHIP_SHUTTER`` of its travel, so it reads as camera motion, not a slide. Same frame
    accounting as ``xfade``."""
    width, height = size
    vertical = direction in ("up", "down")
    span = height if vertical else width
    ease = lambda p: p * p * (3 - 2 * p)  # noqa: E731
    xs = [2 * round(span * ease(t / d) / 2) for t in range(d)]
    if direction in ("right", "down"):  # b sits before a: the window travels back to 0
        xs = [span - x for x in xs]
    expr = "+".join(f"{x}*eq(n,{t})" for t, x in enumerate(xs) if x) or "0"
    smear = []
    for t in range(1, d) if blur else ():
        travel = span * (ease(min(1.0, (t + 0.5) / d)) - ease(max(0.0, (t - 0.5) / d)))
        if (r := round(travel * WHIP_SHUTTER / 2)) >= 1:
            axis = f"sizeX=1:sizeY={r}" if vertical else f"sizeX={r}:sizeY=1"
            smear.append(f"avgblur={axis}:enable='eq(n,{t})'")
    p = tag
    chains = []
    if off:
        chains += [f"[{a}]split[{p}a0][{p}a1]", f"[{p}a0]trim=end_frame={off}[{p}pre]"]
        a = f"{p}a1"
    first, second = (f"[{p}at]", f"[{p}bh]") if direction in ("left", "up") else (f"[{p}bh]", f"[{p}at]")
    stack, window = ("vstack", f"crop=iw:{height}:x=0:y='{expr}'") if vertical         else ("hstack", f"crop={width}:ih:x='{expr}':y=0")
    chains += [
        f"[{a}]trim=start_frame={off}:end_frame={off + d},setpts=PTS-STARTPTS[{p}at]",
        f"[{b}]split[{p}b0][{p}b1]",
        f"[{p}b0]trim=end_frame={d},setpts=PTS-STARTPTS[{p}bh]",
        f"[{p}b1]trim=start_frame={d},setpts=PTS-STARTPTS[{p}post]",
        f"{first}{second}{stack},{window}" + "".join("," + f for f in smear) + f"[{p}mid]",
    ]
    parts = ([f"[{p}pre]"] if off else []) + [f"[{p}mid]", f"[{p}post]"]
    # concat outputs in microseconds; the next xfade needs the items' 1/rate time base
    chains.append(f"{''.join(parts)}concat=n={len(parts)}:v=1:a=0,"  # 1-frame pieces: count
                  f"settb={rate.denominator}/{rate.numerator},setpts=N[{out}]")
    return chains


def mask_wipe(a: str, b: str, matte: str, off: int, d: int, out: str, tag: str,
              rate: Rate) -> list[str]:
    """``b`` revealed through the matte (``opencine.render.matte``) over frames ``off…off+d-1`` of
    ``a`` / ``0…d-1`` of ``b``; same frame accounting as ``xfade`` and ``push``."""
    p, tb = tag, f"{rate.denominator}/{rate.numerator}"
    chains = []
    if off:
        chains += [f"[{a}]split[{p}a0][{p}a1]", f"[{p}a0]trim=end_frame={off}[{p}pre]"]
        a = f"{p}a1"
    chains += [
        f"[{a}]trim=start_frame={off}:end_frame={off + d},setpts=PTS-STARTPTS[{p}at]",
        f"[{b}]split[{p}b0][{p}b1]",
        f"[{p}b0]trim=end_frame={d},setpts=PTS-STARTPTS[{p}bh]",
        f"[{p}b1]trim=start_frame={d},setpts=PTS-STARTPTS[{p}post]",
        f"[{matte}]format=gray,settb={tb},setpts=N,trim=end_frame={d}[{p}m]",
        f"[{p}bh][{p}m]alphamerge,format=yuva420p[{p}ba]",
        f"[{p}at][{p}ba]overlay=format=yuv420,format=yuv420p[{p}mid]",
    ]
    parts = ([f"[{p}pre]"] if off else []) + [f"[{p}mid]", f"[{p}post]"]
    chains.append(f"{''.join(parts)}concat=n={len(parts)}:v=1:a=0,settb={tb},setpts=N[{out}]")
    return chains


FIT_DIM = 0.3  # fit_blur background: this much darker (luma above black, limited range kept)


def item_size(it: EditItem, width: int, height: int) -> tuple[int, int]:
    """The picture size an item is framed at: the output, or a 9:16 column of it (fit_blur)."""
    if it.layout == "fit_blur":
        return max(2, round(height * 9 / 16 / 2) * 2), height
    return width, height


def fit_blur(src: str, out: str, size: tuple[int, int], tag: str) -> list[str]:
    """``src`` (a 9:16 column) centred over a blurred, darker copy of itself filling ``size``."""
    w, h = size
    r = max(4, h // 30)
    return [f"[{src}]split[{tag}f][{tag}b]",
            f"[{tag}b]scale={w}:{h}:force_original_aspect_ratio=increase:flags=bilinear,crop={w}:{h},"
            f"boxblur=luma_radius={r}:luma_power=2:chroma_radius={r // 2}:chroma_power=2,"
            f"lutyuv=y='16+(val-16)*{_f(1 - FIT_DIM)}'[{tag}g]",
            f"[{tag}g][{tag}f]overlay=x=(W-w)/2:y=0,format=yuv420p,setsar=1[{out}]"]


def _drop_defaults(d: dict) -> None:
    """Fields added after a cache key was first written stay out of it at their defaults, so
    plans that do not use them keep their rendered segments."""
    if d.get("ramp") is None:
        d.pop("ramp", None)
    if d.get("color") == "auto":
        d.pop("color")
    if d.get("layout") == "fill":
        d.pop("layout")
    if isinstance(d.get("transition_in"), dict) and d["transition_in"].get("point") is None:
        d["transition_in"].pop("point", None)
    if isinstance(d.get("text"), dict):
        for k, v in (("style", "plain"), ("anim", "none")):
            if d["text"].get(k) == v:
                d["text"].pop(k)


def _identity(m: Media) -> list:
    st = m.path.stat()
    return [str(m.path.resolve()), st.st_size, st.st_mtime_ns]


def _wants_mci(m: Media, sp: Fraction, rate: Rate, final: bool, box: Box) -> bool:
    """Final slow motion (≤ ``MCI_MAX_SPEED``) without enough native frames, on an SDR item
    whose shown window is small enough to interpolate at a usable speed."""
    src_fps = m.fps or Fraction(rate)
    return (final and sp <= MCI_MAX_SPEED and not m.tonemap and box[2] * box[3] <= MCI_MAX_PIXELS
            and src_fps < Fraction(9, 10) * rate / sp)


def _speed_chain(m: Media, sp: Fraction, rate: Rate, lead: int, n_src: int, final: bool,
                 mci: Box | None) -> tuple[str, str]:
    """``(before, after)``: filters from the seek shift through the trim to ``in``, and after.

    Previews (and speed 1) resample to the plan rate first, then retime plan frames (slow motion
    repeats them). Finals retime the native frames: slow motion interpolates when ``mci`` (the
    item's window, cropped first) is given; speed-ups average the native frames each output
    frame spans, a window *starting* at the frame shown (tmix trails, so each average is
    re-stamped with its first frame's time and never reaches back before ``in``)."""
    if not (final and sp != 1):
        retime = (f",settb={rate.denominator}/{rate.numerator * sp.numerator},"
                  f"setpts=N*{sp.denominator},fps={rate}:round=up") if sp != 1 else ""
        return (f"fps={rate}:round=up,trim=start_pts={lead}:end_pts={lead + n_src}",
                f"setpts=PTS-STARTPTS{retime}")
    src_fps = m.fps or Fraction(rate)
    before = []
    if mci:
        x, y, w, h = mci
        before.append(f"crop={w}:{h}:{x}:{y},minterpolate=fps={Fraction(rate) / sp}:{MCI}")
    start = math.ceil(lead / sp)  # `in` lands on this output frame; earlier ones are dropped
    before.append(f"setpts=((PTS*TB-{_f(Fraction(lead) / rate)})/{_f(sp)}"
                  f"+{_f(Fraction(start) / rate)})/TB")
    if sp >= SPEED_BLUR_FROM and (k := round(sp * src_fps / rate)) >= 2:
        back = Fraction(k - 1) / (src_fps * sp)  # k-1 native frames, in output seconds
        before.append(f"tmix=frames={k},setpts=PTS-{_f(back)}/TB")
    before.append(f"fps={rate}:round=up,trim=start_pts={start}")
    return ",".join(before), "setpts=PTS-STARTPTS"


def _ramp_chain(m: Media, keys: list[SpeedKey], rate: Rate, lead: int, n_src: int, final: bool,
                mci: Box | None, slowest: Fraction) -> tuple[str, str]:
    """``(before, after)`` like ``_speed_chain`` for a ramp: the source offset ``s`` is retimed
    to ``T(s)`` (``opencine.plan.ramp.expr``). Previews work on plan-rate frames (``s = N``) in a
    1/``TB_STEPS``-frame time base (setpts truncates); finals on native frames, from the
    seek-relative time, with the frames before ``in`` dropped as for a constant speed."""
    den, num = rate.denominator, rate.numerator
    if not final:
        return (f"fps={rate}:round=up,trim=start_pts={lead}:end_pts={lead + n_src}",
                f"setpts=PTS-STARTPTS,settb={den}/{num * ramps.TB_STEPS},"
                f"setpts='({ramps.expr(keys, 'N')})*{ramps.TB_STEPS}',fps={rate}:round=up")
    before = []
    if mci:
        x, y, w, h = mci
        before.append(f"crop={w}:{h}:{x}:{y},minterpolate=fps={Fraction(rate) / slowest}:{MCI}")
    start = math.ceil(lead / keys[0].speed - 1e-9)  # `in` lands on this output frame
    s = f"(PTS*TB*{num}/{den}-{lead})"
    before.append(f"setpts='(({ramps.expr(keys, s)})+{start})*{den}/{num}/TB'")
    before.append(f"fps={rate}:round=up,trim=start_pts={start}")
    return ",".join(before), "setpts=PTS-STARTPTS"


def build(plan: EditPlan, idxs: list[int], medias: list[Media], start: int,
          size: tuple[int, int], enc: list[str], font: str, final: bool = False,
          luts: list[Path | None] | None = None,
          mattes: dict[int, Path | None] | None = None,
          captions: tuple[Captions, list[Cue]] | None = None) -> Segment:
    """Segment spec → cache key, ffmpeg inputs and filter graph. ``final`` retimes native
    frames for speed changes (see ``_speed_chain``); ``luts``: per plan item, the colour LUT
    (``opencine.render.look``) applied after the geometry, or None; ``mattes``: per plan item with a
    ``mask`` transition, its matte (``opencine.render.matte``), None = too little passed (a whip)."""
    rate: Rate = plan.fps
    W, H = size
    items = [plan.items[i] for i in idxs]
    spec = {"v": RENDER_VERSION, "rate": str(rate), "size": [W, H], "enc": enc, "font": font,
            "final": final, "items": []}
    inputs: list[str] = []
    chains: list[str] = []
    events = []
    pos = 0
    positions = []
    for k, (i, it) in enumerate(zip(idxs, items, strict=True)):
        m = medias[i]
        n_src, n = it.out - it.in_, it.length
        if k and not it.is_cut:
            pos -= it.transition_in.frames
        positions.append(pos)
        d = it.model_dump(by_alias=True, exclude={"note", "gain", "audio_lead", "audio_tail"})
        _drop_defaults(d)
        lut = luts[i] if luts else None
        if lut is not None:
            d["lut"] = lut.stem  # the file is named by its content hash
        if it.transition_in.kind == "mask" and k:
            mp = (mattes or {}).get(i)
            d["matte"] = mp.stem if mp is not None else None
        if k == 0:
            d["transition_in"] = {"kind": "cut", "frames": 0}
        spec["items"].append({"media": _identity(m) + [m.kind, str(m.start_s), m.width, m.height,
                                                       m.prefilter, m.native_filter, m.tonemap,
                                                       str(m.fps)],
                              "item": d})
        fw, fh = item_size(it, W, H)
        geom = geometry(it, m, fw, fh)
        grade = f",{lut_filter(lut)}" if lut is not None else ""
        label = f"v{k}" if (fw, fh) == (W, H) else f"c{k}"  # fit_blur: the column, composed below
        if m.kind == "image":
            inputs += ["-loop", "1", "-framerate", str(rate), "-t", _f(Fraction(n + 2) / rate),
                       "-i", str(m.path)]
            chains.append(f"[{k}:v]setpts=PTS-STARTPTS,fps={rate},trim=end_frame={n},"
                          f"setpts=PTS-STARTPTS,{geom}{grade},setsar=1[{label}]")
        else:
            # Seek `lead` plan frames before `in` and keep the seek-relative timestamps, so the
            # fps grid lands on `in` exactly; round=up makes output frame j show the source
            # frame *containing* its time (frame f covers [f/r, (f+1)/r)), at any source rate.
            t_in = m.start_s + Fraction(it.in_) / rate
            lead = min(LEAD, math.floor(t_in * rate))
            # (-ss lands on the stream's tick, not exactly on t_in − lead: TIE_S absorbs that)
            ss, shift = t_in - Fraction(lead) / rate, -TIE_S
            if lead == 0 and t_in > 0:  # off-grid start inside the first plan frame
                lead, ss = 1, Fraction(0)
                shift += Fraction(1) / rate - t_in
            keys = it.ramp or None
            sp = Fraction(str(ramps.speeds(keys)[0] if keys else it.speed)).limit_denominator(100)
            box = window_box(it, m, fw, fh)
            mci = box if _wants_mci(m, sp, rate, final, box) else None
            if mci:
                geom = geometry(it, m, fw, fh, box)
            if keys:
                before, after = _ramp_chain(m, keys, rate, lead, n_src, final, mci, sp)
            else:
                before, after = _speed_chain(m, sp, rate, lead, n_src, final, mci)
            retimed = bool(keys) or sp != 1
            prefilter = m.native_filter if final and retimed and m.native_filter is not None \
                else m.prefilter
            pre = (prefilter + "," if prefilter else "") + f"settb=AVTB,setpts=PTS{'+' if shift > 0 else '-'}{_f(abs(shift))}/TB,"
            if ss > 0:
                inputs += ["-ss", _f(ss)]
            inputs += ["-t", _f(Fraction(lead + n_src + 2) / rate), "-i", str(m.path)]
            chains.append(f"[{k}:v]{pre}{before},{after},"
                          f"tpad=stop_mode=clone:stop=4,trim=end_frame={n},setpts=PTS-STARTPTS,"
                          f"{geom}{grade},setsar=1[{label}]")
        if label != f"v{k}":
            chains += fit_blur(label, f"v{k}", (W, H), f"b{k}")
        if it.text is not None:
            t_end = min(n, it.text.end if it.text.end is not None else n)
            text = it.text
            if has_placeholder(text.content):
                text = text.model_copy(update={"content": fill(text.content, moment(m, it.in_, rate))})
            events.append((Fraction(pos + it.text.start) / rate, Fraction(pos + t_end) / rate, text))
        pos += n
    total = pos

    acc = "v0"
    for k in range(1, len(items)):
        it, d = items[k], items[k].transition_in.frames
        off = positions[k]
        out = f"x{k}"
        kind = it.transition_in.kind
        matte = (mattes or {}).get(idxs[k]) if kind == "mask" else None
        if matte is not None:
            n_in = inputs.count("-i")  # the matte is the next ffmpeg input
            inputs += ["-i", str(matte)]
            chains += mask_wipe(acc, f"v{k}", f"{n_in}:v", off, d, out, f"m{k}", rate)
        elif (pk := PUSH.get("whip" if kind == "mask" else kind)) is not None:
            chains += push(acc, f"v{k}", off, d, (W, H), out, f"w{k}", rate, *pk)
        else:
            chains.append(f"[{acc}][v{k}]xfade=transition={XFADE[it.transition_in.kind]}:"
                          f"duration={_f(Fraction(d) / rate)}:offset={_f(Fraction(off) / rate)}[{out}]")
        acc = out

    seg = Segment(items=idxs, start=start, frames=total, inputs=inputs)
    cap_lines = caption_events(captions[1], captions[0], Fraction(start) / rate,
                               Fraction(start + total) / rate, W, H) if captions else []
    if events or cap_lines:
        seg.ass = ass_script(events, W, H, font, (captions[0], cap_lines) if cap_lines else None,
                             accent=(captions[0].highlight if captions else None) or "#FFD21F")
        spec["ass"] = seg.ass
    bars = letterbox_filter(plan.letterbox, W, H)
    if bars:
        spec["letterbox"] = plan.letterbox
    seg.key = cache_key("segment", spec)
    chains.append(f"[{acc}]" + (f"{bars}," if bars else "") + ("{ASS}," if seg.ass else "")
                  + f"format=yuv420p,{BT709_TAGS}[vout]")
    seg.graph = ";".join(chains)
    seg.out = "vout"
    return seg


def letterbox_filter(ratio: float | None, w: int, h: int) -> str | None:
    """Black bars that frame a ``w``×``h`` picture at ``ratio`` (width / height): top and
    bottom when the frame is taller than that, the sides when it is wider; None when no bar
    would be a pixel high."""
    if not ratio:
        return None
    if w / h < ratio:
        bar, box = round((h - w / ratio) / 2), "x=0:y={}:w=iw:h={}"
        far = "ih-{}"
    else:
        bar, box = round((w - h * ratio) / 2), "x={}:y=0:w={}:h=ih"
        far = "iw-{}"
    if bar < 1:
        return None
    return ",".join(f"drawbox={box.format(at, bar)}:color=black:t=fill"
                    for at in ("0", far.format(bar)))


def finalize_graph(seg: Segment, ass_path: Path | None) -> str:
    """Substitute the ASS filter (its path depends on where the cache lives)."""
    return seg.graph.replace("{ASS}", ass_filter(ass_path)) if ass_path else seg.graph
