"""Images for the Director, coarse → fine; made lazily, cached under each clip's ``proxy_key``.

- **L1** ``overview``: representative thumbnails (≤ 24 scenes/sheet), ``L1_TOKENS`` in total.
- **L2** ``scene_strip``: 4–6 frames of one scene, each labelled with its L3 ref (``S03@00:01.25``).
- **L3** ``frame_at``: one proxy frame: ``S03@2.5`` / ``S03@00:02.5`` (seconds into the scene),
  ``C01@12.4`` (seconds into the clip) or ``S03`` (representative); past the end is an error.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel

from opencine.core.cache import cache_key
from opencine.core.ffmpeg import run
from opencine.core.ledger import image_tokens
from opencine.core.time import Rate, fmt_ts, frames_for, parse_ts, to_frame, to_seconds
from opencine.index import frames
from opencine.index.frames import frame_name
from opencine.index.model import Scene, SceneIndex, load_index
from opencine.index.steps import StepCache, place
from opencine.ingest import proxy

if TYPE_CHECKING:
    from opencine.core.project import Project

VERSION = 2
MAX_EDGE = 1568  # L1 / L3 long edge
L1_TOKENS = 1800  # the overview sheets in total
L1_PER_SHEET = 24
L1_THUMB_MAX = 360  # thumbnail long edge
STRIP_EDGE = 1024
STRIP_MIN, STRIP_MAX = 4, 6
STRIP_GAP_S = 0.25  # strip frames closer than this are duplicates
JPEG_Q = 85
BG = (24, 24, 24)


class ImageRef(BaseModel):
    path: str  # absolute
    width: int
    height: int
    tokens: int  # image_tokens(width, height)

    @classmethod
    def of(cls, path: Path) -> ImageRef:
        with Image.open(path) as im:
            w, h = im.size
        return cls(path=str(Path(path).resolve()), width=w, height=h, tokens=image_tokens(w, h))


# ---- layout ---------------------------------------------------------------
def _grid(n: int, aspect: float, max_edge: int, max_cell: int,
          max_tokens: int | None = None) -> tuple[int, int, int, int]:
    """``(cols, rows, cell_w, cell_h)`` with the biggest cells of ``aspect`` (w/h) that fit."""
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        s = min(max_edge / max(cols * aspect, rows), max_cell / max(aspect, 1))
        if max_tokens:
            s = min(s, math.sqrt(max_tokens * 784 * 0.9 / (cols * rows * aspect)))
        score = (round(s, 3), -(cols * rows - n), -abs(cols * aspect - rows))
        if best is None or score > best[0]:
            best = (score, cols, rows, s)
    _, cols, rows, s = best
    while max_tokens and image_tokens(cols * int(s * aspect), rows * int(s)) > max_tokens:
        s *= 0.97
    return cols, rows, int(s * aspect), int(s)


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _tile(sheet: Image.Image, img: Path | Image.Image, box: tuple[int, int, int, int],
          label: str) -> None:
    x, y, w, h = box
    if isinstance(img, Image.Image):
        thumb = ImageOps.contain(img.convert("RGB"), (w - 2, h - 2))
    else:
        with Image.open(img) as im:
            # exif_transpose: phone stills carry their rotation in EXIF (proxy frames have none)
            thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (w - 2, h - 2))
    sheet.paste(thumb, (x + (w - thumb.width) // 2, y + (h - thumb.height) // 2))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(12, min(24, h // 8)))
    x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=font)
    pad = 3
    draw.rectangle((x + 1, y + 1, x + 1 + x1 - x0 + 2 * pad, y + 1 + y1 - y0 + 2 * pad), fill=(0, 0, 0))
    draw.text((x + 1 + pad - x0, y + 1 + pad - y0), label, font=font, fill=(255, 255, 255))


def _compose(tiles: list[tuple[Path | Image.Image, str]], aspect: float, max_edge: int,
             max_cell: int, max_tokens: int | None, out: Path) -> None:
    cols, rows, cw, ch = _grid(len(tiles), aspect, max_edge, max_cell, max_tokens)
    sheet = Image.new("RGB", (cols * cw, rows * ch), BG)
    for i, (img, label) in enumerate(tiles):
        _tile(sheet, img, ((i % cols) * cw, (i // cols) * ch, cw, ch), label)
    sheet.save(out, "JPEG", quality=JPEG_Q)


def compose(tiles: list[tuple[Path | Image.Image, str]], out: Path, *, aspect: float = 9 / 16,
            max_cell: int = 384, max_edge: int = MAX_EDGE, max_tokens: int | None = None) -> ImageRef:
    """A labelled grid of ``(image path or PIL image, label)`` tiles at ``out``, for other sheets
    (self-check, storyboard). Cells keep ``aspect``, long edge ≤ ``max_cell``, within ``max_edge``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _compose(tiles, aspect, max_edge, max_cell, max_tokens, out)
    return ImageRef.of(out)


# ---- frames ---------------------------------------------------------------
def _frame_jpg(project: Project, index: SceneIndex, clip: str, frame: int,
               steps: StepCache) -> Path:
    """Exact proxy frame as a JPEG (long edge ≤ ``MAX_EDGE``), cached."""
    c = index.clip(clip)
    rate = Rate(index.rate)
    frame = max(0, min(frame, c.frames - 1))

    def make(out: Path) -> None:
        # seek half a frame early: accurate seek drops frames with pts < target, landing exactly
        start = max(0.0, (frame - 0.5) / rate.fps)
        run(["-y", "-ss", f"{start:.6f}", "-i", str(project.root / c.proxy),
             "-frames:v", "1", "-q:v", "2", "-update", "1", str(out)])
        with Image.open(out) as im:
            if max(im.size) > MAX_EDGE:
                im = ImageOps.contain(im, (MAX_EDGE, MAX_EDGE))
                im.save(out, "JPEG", quality=JPEG_Q)

    key = cache_key(c.proxy_key or c.sha256, index.rate, proxy.PARAMS, frame, VERSION)
    return steps.file("frame", key, ".jpg", make)


def _aspect(index: SceneIndex, scenes: list[Scene]) -> float:
    """Cell aspect for the majority orientation of ``scenes``' clips."""
    clips = [index.clip(s.clip) for s in scenes]
    portrait = sum(c.height > c.width for c in clips)
    pick = [c for c in clips if (c.height > c.width) == (portrait * 2 > len(clips))]
    return pick[0].width / pick[0].height


def _representative(scene: Scene) -> float:
    return scene.keyframes[0] if scene.keyframes else (scene.start_s + scene.end_s) / 2


def _keyframes(scene: Scene, rate: Rate) -> list[int]:
    """Keyframes as proxy frames (stored seconds are rounded to ms: take the nearest frame)."""
    return scene.keyframe_frames or [to_frame(t, rate, "nearest") for t in scene.keyframes]


def _peak(scene: Scene, rate: Rate) -> int | None:
    if scene.motion_peak_frame is not None:
        return scene.motion_peak_frame
    return None if scene.motion_peak_s is None else to_frame(scene.motion_peak_s, rate, "nearest")


def _representative_frame(scene: Scene, rate: Rate) -> int:
    keys = _keyframes(scene, rate)
    return keys[0] if keys else (scene.start_frame + scene.end_frame - 1) // 2


def l1_budget(l0_tokens: int | None = None) -> int:
    """Total L1 tokens: always ``L1_TOKENS`` (``l0_tokens`` is ignored)."""
    return L1_TOKENS


# ---- L1 -------------------------------------------------------------------
def overview(project: Project, *, steps: StepCache | None = None,
             l0_tokens: int | None = None) -> list[ImageRef]:
    """L1 sheets ``index/sheets/overview.jpg`` (or ``overview1.jpg``, ``overview2.jpg``…);
    ``l0_tokens`` is ignored."""
    index = load_index(project)
    steps = steps or StepCache(project.cache_dir)
    rate = Rate(index.rate)
    budget = l1_budget()
    shown = index.shown_scenes()  # a clip sent twice appears once
    n = len(shown)
    k = max(1, math.ceil(n / L1_PER_SHEET))
    per = math.ceil(n / k) if n else 0
    groups = [shown[i:i + per] for i in range(0, n, per)] if n else []
    names = ["overview.jpg"] if k == 1 else [f"overview{i}.jpg" for i in range(1, k + 1)]

    refs = []
    for name, group in zip(names, groups, strict=False):
        tiles = []
        for s in group:
            jpg = project.frames_dir / frame_name(s.clip, _representative(s))
            if not jpg.exists():
                jpg = _frame_jpg(project, index, s.clip, _representative_frame(s, rate), steps)
            tiles.append((jpg, f"{s.id} {fmt_ts(s.start_s, 0)}"))
        aspect = _aspect(index, group)
        clip_keys = [index.clip(s.clip).proxy_key or index.clip(s.clip).sha256 for s in group]
        key = cache_key("L1", [(t[1], t[0].name) for t in tiles], clip_keys, frames.PARAMS,
                        proxy.PARAMS, index.rate, aspect, budget // k, VERSION)
        sheet = steps.file("sheet", key, ".jpg", lambda out, t=tiles, a=aspect: _compose(
            t, a, MAX_EDGE, L1_THUMB_MAX, budget // k, out))
        refs.append(ImageRef.of(place(sheet, project.sheets_dir / name)))
    for stale in project.sheets_dir.glob("overview*.jpg"):
        if stale.name not in names[:len(groups)]:
            stale.unlink()
    return refs


def images_sheet(project: Project, *, steps: StepCache | None = None) -> ImageRef:
    """L1 for stills: ``index/sheets/images.jpg``, one labelled tile per image (I01…)."""
    from opencine.index.table import describe_images

    images = describe_images(project)
    if not images:
        raise ValueError(f"project {project.pid} has no images")
    steps = steps or StepCache(project.cache_dir)
    portrait = sum(i["height"] > i["width"] for i in images) * 2 > len(images)
    aspect = 9 / 16 if portrait else 16 / 9
    tiles = [(project.root / i["file"], i["id"]) for i in images]
    stats = [(t[0].name, t[0].stat().st_size, t[0].stat().st_mtime_ns) for t in tiles]
    key = cache_key("L1img", stats, aspect, VERSION)
    sheet = steps.file("sheet", key, ".jpg", lambda out: _compose(
        tiles, aspect, MAX_EDGE, L1_THUMB_MAX, L1_TOKENS, out))
    return ImageRef.of(place(sheet, project.sheets_dir / "images.jpg"))


# ---- L2 -------------------------------------------------------------------
def _strip_frames(scene: Scene, rate: Rate) -> list[int]:
    """First, last, motion peak, medoids (deduped, ≤ 6), padded to 4 by halving the widest gap."""
    first, last = scene.start_frame, scene.end_frame - 1
    wanted = [first, last]
    if (peak := _peak(scene, rate)) is not None:
        wanted.append(peak)
    wanted += _keyframes(scene, rate)
    gap = frames_for(STRIP_GAP_S, rate)
    chosen: list[int] = []
    for f in wanted:
        if first <= f <= last and len(chosen) < STRIP_MAX and all(abs(f - c) >= gap for c in chosen):
            chosen.append(f)
    chosen.sort()
    while 2 <= len(chosen) < STRIP_MIN:  # a scene shorter than STRIP_GAP_S keeps its one frame
        width, i = max((b - a, i) for i, (a, b) in enumerate(zip(chosen, chosen[1:], strict=False)))
        if width < 2:
            break
        chosen.insert(i + 1, chosen[i] + width // 2)
    return chosen


def strip_label(scene: Scene, frame: int, rate: Rate) -> str:
    """The L3 ref of ``frame`` (``S03@00:01.25``): seconds into the scene, as ``frame`` takes."""
    return f"{scene.id}@{fmt_ts(to_seconds(frame - scene.start_frame, rate), 2)}"


def scene_strip(project: Project, sid: str, *, steps: StepCache | None = None) -> ImageRef:
    """L2 strip ``index/sheets/<sid>.jpg``."""
    index = load_index(project)
    steps = steps or StepCache(project.cache_dir)
    scene = index.scene(sid)
    rate = Rate(index.rate)
    c = index.clip(scene.clip)
    tiles = [(_frame_jpg(project, index, scene.clip, f, steps), strip_label(scene, f, rate))
             for f in _strip_frames(scene, rate)]
    key = cache_key("L2", scene.model_dump(), c.proxy_key or c.sha256, c.width, c.height, VERSION)
    sheet = steps.file("sheet", key, ".jpg", lambda out: _compose(
        tiles, c.width / c.height, STRIP_EDGE, max(c.width, c.height), None, out))
    return ImageRef.of(place(sheet, project.sheets_dir / f"{sid}.jpg"))


# ---- L3 -------------------------------------------------------------------
_REF = re.compile(r"^(?P<id>[SC]\d+)(?:@(?P<t>[\d:.,]+))?$", re.IGNORECASE)


def _span(n: int, rate: Rate) -> str:
    return f"{float(Fraction(n) / rate):.2f} s ({n} frames)"


def resolve_ref(index: SceneIndex, ref: str) -> tuple[str, int]:
    """``ref`` → ``(clip id, proxy frame)``. Offsets go to the nearest frame; past the end raises."""
    m = _REF.match(ref.strip())
    if not m:
        raise ValueError(f"bad frame ref {ref!r}; use S03, S03@2.5, S03@00:02.5 or C01@12.4")
    rid, t = m["id"].upper(), m["t"]
    try:
        offset = parse_ts(t.replace(",", ".")) if t else None
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"bad time in frame ref {ref!r}") from None
    rate = Rate(index.rate)
    if rid.startswith("S"):
        s = index.scene(rid)
        if offset is None:
            return s.clip, _representative_frame(s, rate)
        f = s.start_frame + to_frame(offset, rate, "nearest")
        if f >= s.end_frame:
            n = s.end_frame - s.start_frame
            raise ValueError(f"{ref} is past the end of {rid}: the offset is seconds into the "
                             f"scene, which lasts {_span(n, rate)}; its last frame is "
                             f"{strip_label(s, s.end_frame - 1, rate)}")
        return s.clip, f
    if offset is None:
        raise ValueError(f"clip ref needs a time, e.g. {rid}@12.4")
    c = index.clip(rid)
    f = to_frame(offset, rate, "nearest")
    if f >= c.frames:
        raise ValueError(f"{ref} is past the end of {rid}, which lasts {_span(c.frames, rate)}")
    return c.id, f


def frame_at(project: Project, ref: str, *, steps: StepCache | None = None) -> ImageRef:
    """L3: one full proxy frame ``index/sheets/<clip>_t<ms>.jpg``."""
    index = load_index(project)
    clip, f = resolve_ref(index, ref)
    jpg = _frame_jpg(project, index, clip, f, steps or StepCache(project.cache_dir))
    t_exact = float(to_seconds(f, Rate(index.rate)))
    return ImageRef.of(place(jpg, project.sheets_dir / frame_name(clip, t_exact)))


SHARPER = 1.15  # a neighbour replaces the asked frame only when this much sharper


def _sharpness(png: Path) -> float:
    """Variance of the Laplacian on a ≤1024 px grey copy (higher = sharper)."""
    import cv2
    import numpy as np

    with Image.open(png) as im:
        g = ImageOps.contain(im.convert("L"), (1024, 1024))
        return float(cv2.Laplacian(np.asarray(g, dtype=np.float32), cv2.CV_32F).var())


def source_frame(project: Project, ref: str, *, steps: StepCache | None = None,
                 search: int = 2) -> Path:
    """The frame of ``ref`` from the **source** at full resolution, as ``gen/refs/<clip>_t<ms>.png``:
    the input for generation (proxy frames are ≤ 720p JPEG).

    Rotation and HDR tone-mapping match the proxy, so colours match the edit. Up to ``search``
    frames either side (same scene) are tried too, since handheld frames are often blurred; a
    neighbour wins only if ``SHARPER`` × sharper. ``<png>.json`` records the pick.
    """
    from opencine.ingest.probe import probe

    index = load_index(project)
    clip, f = resolve_ref(index, ref)
    c = index.clip(clip)
    rate = Rate(index.rate)
    t = float(to_seconds(f, rate))
    src = project.root / c.source
    sc = steps or StepCache(project.cache_dir)

    def grab(frame: int) -> Path:
        tf = float(to_seconds(frame, rate))

        def make(out: Path) -> None:
            # proxy frame f shows the source frame at t (CFR resample); seek accurately to it
            vf = [proxy._tonemap(probe(src))] if c.hdr else []
            run(["-y", "-ss", f"{tf:.6f}", "-i", str(src), "-frames:v", "1",
                 "-vf", ",".join([*vf, "setsar=1", "format=rgb24"]), "-update", "1", str(out)])

        key = cache_key(c.sha256, "source_frame", round(tf * 1000), VERSION)
        return sc.file("source_frame", key, ".png", make)

    scene = next((s for s in index.scenes
                  if s.clip == clip and s.start_frame <= f < s.end_frame), None)
    lo, hi = (scene.start_frame, scene.end_frame - 1) if scene else (0, c.frames - 1)
    near = [g for k in range(1, max(search, 0) + 1) for g in (f - k, f + k) if lo <= g <= hi]

    def pick() -> dict:
        scores = {g: _sharpness(grab(g)) for g in [f, *near]}
        best = max(scores, key=lambda g: scores[g])
        chosen = best if scores[best] > scores[f] * SHARPER else f
        return {"frame": chosen, "sharpness": {str(g - f): round(v, 1) for g, v in scores.items()}}

    choice = (sc.json("source_frame_pick", cache_key(c.sha256, f, near, SHARPER, VERSION), pick)
              if near else {"frame": f, "sharpness": {}})
    dst = place(grab(choice["frame"]),
                project.root / "gen" / "refs" / frame_name(clip, t).replace(".jpg", ".png"))
    chosen_t = float(to_seconds(choice["frame"], rate))
    side = {"ref": ref, "t": round(t, 6), "chosen_t": round(chosen_t, 6),
            "offset_frames": choice["frame"] - f, "sharpness": choice["sharpness"]}
    dst.with_suffix(".json").write_text(json.dumps(side, indent=2), "utf-8")
    return dst
