"""Resolve an ``EditItem`` source to the file the renderer decodes, plus its geometry and limits."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from fractions import Fraction
from functools import cached_property, partial
from pathlib import Path
from typing import Literal

from PIL import Image

from opencine.core.ffmpeg import probe_json
from opencine.core.project import Project
from opencine.core.time import Rate, to_frame
from opencine.index.model import ClipInfo, SceneIndex, load_index
from opencine.ingest.probe import HDR_TRANSFERS, ProbeInfo
from opencine.ingest.proxy import decode_filters, input_tags
from opencine.plan.schema import ClipSource, EditItem, GenSource, ImageSource, SceneSource, Source


@dataclass(frozen=True)
class Media:
    kind: Literal["video", "image"]
    path: Path  # decoded file (proxy, generated clip or still)
    start_s: Fraction  # file time of source frame 0 (scene start for scenes)
    width: int  # decoded pixels
    height: int
    full_width: int  # original resolution (for punch-in headroom)
    full_height: int
    soft_frames: int | None  # plan-rate frames in the referenced range (scene length); None = unbounded
    hard_frames: int | None  # plan-rate frames until the file ends
    audio: Path | None  # file whose audio aligns with ``path`` time
    # original source, decoded by the final render instead of the 720p proxy (scene/clip only)
    master: Path | None = None
    master_width: int = 0  # after the proxy normalisation (rotation, even size)
    master_height: int = 0
    master_filter: str = ""  # size normalisation at native size + fps=rate (colour untouched)
    prefilter: str = ""  # applied right after decode (master view; untagged-matrix tags)
    fps: Fraction | None = None  # native frame rate of ``path`` (None: unknown / a still)
    tonemap: str = ""  # HDR → SDR BT.709, applied after the renderer's crop and scale
    native_filter: str | None = None  # ``prefilter`` without the rate conversion (None: same)
    master_native: str = ""
    master_tonemap: str = ""
    master_fps: Fraction | None = None
    shot_at: datetime | None = None  # wall-clock time of source frame 0 (clips with a shooting time)
    # the master's filters need a probe of it: done only when a final render decodes it
    master_probe: Callable[[], MasterView] | None = field(default=None, compare=False, repr=False)

    def decoded(self, final: bool) -> Media:
        """The view the renderer decodes: the master for final renders, else the proxy."""
        if not (final and self.master):
            return self
        if self.master_probe is not None:
            mv = self.master_probe()
            return replace(self, path=self.master, width=mv.width, height=mv.height,
                           prefilter=mv.prefilter, native_filter=mv.native, tonemap=mv.tonemap,
                           fps=mv.fps, master=None, master_probe=None)
        return replace(self, path=self.master, width=self.master_width, height=self.master_height,
                       prefilter=self.master_filter, native_filter=self.master_native,
                       tonemap=self.master_tonemap, fps=self.master_fps, master=None)


def _shot(clip: ClipInfo, offset_s: Fraction) -> datetime | None:
    if not clip.shot_at:
        return None
    try:
        return datetime.fromisoformat(clip.shot_at) + timedelta(seconds=float(offset_s))
    except ValueError:
        return None


def moment(m: Media, in_frame: int, rate: Rate) -> datetime | None:
    """Wall-clock time of an item's first source frame, when its clip has a shooting time."""
    return None if m.shot_at is None else m.shot_at + timedelta(seconds=in_frame / float(rate))


def _even(v: float) -> int:
    return max(2, round(v / 2) * 2)


def _master_size(clip: ClipInfo) -> tuple[int, int, int]:
    w, h = clip.src_width, clip.src_height
    short = max(2, min(w, h) // 2 * 2)
    mw, mh = (_even(short * w / h), short) if w >= h else (short, _even(short * h / w))
    return mw, mh, short


def _master_info(clip: ClipInfo, color_space: str | None) -> ProbeInfo:
    trc = {v: k for k, v in HDR_TRANSFERS.items()}.get(clip.hdr or "")
    return ProbeInfo.model_construct(hdr=clip.hdr, color_transfer=trc, color_primaries=None,
                                     color_space=color_space, width=clip.src_width,
                                     height=clip.src_height)


@dataclass(frozen=True)
class MasterView:
    width: int
    height: int
    prefilter: str  # size (+ untagged-matrix tags) + fps=rate
    native: str  # the same without the rate conversion
    tonemap: str  # HDR only, run by the renderer after its crop
    fps: Fraction | None = None  # native frame rate


def master_view(clip: ClipInfo, rate: Rate, color_space: str | None,
                fps: Fraction | None = None) -> MasterView:
    """How the final render decodes the original: like the proxy, at native size, with the HDR
    tone-map split off (it runs on the cropped output pixels) and a native-rate variant."""
    mw, mh, short = _master_size(clip)
    info = _master_info(clip, color_space)
    pre, tm = decode_filters(info, rate, short, fps=True)
    native, _ = decode_filters(info, rate, short, fps=False)
    return MasterView(mw, mh, pre, native, tm, fps)


def _rate_of(stream: dict) -> Fraction | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            r = Fraction(stream.get(key) or "0")
        except (ValueError, ZeroDivisionError):
            continue
        if r > 0:
            return r
    return None


def _need(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def project_path(project: Project | None, p: str) -> Path:
    path = Path(p)
    if path.is_absolute() or project is None:
        return path
    return project.root / path


class Resolver:
    """Resolves sources for one project at a plan rate; caches the index and probes."""

    def __init__(self, project: Project, rate: Rate):
        self.project, self.rate = project, rate
        self._probes: dict[Path, Media] = {}
        self._colour: dict[Path, str | None] = {}  # master matrix tags
        self._fps: dict[Path, Fraction | None] = {}  # master native frame rates

    @cached_property
    def index(self) -> SceneIndex:
        return load_index(self.project)

    def media(self, item: EditItem) -> Media:
        return self.source(item.source)

    def sound(self, item: EditItem) -> Media | None:
        """The footage whose sound ``item`` plays instead of its own (``item.sound``), else None."""
        return None if item.sound is None else self.source(item.sound.source)

    def source(self, src: Source) -> Media:
        if isinstance(src, SceneSource | ClipSource):
            idx = self.index
            ir = Rate(idx.rate)
            if isinstance(src, SceneSource):
                sc = idx.scene(src.scene)
                clip, first, soft_end = idx.clip(sc.clip), sc.start_frame, sc.end_frame
            else:
                clip = idx.clip(src.clip)
                first, soft_end = 0, clip.frames
            span = lambda n: to_frame(Fraction(n) / ir, self.rate, "floor")  # noqa: E731
            master = self.project.root / clip.source
            has_master = master.is_file()
            mw, mh, _ = _master_size(clip)
            return Media(
                kind="video", path=_need(self.project.root / clip.proxy), start_s=Fraction(first) / ir,
                width=clip.width, height=clip.height,
                full_width=clip.src_width, full_height=clip.src_height,
                soft_frames=span(soft_end - first), hard_frames=span(clip.frames - first),
                audio=_need(self.project.root / clip.audio) if clip.audio else None,
                master=master if has_master else None, master_width=mw if has_master else 0,
                master_height=mh if has_master else 0, fps=Fraction(ir),
                master_probe=partial(self._master, clip, master, ir) if has_master else None,
                shot_at=_shot(clip, Fraction(first) / ir),
            )
        if isinstance(src, GenSource):
            return self._probe_video(project_path(self.project, src.gen))
        assert isinstance(src, ImageSource)
        path = project_path(self.project, src.image)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            w, h = im.size
        return Media("image", path, Fraction(0), w, h, w, h, None, None, None)

    def _master(self, clip: ClipInfo, path: Path, rate: Rate) -> MasterView:
        """The original's decode view; its matrix tag and frame rate come from a (cached) probe."""
        if path not in self._fps:
            v = _video_stream(probe_json(path), path)
            self._colour[path] = v.get("color_space")
            self._fps[path] = _rate_of(v)
        return master_view(clip, rate, self._colour[path], self._fps[path])

    def _generated(self, path: Path) -> bool:
        gen = (self.project.root / "gen").resolve()
        return gen in path.resolve().parents

    def _probe_video(self, path: Path) -> Media:
        if path not in self._probes:
            if not path.is_file():
                raise FileNotFoundError(path)
            info = probe_json(path)
            v = _video_stream(info, path)
            has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
            n = _plan_frames(v, info.get("format", {}), self.rate)
            w, h = int(v["width"]), int(v["height"])
            # untagged generated clips are BT.601 (PyAV/diffusers default); others by size
            tags = input_tags(v.get("color_space"), w, h, generated=self._generated(path))
            self._probes[path] = Media("video", path, Fraction(0), w, h, w, h, n, n,
                                       path if has_audio else None, prefilter=tags,
                                       fps=_rate_of(v))
        return self._probes[path]


FRAME_EPS = Fraction(1, 1000)  # of a frame: ffprobe's 6-decimal duration can fall just short


def _plan_frames(v: dict, fmt: dict, rate: Rate) -> int:
    """Plan-rate frames until the stream ends: its frame count when it already runs at the plan
    rate, else its exact duration (``duration_ts × time_base``, or the rounded seconds) × rate,
    floored with a tolerance so 0.733333 s at 30 fps is 22 frames, not 21."""
    nb = int(v.get("nb_frames") or 0)
    if nb and _rate_of(v) == rate:
        return nb
    if v.get("duration_ts") and v.get("time_base"):
        dur = int(v["duration_ts"]) * Fraction(v["time_base"])
    else:
        dur = Fraction(str(v.get("duration") or fmt.get("duration") or "0"))
    return math.floor(dur * rate + FRAME_EPS)


def _video_stream(info: dict, path: Path) -> dict:
    v = next((s for s in info["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise ValueError(f"{path} has no video stream")
    return v


ZOOM_OUT_FROM = 1.25  # a zoom_out transition: the incoming shot starts this much punched in
KEN_BURNS = 1.1


def zoom_keys(it: EditItem, kind: str) -> list[tuple[int, float, float, float]]:
    """Zoom keys ``(at, scale, cx, cy)`` the renderer applies to an item: its own ``zoom``; else
    for a ``zoom_out`` transition in an ease-out settle from ``ZOOM_OUT_FROM`` to 1 over the
    overlap; else a still's Ken Burns 1 → ``KEN_BURNS``; else none."""
    if it.zoom:
        return [(k.at, k.scale, k.cx, k.cy) for k in it.zoom]
    if it.transition_in.kind == "zoom_out" and not it.is_cut:
        d, z = it.transition_in.frames, ZOOM_OUT_FROM - 1
        keys: dict[int, float] = {}
        for p in (0, 1 / 3, 2 / 3, 1):
            keys.setdefault(round(p * d), 1 + z * (1 - p) ** 2)
        return [(a, sc, 0.5, 0.5) for a, sc in sorted(keys.items())]
    if kind == "image":
        return [(0, 1.0, 0.5, 0.5), (max(1, it.length - 1), KEN_BURNS, 0.5, 0.5)]
    return [(0, 1.0, 0.5, 0.5)]


def cover(box: tuple[float, float, float, float] | None, sw: int, sh: int,
          out_w: int, out_h: int) -> tuple[float, float, float, float]:
    """Crop box (fractions, None = whole frame) → pixel box ``(x, y, w, h)`` with the output
    aspect, centre-trimmed from the box ("cover")."""
    x, y, w, h = box or (0.0, 0.0, 1.0, 1.0)
    x, y, w, h = x * sw, y * sh, w * sw, h * sh
    aspect = out_w / out_h
    if w / h > aspect:
        nw = h * aspect
        x, w = x + (w - nw) / 2, nw
    else:
        nh = w / aspect
        y, h = y + (h - nh) / 2, nh
    return x, y, w, h
