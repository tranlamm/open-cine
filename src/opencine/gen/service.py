"""Batch generation API behind the CLI and MCP tools.

Every output is a variant ``gen/stills/<slot>~<seed>.png`` / ``gen/clips/<slot>~<seed>.mp4`` with a
``.json`` provenance sidecar; ``<slot>.png`` / ``<slot>.mp4`` hard-links the latest. Clips keep the
provider's file as ``*.raw.mp4`` and go through CPU QC (``gen.qc``) and post-process
(``gen.post``). Spend goes to the ledger and a manifest to ``gen/manifests/``; unchanged slots are
reused; quota exhaustion stops the batch (it never switches provider by itself).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError, model_validator

from opencine.core.cache import cache_key, sha256_file
from opencine.core.config import get_settings
from opencine.core.ffmpeg import probe_json
from opencine.core.ledger import BudgetExceeded, Ledger
from opencine.core.project import Project
from opencine.core.retention import link, remove, stale_raws
from opencine.gen import lint, takes
from opencine.gen import slots as slot_state
from opencine.gen.base import (
    AuthError,
    EditMode,
    GenProvider,
    GenResult,
    Kind,
    ProviderError,
    QuotaExhausted,
)
from opencine.gen.providers import get_provider, h3ref
from opencine.gen.source import VIDEO_FRAME, SourceWindow, cut_window, video_frame, window_key
from opencine.plan.schema import CropBox

SLOT_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
ASPECT = r"^[1-9][0-9]*:[1-9][0-9]*$"
STOP_ERRORS = (QuotaExhausted, AuthError, BudgetExceeded)
MAX_SEED = 2**31 - 1
PIPELINE_VERSION = 1  # service-side processing (crop, QC, post-process), part of the cache key
# slot fields added after outputs were first cached: a pre-fingerprint output is reused only
# while they keep their defaults (it was made without them)
NEW_FIELDS = {"crop", "variants", "canvas", "retime", "match_colour", "trim_head", "model",
              "colour_ref"}
# clip flags that a multi-shot prompt causes on purpose: a hard cut, the rewrite's extra
# [Shot N], the colour of the last shot differing from the first
MULTISHOT_OK = {"cut", "multishot", "colour_drift"}
_lock = threading.Lock()  # the refined-prompt store (parallel slots)


class ImageSlot(BaseModel):
    id: str = Field(pattern=SLOT_ID)
    # relative to the project root, or absolute; none = text-to-image (the prompt describes
    # the whole picture instead of an edit)
    refs: list[str] = []
    prompt: str = ""  # edit instruction, English; a new camera angle is written here too
    aspect: str = Field("9:16", pattern=ASPECT)
    seed: int | None = None
    # refs[0] only: the window to keep (fractions of the frame), e.g. the subject of a
    # landscape frame for a 9:16 still; the still is then centred inside it. None = centred
    crop: CropBox | None = None
    variants: int = Field(1, ge=1, le=4)  # stills to make: seeds seed, seed+1, ... (best-of-N)
    # hf model choice for this slot (hf_space.MODELS["image"]); None = the default
    model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_angle_mode(cls, data: Any) -> Any:
        # no separate angle mode: Qwen-Image-2.1 moves the camera from the prompt
        if isinstance(data, dict) and (data.get("mode", "edit") != "edit" or data.get("camera")):
            raise ValueError("mode='angle' and camera were removed: write the new camera angle "
                             "in the prompt")
        return data

    @model_validator(mode="after")
    def _has_prompt(self) -> ImageSlot:
        if not self.prompt.strip():
            raise ValueError("an image slot needs a prompt")
        if self.crop is not None and not self.refs:
            raise ValueError("crop cuts refs[0]: give a ref or drop crop")
        return self


class VideoSlot(BaseModel):
    id: str = Field(pattern=SLOT_ID)
    first: str | None = None  # None = text-to-video (H3 T2VA)
    # reference pictures the whole clip follows (hf Ref2VA: a character, a place …): <Picture N> in
    # the prompt, numbered after the first frame; none = T2V / I2V / first+last
    refs: list[str] = []
    prompt: str
    seconds: float = 5.0
    aspect: str = Field("9:16", pattern=ASPECT)
    last: str | None = None
    seed: int | None = None
    canvas: Literal["fast", "mid", "full"] | None = None  # hf quality tier; None = configured
    # to the project rate: conform = generated frames played at it (24->30 fps = 1.25x faster),
    # interp = motion-interpolated (slow CPU, keeps speed), none = the render repeats frames
    retime: Literal["conform", "interp", "none"] = "conform"
    match_colour: bool = True  # colour of the first frames matched to the first-frame input
    # … or first to this footage frame (the one the still was edited from; filled in from the
    # still's record when it has one): the still's own colour drift is corrected too
    colour_ref: str | None = None
    trim_head: int | None = Field(None, ge=0, le=24)  # frames cut at the start; None = QC decides
    model: str | None = None  # hf model choice (hf_space.MODELS["video"]); None = the default

    @model_validator(mode="after")
    def _last_needs_first(self) -> VideoSlot:
        if self.last and not self.first:
            raise ValueError("last needs first (first+last frame); for text-to-video drop both")
        if self.refs:
            if self.last:
                raise ValueError("a refs clip takes a first frame at most: drop last")
            if self.pictures > h3ref.MAX_PICTURES:
                raise ValueError(f"{self.pictures} pictures: a refs clip takes ≤ {h3ref.MAX_PICTURES} "
                                 "(the first frame counts; more run the conditioner out of GPU "
                                 "memory): keep the characters and the place that matter")
            if problems := h3ref.prompt_problems(self.prompt, self.pictures):
                raise ValueError(problems[0])
        return self

    @property
    def pictures(self) -> int:
        """``<Picture N>`` a refs clip gives: the first frame (Picture 1), then the refs in order."""
        return len(self.refs) + (1 if self.refs and self.first else 0)

    @property
    def shots(self) -> int:
        """Shots the prompt asks for (``[Shot N]``): more than one = cuts inside the clip on
        purpose (a multi-shot clip), so the cut checks do not flag them."""
        return max(map(int, _SHOT.findall(self.prompt)), default=1)


class EditSlot(BaseModel):
    """Video-to-video on a footage window: ``swap`` the person for the character in ``refs[0]``
    (a full-body picture of the new character), or ``restyle`` the window after ``refs[0]`` (its
    first frame, restyled with ``gen_image``). The clip keeps the window's length, motion and
    camera; it lands in ``gen/clips/<id>~<seed>.mp4`` like any clip."""

    id: str = Field(pattern=SLOT_ID)
    source: SourceWindow
    mode: EditMode
    refs: list[str] = Field(min_length=1)  # paths or frame refs (S03@2.5, gen/clips/x.mp4@0)
    prompt: str = ""  # English: the new character / look and what to keep
    subject: str = "person"  # swap: who to replace, English (found on the first frame)
    control: Literal["depth", "canny"] = "depth"  # restyle: the structure kept (canny: line art)
    aspect: str = Field("9:16", pattern=ASPECT)
    seed: int | None = None
    # to the project rate keeping real time: interp = motion-interpolated, none = the render
    # repeats frames. Never conform: the clip must stay in step with the footage's sound.
    retime: Literal["interp", "none"] = "interp"
    # colour of the first frames matched to the window's first frame; None = swap yes, restyle
    # no (a restyle changes the colours on purpose)
    match_colour: bool | None = None
    model: str | None = None  # hf model choice (hf_space.MODELS[mode]); None = the mode's default

    @model_validator(mode="before")
    @classmethod
    def _window(cls, data: Any) -> Any:
        """``source`` may be written ``"S03@1.2-6.2"`` (seconds into the scene)."""
        if isinstance(data, dict) and isinstance(data.get("source"), str):
            m = re.fullmatch(r"\s*([Ss]\d+)@(\d+(?:[.,]\d+)?)-(\d+(?:[.,]\d+)?)\s*", data["source"])
            if not m:
                raise ValueError("source: {scene, in_s, out_s} or 'S03@1.2-6.2'")
            data = {**data, "source": {"scene": m[1].upper(), "in_s": float(m[2].replace(",", ".")),
                                       "out_s": float(m[3].replace(",", "."))}}
        return data

    @property
    def colour(self) -> bool:
        return self.mode == "swap" if self.match_colour is None else self.match_colour


class ManifestItem(BaseModel):
    slot: str
    status: Literal["ok", "cached", "error", "skipped"]
    path: str | None = None  # relative to the project root
    provider: str
    model: str | None = None
    prompt: str
    seed: int | None = None
    quota_used: float = 0.0
    quota_unit: str | None = None
    usd_est: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    key: str | None = None
    fp: str | None = None  # provider fingerprint the output was made with
    flags: list[str] = []  # QC / prompt warnings (opencine.gen.qc), for the Director to look at
    frames: int | None = None  # clips: final length in frames at the project rate
    hints: list[str] = []  # prompt checks (opencine.gen.lint) and a take that changed several things
    changes: dict[str, Any] | None = None  # what this take changed from the slot's last one
    extra: dict[str, Any] = {}


class Manifest(BaseModel):
    project: str
    kind: Kind
    provider: str
    created: str
    items: list[ManifestItem] = []
    stopped: bool = False
    message: str | None = None
    reset_at: datetime | None = None
    warm: dict[str, Any] = {}
    path: str | None = None

    @property
    def ok(self) -> bool:
        return not self.stopped and all(i.status in ("ok", "cached") for i in self.items)


def select_provider(kind: Kind, provider: str | None = None) -> str:
    s = get_settings()
    return provider or {"image": s.gen_image, "video": s.gen_video, "edit": s.gen_edit}[kind]


def _store_kind(kind: Kind) -> Kind:
    """Where a kind's outputs live and are approved: an edit, an enhanced or a face-refined clip
    is a clip."""
    return "video" if kind in ("edit", "enhance", "face") else kind


_FRAME_REF = re.compile(r"^[SC]\d+(@[\d:.,]+)?$", re.IGNORECASE)
_PROXY_FRAME = re.compile(r"^(C\d+)_t(\d{7})\.jpg$")  # index/frames, index/sheets


_PANEL = re.compile(r"^(?P<path>.+?)#(?P<which>face|body|panel=[1-9])$")
SHEET_PANELS = 5  # a close-up, then front, three-quarter, profile, back
PANEL_VERSION = 3  # part of the panel cache key: bump when the cut changes


def _resolve(project: Project, p: str) -> Path:
    """An input path; a frame ref (``S03@2.5``, ``C01@12.4``) or a proxy frame of the
    index becomes the source frame at full resolution (generation needs the detail);
    ``<sheet>#face`` / ``#body`` / ``#panel=N`` a panel of a character sheet (``_panel_ref``)."""
    from opencine.index.sheets import source_frame

    if m := _PANEL.match(p.strip()):
        which = m["which"]
        n = {"face": 1, "body": 2}.get(which) or int(which.split("=")[1])
        return _panel_ref(project, _resolve(project, m["path"]), n)
    if _FRAME_REF.match(p.strip()):
        return source_frame(project, p.strip())
    if VIDEO_FRAME.match(p.strip()):  # a frame of a clip: gen/clips/7~12.mp4@0 / @last
        return video_frame(project, p.strip())
    path = Path(p) if Path(p).is_absolute() else project.root / p
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {p}")
    m = _PROXY_FRAME.match(path.name)
    if m and path.parent.resolve() in {project.frames_dir.resolve(), project.sheets_dir.resolve()}:
        return source_frame(project, f"{m[1]}@{int(m[2]) / 1000:.3f}")
    return path


def _rel(project: Project, p: Path) -> str:
    try:
        return p.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return str(p)


def _crop_ref(project: Project, path: Path, box: CropBox) -> Path:
    """``path`` cut to ``box`` (fractions), cached as ``gen/refs/crops/<stem>_<key>.png``."""
    key = cache_key(sha256_file(path), [box.x, box.y, box.w, box.h])
    out = project.root / "gen" / "refs" / "crops" / f"{path.stem}_{key[:8]}.png"
    if not out.exists():
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            x0, y0 = round(box.x * w), round(box.y * h)
            x1, y1 = min(w, round((box.x + box.w) * w)), min(h, round((box.y + box.h) * h))
            if x1 - x0 < 16 or y1 - y0 < 16:
                raise ValueError(f"crop {box.model_dump()} leaves {x1 - x0}x{y1 - y0} px")
            out.parent.mkdir(parents=True, exist_ok=True)
            im.crop((x0, y0, x1, y1)).save(out)
    return out


def _figures(rgb: np.ndarray, panels: int = SHEET_PANELS) -> list[tuple[int, int, int, int]]:
    """Boxes of the figures standing out from a sheet's plain ground (the median of the image's edge
    pixels), left to right: blobs of ≥ 1 % of the picture after a closing, those sharing most of their
    columns merged (a head apart from its body). Fewer than ``panels``: the widest blob is cut at its
    emptiest column when that is a clear dip (two figures touching, a close-up box against an arm)."""
    import cv2

    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.int16)
    mask = (np.abs(rgb.astype(np.int16) - np.median(edge, axis=0)).max(axis=2) > 24).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = sorted([int(x), int(y), int(x + bw), int(y + bh)]
                   for x, y, bw, bh, area in stats[1:count] if area >= 0.01 * h * w)
    merged: list[list[int]] = []
    for b in boxes:
        if merged and min(merged[-1][2], b[2]) - b[0] > 0.5 * min(b[2] - b[0], merged[-1][2] - merged[-1][0]):
            m = merged[-1]
            merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
        else:
            merged.append(b)
    for _ in range(panels):
        if not 0 < len(merged) < panels:
            break
        i = max(range(len(merged)), key=lambda k: merged[k][2] - merged[k][0])
        x0, y0, x1, y1 = merged[i]
        occ = mask[y0:y1, x0:x1].sum(axis=0)
        lo, hi = int(0.15 * len(occ)), int(0.85 * len(occ))
        if hi <= lo:
            break
        cut = lo + int(np.argmin(occ[lo:hi]))
        if occ[cut] > 0.35 * np.median(occ):
            break  # no clear gap: one wide figure
        parts = []
        for a, b in ((x0, x0 + cut), (x0 + cut, x1)):
            part = mask[y0:y1, a:b]
            rows, cols = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
            parts.append([a + int(cols[0]), y0 + int(rows[0]), a + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1])
        merged[i:i + 1] = parts
    return [tuple(b) for b in merged]


def sheet_panel(rgb: np.ndarray, n: int, panels: int = SHEET_PANELS) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` of panel ``n`` (1 = left) of a sheet of ``panels`` figures on a plain ground:
    the ``n``-th figure found (``_figures``) with a 4 % margin that stops at its neighbours (at the middle
    of their overlap when two figures share columns: a close-up's shoulder above a hand); when the
    sheet does not split into ``panels`` figures, its ``n``-th fifth tightened to what stands out."""
    h, w = rgb.shape[:2]
    if not 1 <= n <= panels:
        raise ValueError(f"panel {n}: a sheet has panels 1-{panels}")
    figs = _figures(rgb, panels)
    if len(figs) == panels:
        def edge(i: int, side: int) -> int:  # figure i's limit toward figure i + 1 (side 2) or back (0)
            a, b = figs[i][2], figs[i + 1][0]
            return (a + b) // 2 if a > b else (a if side == 0 else b)

        x0, y0, x1, y1 = figs[n - 1]
        px, py = round(0.04 * (x1 - x0)), round(0.04 * (y1 - y0))
        left = edge(n - 2, 0) if n > 1 else 0
        right = edge(n - 1, 2) if n < panels else w
        return max(left, x0 - px), max(0, y0 - py), min(right, x1 + px), min(h, y1 + py)
    c0, c1 = round((n - 1) * w / panels), round(n * w / panels)
    col = rgb[:, c0:c1].astype(np.int16)
    edge = np.concatenate([col[0], col[-1], col[:, 0], col[:, -1]])
    ground = np.median(edge, axis=0)
    mask = np.abs(col - ground).max(axis=2) > 24
    rows = np.flatnonzero(mask.sum(axis=1) > max(1, 0.005 * (c1 - c0)))
    cols = np.flatnonzero(mask.sum(axis=0) > max(1, 0.005 * h))
    if not len(rows) or not len(cols):
        return c0, 0, c1, h
    px, py = round(0.04 * (c1 - c0)), round(0.04 * h)
    return (int(max(c0, c0 + cols[0] - px)), int(max(0, rows[0] - py)),
            int(min(c1, c0 + cols[-1] + 1 + px)), int(min(h, rows[-1] + 1 + py)))


def _panel_ref(project: Project, path: Path, n: int) -> Path:
    """Panel ``n`` of the character sheet ``path`` (``#face`` = 1, the close-up; ``#body`` = 2, the
    front view), cut on the CPU and cached as ``gen/refs/crops/<stem>_p<n>_<key>.png``."""
    key = cache_key(sha256_file(path), "panel", n, SHEET_PANELS, PANEL_VERSION)
    out = project.root / "gen" / "refs" / "crops" / f"{path.stem}_p{n}_{key[:8]}.png"
    if not out.exists():
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            box = sheet_panel(np.asarray(im), n)
            out.parent.mkdir(parents=True, exist_ok=True)
            im.crop(box).save(out)
    return out


def _inputs(project: Project, slot: ImageSlot | VideoSlot | EditSlot,
            cut: bool = True) -> list[Path]:
    """Input files of a slot; an edit's footage window first (``cut=False``: not cut yet, a
    placeholder, for a cache lookup that only needs its key)."""
    if isinstance(slot, ImageSlot):
        inputs = [_resolve(project, r) for r in slot.refs]
        if slot.crop is not None:
            inputs[0] = _crop_ref(project, inputs[0], slot.crop)
        return inputs
    if isinstance(slot, EditSlot):
        window = cut_window(project, slot.source, slot.aspect) if cut else Path(slot.source.label())
        return [window, *(_resolve(project, r) for r in slot.refs)]
    return [_resolve(project, p) for p in (slot.first, slot.last, *slot.refs) if p]


def _with_colour_ref(project: Project, slot: VideoSlot) -> VideoSlot:
    """``colour_ref`` from the first-frame still's record (its first footage frame ref)."""
    if slot.colour_ref or not slot.match_colour or not slot.first:
        return slot
    side = Path(slot.first) if Path(slot.first).is_absolute() else project.root / slot.first
    try:
        refs = json.loads(side.with_suffix(".json").read_text("utf-8")).get("extra", {}).get("refs") or []
    except (OSError, ValueError):
        return slot
    if refs and _FRAME_REF.match(str(refs[0]).strip()):
        return slot.model_copy(update={"colour_ref": str(refs[0]).strip()})
    return slot


def _keys(project: Project, prov: GenProvider, kind: Kind,
          slot: ImageSlot | VideoSlot | EditSlot, inputs: list[Path]) -> tuple[str, str | None, str]:
    """``(cache key, legacy key or None, fingerprint id)`` of a slot. An edit's footage window
    counts by what it shows (``window_key``), not by its encoded bytes."""
    if isinstance(slot, EditSlot):
        shas = [window_key(project, slot.source, slot.aspect), *map(sha256_file, inputs[1:])]
        fp = prov.fingerprint(kind, slot.model, mode=slot.mode)
        from opencine.gen.post import POST_VERSION

        pipe = {"v": PIPELINE_VERSION, "post": POST_VERSION, "rate": str(project.rate)}
        key = cache_key(prov.name, kind, slot.model_dump(exclude={"id"}), shas, fp, pipe)
        return key, None, cache_key(fp)
    shas = [sha256_file(p) for p in inputs]
    fp = prov.fingerprint(kind, slot.model) if slot.model else prov.fingerprint(kind)
    pipe: dict[str, Any] = {"v": PIPELINE_VERSION}
    if kind == "video":
        from opencine.gen.post import POST_VERSION

        pipe.update(post=POST_VERSION, rate=str(project.rate))
    # no model choice: leave model out of the key so older cached outputs stay valid
    dump = slot.model_dump(exclude={"id"} if slot.model else {"id", "model"})
    key = cache_key(prov.name, kind, dump, shas, fp, pipe)
    fields = type(slot).model_fields
    untouched = all(getattr(slot, f) == fields[f].default for f in NEW_FIELDS if f in fields)
    legacy = (cache_key(prov.name, kind, slot.model_dump(exclude={"id", *NEW_FIELDS}), shas)
              if untouched else None)
    return key, legacy, cache_key(fp)


def _dir_ext(project: Project, kind: Kind) -> tuple[Path, str]:
    return (project.stills_dir, ".png") if kind == "image" else (project.clips_dir, ".mp4")


def _find_cached(project: Project, kind: Kind, slot_id: str, key: str, legacy: str | None,
                 model: str | None) -> tuple[dict[str, Any] | None, bool]:
    """``(sidecar, is_legacy)`` of a previous output of this exact request: a variant sidecar
    with ``key`` whose file still exists, else a pre-variant ``<slot>.json`` made before
    fingerprints (``legacy`` key) by the same ``model`` (another model's output is never passed
    off as this one's)."""
    d, ext = _dir_ext(project, kind)
    for side in sorted(d.glob(f"{slot_id}~*.json")):
        prev = _read_json(side)
        if prev.get("key") == key and prev.get("path") and (project.root / prev["path"]).exists():
            return prev, False
    side = d / f"{slot_id}.json"
    if legacy and model and side.exists():
        prev = _read_json(side)
        if (prev.get("fp") is None and prev.get("key") == legacy and prev.get("model") == model
                and (d / f"{slot_id}{ext}").exists()):
            return prev, True
    return None, False


def _read_json(path: Path) -> dict[str, Any]:
    """A JSON object from ``path``; ``{}`` when it is missing or unreadable (a torn write
    must not take a batch down)."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _model(prov: GenProvider, kind: Kind) -> str | None:
    """The default model's name, for outputs made before fingerprints (none for edits)."""
    return None if kind == "edit" else prov.fingerprint(kind).get("model")


def _is_cached(project: Project, prov: GenProvider, kind: Kind,
               slot: ImageSlot | VideoSlot) -> bool:
    try:
        inputs = _inputs(project, slot, cut=False)
        key, legacy, _ = _keys(project, prov, kind, slot, inputs)
        return _find_cached(project, kind, slot.id, key, legacy, _model(prov, kind))[0] is not None
    except Exception:  # missing input etc.: _one reports it
        return False


def _alias(project: Project, kind: Kind, slot_id: str, path: str) -> None:
    """``<slot>.png|mp4`` + ``<slot>.json``: a hard link to a variant and a copy of its sidecar
    (the batch calls it in slot order, so the alias is the last variant asked for)."""
    d, ext = _dir_ext(project, kind)
    src = project.root / path
    if not src.exists() or src.name == f"{slot_id}{ext}":  # legacy: it is the alias already
        return
    link(src, d / f"{slot_id}{ext}")
    if src.with_suffix(".json").exists():
        shutil.copyfile(src.with_suffix(".json"), d / f"{slot_id}.json")


# ---- H3 prompt rewrites: deterministic (greedy decode), so one per prompt is enough -----
_SHOT = re.compile(r"\[Shot (\d+)\]")


def _one_shot(text: str) -> bool:
    """No ``[Shot 2]``…: a rewrite with a cut must not be reused (every re-roll would get
    the same hard cut); the next call goes back to the rewriter or to a one-shot IR prompt."""
    return max(map(int, _SHOT.findall(text)), default=1) <= 1


def _refined_store(project: Project) -> Path:
    return project.root / "gen" / "refined.json"


def _refined_key(prov: GenProvider, slot: VideoSlot, inputs: list[Path]) -> str:
    return cache_key(prov.name, "refined", slot.prompt, slot.seconds, slot.last is not None,
                     [sha256_file(p) for p in inputs])


def _refined_lookup(project: Project, key: str) -> str | None:
    with _lock:
        text = (_read_json(_refined_store(project)).get(key) or {}).get("refined")
    return text if isinstance(text, str) and _one_shot(text) else None


def _refined_save(project: Project, key: str, prompt: str, refined: str) -> None:
    """Remember a one-shot rewrite (atomic write; best effort, never raises)."""
    if not _one_shot(refined):
        return
    path = _refined_store(project)
    with _lock:
        data = _read_json(path)
        data[key] = {"prompt": prompt, "refined": refined,
                     "ts": datetime.now().isoformat(timespec="seconds")}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
            os.replace(tmp, path)
        except OSError:
            pass


def _finish_clip(project: Project, slot: VideoSlot, inputs: list[Path], raw: Path,
                 out: Path) -> tuple[dict[str, Any], list[str], int | None]:
    """Check ``raw`` and post-process it into ``out`` → (extra, flags, frames at the project
    rate). Never raises for a check or a post-process failure: the clip is paid for. A
    text-to-video clip (no inputs) has no first frame to check or match colour to."""
    from opencine.gen.post import exact_duration, frames_at, post_clip
    from opencine.gen.qc import qc_clip

    flags: list[str] = []
    first = inputs[0] if inputs and slot.first else None
    try:
        qc = qc_clip(raw, first)
    except Exception as e:
        qc = {"flags": ["qc_failed"], "error": f"{type(e).__name__}: {e}", "trim_head": 0}
    trim = slot.trim_head if slot.trim_head is not None else int(qc.get("trim_head") or 0)
    ref = None
    if slot.match_colour and slot.colour_ref:
        try:
            ref = _resolve(project, slot.colour_ref)
        except (FileNotFoundError, KeyError, ValueError):
            ref = None
    try:
        post = post_clip(raw, out, rate=project.rate, retime=slot.retime, trim_head=trim,
                         still=first if slot.match_colour else None, ref=ref)
        frames = post.get("frames")
    except Exception as e:
        shutil.copyfile(raw, out)
        post, flags = {"error": f"{type(e).__name__}: {e}"}, ["post_failed"]
        try:
            info = probe_json(out)
            v = next(s for s in info["streams"] if s.get("codec_type") == "video")
            frames = frames_at(exact_duration(v, info["format"].get("duration")), project.rate)
        except Exception:
            frames = None
    return ({"qc": qc, "post": post, "raw": _rel(project, raw)},
            [*flags, *qc.get("flags", [])], frames)


def _finish_edit(project: Project, slot: EditSlot, inputs: list[Path], raw: Path, out: Path,
                 covers_s: float | None = None, coloured: bool = False
                 ) -> tuple[dict[str, Any], list[str], int | None]:
    """``_finish_clip`` for a video-to-video clip: no head trim and no first-frame check (the
    clip must start on the window's first frame, and a swap changes it on purpose), its frames
    spread over the footage they cover (``covers_s``, the provider's; else the whole window) at
    the project rate, colour pulled to the window's first frame for a swap unless the provider
    already matched the whole window (``coloured``). ``length`` flags a clip that does not cover
    the window (the model's frame limit)."""
    from opencine.gen.post import exact_duration, frames_at, post_clip
    from opencine.gen.qc import qc_clip

    flags: list[str] = []
    try:
        qc = qc_clip(raw, None)
    except Exception as e:
        qc = {"flags": ["qc_failed"], "error": f"{type(e).__name__}: {e}"}
    qc["flags"] = [f for f in qc.get("flags", []) if f != "flash_head"]
    first = None
    if slot.colour and not coloured:
        try:
            from opencine.gen.source import video_frame

            first = video_frame(project, f"{inputs[0].as_posix()}@0")
        except Exception:  # noqa: BLE001 — colour match is best effort
            first = None
    try:
        span = Fraction(str(covers_s or slot.source.seconds))
        post = post_clip(raw, out, rate=project.rate, retime=slot.retime, trim_head=0, still=first,
                         span=span)
        frames = post.get("frames")
    except Exception as e:
        shutil.copyfile(raw, out)
        post, flags = {"error": f"{type(e).__name__}: {e}"}, ["post_failed"]
        try:
            info = probe_json(out)
            v = next(s for s in info["streams"] if s.get("codec_type") == "video")
            frames = frames_at(exact_duration(v, info["format"].get("duration")), project.rate)
        except Exception:
            frames = None
    want = round(slot.source.seconds * float(project.rate))
    if frames is not None and frames < want - 1:
        flags.append("length")
    source = slot.source.model_dump(mode="json", exclude_none=True, exclude={"reframe"})
    return ({"qc": qc, "post": post, "raw": _rel(project, raw), "mode": slot.mode,
             "source": source, "window": _rel(project, inputs[0]), "window_frames": want},
            [*flags, *qc.get("flags", [])], frames)


def _reprocess_legacy(project: Project, slot: VideoSlot, inputs: list[Path],
                      prev: dict[str, Any], key: str, fp: str) -> ManifestItem:
    """A clip made before post-processing, by the same model: keep it as the raw of a new
    variant and post-process it (CPU only, no quota) instead of reusing it as it was."""
    d, ext = _dir_ext(project, "video")
    seed = prev.get("seed") if prev.get("seed") is not None else int(key[:8], 16) % MAX_SEED
    out = d / f"{slot.id}~{seed}{ext}"
    raw = out.with_name(f"{out.stem}.raw.mp4")
    shutil.copyfile(d / f"{slot.id}{ext}", raw)
    extra, flags, frames = _finish_clip(project, slot, inputs, raw, out)
    item = ManifestItem(**{**prev, "status": "ok", "path": _rel(project, out), "seed": seed,
                           "key": key, "fp": fp, "flags": list(dict.fromkeys(flags)),
                           "frames": frames,
                           "extra": {**(prev.get("extra") or {}), **extra,
                                     "reprocessed": "legacy clip"}})
    out.with_suffix(".json").write_text(item.model_dump_json(indent=2), "utf-8")
    return item.model_copy(update={"status": "cached", "quota_used": 0.0, "usd_est": 0.0,
                                   "latency_s": 0.0})


def _notes(project: Project) -> str | None:
    """The film bible (``Brief.notes``): the text the prompts copy their descriptions from."""
    from opencine.plan.store import load_plan, versions

    try:
        return load_plan(project, "brief")[1].notes if versions(project, "brief") else None
    except (FileNotFoundError, ValueError):
        return None


def _lint(project: Project, slot: ImageSlot | VideoSlot | EditSlot) -> list[str]:
    """Prompt hints for a slot (never raises: a check must not cost a paid call)."""
    try:
        if isinstance(slot, VideoSlot):
            return lint.video_hints(slot.prompt, slot.seconds, refs=len(slot.refs),
                                    first=bool(slot.first), notes=_notes(project))
        if isinstance(slot, ImageSlot):
            return lint.image_hints(slot.prompt, len(slot.refs), _notes(project))
    except Exception:  # noqa: BLE001 — hints are best effort
        pass
    return []


def _take(project: Project, kind: Kind, slot: ImageSlot | VideoSlot | EditSlot) -> dict[str, Any]:
    """What this request changes from the slot's latest take (its ``<slot>.json``)."""
    d, _ = _dir_ext(project, kind)
    made = (slot_state.load(project)[_store_kind(kind)].get(slot.id) or {}).get("variants", [])
    return takes.compare(_read_json(d / f"{slot.id}.json"), takes.request_of(slot), slot.prompt,
                         len(made) + 1)


def _with_defaults(project: Project, prov: GenProvider, kind: Kind, slots: list) -> list:
    """Slots of a film without an aspect take the film's; clips with refs and no model the
    provider's Ref2VA model."""
    if kind in ("image", "video") and project.film:
        aspect = film_aspect(project)
        slots = [s if "aspect" in s.model_fields_set else s.model_copy(update={"aspect": aspect})
                 for s in slots]
    if kind == "video" and prov.ref_model:
        slots = [s.model_copy(update={"model": prov.ref_model}) if s.refs and not s.model else s
                 for s in slots]
    return slots


def _one(project: Project, prov: GenProvider, kind: Kind,
         slot: ImageSlot | VideoSlot | EditSlot) -> tuple[ManifestItem, ProviderError | None]:
    """Run one slot. Never raises: any failure becomes an ``error`` item — one that still
    carries the spend when the provider had already returned (the call was paid for)."""
    base: dict[str, Any] = {"slot": slot.id, "provider": prov.name, "prompt": slot.prompt,
                            "hints": _lint(project, slot)}
    res: GenResult | None = None
    try:
        if isinstance(slot, VideoSlot):
            slot = _with_colour_ref(project, slot)
        edit = isinstance(slot, EditSlot)
        inputs = _inputs(project, slot, cut=not edit)  # a window is cut only to run it
        key, legacy, fp = _keys(project, prov, kind, slot, inputs)
        prev, is_legacy = _find_cached(project, kind, slot.id, key, legacy, _model(prov, kind))
        if prev and is_legacy and isinstance(slot, VideoSlot):
            return _reprocess_legacy(project, slot, inputs, prev, key, fp), None
        if prev:
            return ManifestItem(**{**prev, "status": "cached", "quota_used": 0.0,
                                   "usd_est": 0.0, "latency_s": 0.0}), None
        # a seedless request gets a seed derived from it: a rerun after a crash is the same
        # call (same file name, same remote job), and the seed is on record either way
        seed = slot.seed if slot.seed is not None else int(key[:8], 16) % MAX_SEED
        d, ext = _dir_ext(project, kind)
        out = d / f"{slot.id}~{seed}{ext}"
        changes = _take(project, kind, slot)
        if note := takes.hint(changes):
            base["hints"] = [*base["hints"], note]
        if isinstance(slot, EditSlot):
            inputs = _inputs(project, slot)
            raw = out.with_name(f"{out.stem}.raw.mp4")
            res = prov.edit_video(inputs[0], inputs[1:], slot.prompt, slot.mode, slot.aspect,
                                  out=raw, seed=seed, model=slot.model, subject=slot.subject,
                                  control=slot.control)
            extra = dict(res.extra)
            flags = list(extra.pop("flags", []))
            extra["refs"] = list(slot.refs)
            post_extra, post_flags, frames = _finish_edit(project, slot, inputs, raw, out,
                                                          extra.get("covers_s"),
                                                          coloured="colour" in extra)
            extra.update(post_extra)
            flags += post_flags
        elif isinstance(slot, ImageSlot):
            res = prov.edit_image(inputs, slot.prompt, slot.aspect, out=out, seed=seed,
                                  **({"model": slot.model} if slot.model else {}))
            extra, frames = dict(res.extra), None
            extra["refs"] = list(slot.refs)  # what it was made from (a clip's colour target)
            flags = list(extra.pop("flags", []))
            try:
                from opencine.gen.qc import qc_still

                extra["qc"] = qc_still(res.path, inputs[0] if inputs else None)
                flags += extra["qc"]["flags"]
            except Exception as e:
                extra["qc"], flags = {"error": f"{type(e).__name__}: {e}"}, [*flags, "qc_failed"]
        else:
            raw = out.with_name(f"{out.stem}.raw.mp4")
            rkey = _refined_key(prov, slot, inputs)
            refined = _refined_lookup(project, rkey)
            kw: dict[str, Any] = {"canvas": slot.canvas} if slot.canvas else {}
            if slot.model:
                kw["model"] = slot.model
            keys = sum(1 for p in (slot.first, slot.last) if p)
            if slot.refs:
                kw["refs"] = inputs[keys:]
            res = prov.video(inputs[0] if slot.first else None, refined or slot.prompt,
                             slot.seconds, slot.aspect, inputs[keys - 1] if slot.last else None,
                             out=raw, seed=seed, **kw)
            extra = dict(res.extra)
            flags = list(extra.pop("flags", []))
            if refined:
                extra["prompt_reused"] = "refined"  # this prompt's earlier rewrite, as is
            elif text := extra.get("refined_prompt"):
                _refined_save(project, rkey, slot.prompt, text)
            post_extra, post_flags, frames = _finish_clip(project, slot, inputs, raw, out)
            extra.update(post_extra)
            flags += post_flags
            if slot.shots > 1:  # cuts the prompt asked for: where they landed is qc cuts_s
                flags = [f for f in flags if f not in MULTISHOT_OK]
                extra["shots"] = slot.shots
            if slot.refs:
                extra["refs"] = list(slot.refs)
        extra["request"] = takes.request_of(slot)
        item = ManifestItem(**base, status="ok", path=_rel(project, out), model=res.model,
                            seed=res.seed if res.seed is not None else seed,
                            quota_used=res.quota_used, quota_unit=res.quota_unit,
                            usd_est=res.usd_est, latency_s=res.latency_s, key=key, fp=fp,
                            flags=list(dict.fromkeys(flags)), frames=frames, changes=changes,
                            extra=extra)
        dump = item.model_dump(mode="json")
        out.with_suffix(".json").write_text(json.dumps(dump, indent=2, ensure_ascii=False),
                                            "utf-8")
        return item, None
    except Exception as e:  # a batch always ends with a manifest
        err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
        if res is not None and not (err.quota_used or err.usd_est):
            err.spent(res.quota_used, res.quota_unit, res.usd_est)  # paid before it failed
        return ManifestItem(**base, status="error", error=f"{type(e).__name__}: {e}",
                            model=res.model if res else None,
                            quota_used=err.quota_used, quota_unit=err.quota_unit,
                            usd_est=err.usd_est, seed=slot.seed), err


def _expand(slots: list) -> list:
    """Slots with ``variants`` > 1 become one run per seed (``seed``, ``seed`` + 1, ...; a
    seedless slot starts from a seed derived from its content, so reruns hit the cache)."""
    runs = []
    for s in slots:
        n = getattr(s, "variants", 1)
        if n <= 1:
            runs.append(s)
            continue
        start = s.seed if s.seed is not None else int(cache_key(
            "variants", s.model_dump(exclude={"id", "seed", "variants"}))[:8], 16) % MAX_SEED
        runs += [s.model_copy(update={"seed": (start + i) % MAX_SEED, "variants": 1})
                 for i in range(n)]
    return runs


def _stop_message(name: str, err: Exception, done: int, left: int) -> str:
    if isinstance(err, QuotaExhausted):
        when = (f"resets at {err.reset_at:%Y-%m-%d %H:%M}" if err.reset_at
                else "reset time unknown (top up / check the provider)")
        return (f"{name} quota exhausted, {when}. {done} slot(s) done, {left} not run. "
                f"Wait and rerun the same command (finished slots are reused), or rerun "
                f"with another provider once approved. Detail: {err}")
    return f"{name}: batch stopped ({type(err).__name__}): {err}. {done} done, {left} not run."


def _record(ledger: Ledger, name: str, kind: Kind, item: ManifestItem) -> None:
    """Ledger every slot that spent something: ok items, and failed ones that still cost."""
    if item.status == "ok" or (item.status == "error" and (item.quota_used or item.usd_est)):
        ledger.record(name, f"gen_{kind}", item.quota_used, usd_est=item.usd_est,
                      quota_note=f"{item.quota_used:g} {item.quota_unit}",
                      slot=item.slot, model=item.model, path=item.path, status=item.status)


def film_aspect(project: Project) -> str:
    """The frame of a film's stills and clips when a slot names none: its ShotPlan's deliverable
    (template film: 1920x1080), else 16:9."""
    from opencine.plan.schema import DELIVERABLES
    from opencine.plan.store import load_plan
    from opencine.plan.templates import template

    try:
        _, sp = load_plan(project, "shotplan")
    except FileNotFoundError:
        return "16:9"
    w, h = DELIVERABLES[sp.deliverable or template(sp.template).deliverable]
    return "16:9" if w >= h else "9:16"


def _run_batch(project: Project, kind: Kind, slots: list, provider: str | None) -> Manifest:
    ids = [s.id for s in slots]
    if dup := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate slot id(s) in one batch: {', '.join(dup)}")
    name = select_provider(kind, provider)
    prov = get_provider(name, project)
    slots = _with_defaults(project, prov, kind, slots)
    ledger = Ledger(project)
    man = Manifest(project=project.pid, kind=kind, provider=name,
                   created=datetime.now().isoformat(timespec="seconds"))

    def stop(err: Exception, left: int) -> None:
        done = sum(i.status in ("ok", "cached") for i in man.items)
        man.stopped, man.message = True, _stop_message(name, err, done, left)
        man.reset_at = getattr(err, "reset_at", None)

    runs = _expand(slots)
    pending = list(range(len(runs)))
    try:
        todo = {i for i in pending if not _is_cached(project, prov, kind, runs[i])}
        # past its re-roll limit: refuse the slot's new runs (its cached variants still come back)
        if refuse := slot_state.refused(project, _store_kind(kind), {runs[i].id for i in todo}):
            for i in sorted(i for i in todo if runs[i].id in refuse):
                man.items.append(ManifestItem(slot=runs[i].id, status="error", provider=name,
                                              prompt=runs[i].prompt, seed=runs[i].seed,
                                              error=f"RerollLimit: {refuse[runs[i].id]}"))
            pending = [i for i in pending if not (i in todo and runs[i].id in refuse)]
            todo = {i for i in todo if runs[i].id not in refuse}
        if todo:  # wake remote backends once, only when something will really run
            man.warm = prov.prepare(kind, [runs[i] for i in sorted(todo)])
        while pending:
            n = max(prov.max_parallel, 1)
            chunk = pending[:n]
            ledger.check_cap(sum(prov.estimate_usd(kind, getattr(runs[i], "seconds", 0))
                                 for i in chunk if i in todo))
            if len(chunk) == 1:
                results = [_one(project, prov, kind, runs[chunk[0]])]
            else:
                with ThreadPoolExecutor(len(chunk)) as ex:
                    results = list(ex.map(lambda i: _one(project, prov, kind, runs[i]), chunk))
            pending = pending[n:]
            first_stop: Exception | None = None
            for item, err in results:
                man.items.append(item)
                _record(ledger, name, kind, item)
                if item.status in ("ok", "cached") and item.path:
                    _alias(project, kind, item.slot, item.path)
                if isinstance(err, STOP_ERRORS) and first_stop is None:
                    first_stop = err
            if first_stop is not None:
                stop(first_stop, len(pending))
                break
    except Exception as e:  # budget, warm-up, anything else: stop and report in the manifest
        stop(e, len(pending))
    ran: dict[str, list[str]] = {}
    for item in man.items:  # one more batch for every slot that really ran
        if item.status == "ok" and item.path:
            ran.setdefault(item.slot, []).append(item.path)
    slot_state.record_runs(project, _store_kind(kind), ran)
    remove(stale_raws(project))
    man.items += [ManifestItem(slot=runs[i].id, status="skipped", provider=name,
                               prompt=runs[i].prompt, seed=runs[i].seed) for i in pending]
    out = project.root / "gen" / "manifests" / f"{kind}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    man.path = _rel(project, out)
    out.write_text(man.model_dump_json(indent=2), "utf-8")
    return man


def gen_images(project: Project, slots: list[ImageSlot], provider: str | None = None) -> Manifest:
    """Storyboard stills (image edit from real frames, or text-to-image without refs) for a
    batch of slots."""
    return _run_batch(project, "image", slots, provider)


def gen_videos(project: Project, slots: list[VideoSlot], provider: str | None = None) -> Manifest:
    """T2V / I2V / first-last-frame clips for a batch of slots."""
    return _run_batch(project, "video", slots, provider)


def gen_edits(project: Project, slots: list[EditSlot], provider: str | None = None) -> Manifest:
    """Video-to-video clips (character swap, restyle) of footage windows, for a batch of slots."""
    return _run_batch(project, "edit", slots, provider)


def check_slots(project: Project, kind: Literal["image", "video"], slots: list[dict[str, Any]],
                provider: str | None = None) -> dict[str, Any]:
    """A batch looked at, not run (``gen_check``, free): per slot its prompt hints, what it changes
    from the slot's last take, whether it would come from the cache, and (hf) the quota its call
    reserves; ``quota``: whether the calls that would run fit the daily quota left, placed on the
    accounts as the batch would place them (``HfSpaceProvider.fit``). A slot that does not
    validate gets its ``error``."""
    name = select_provider(kind, provider)
    prov = get_provider(name, project)
    cls = ImageSlot if kind == "image" else VideoSlot
    items: list[dict[str, Any]] = []
    calls: list[tuple[str, str, float]] = []
    for raw in slots:
        sid = raw.get("id") if isinstance(raw, dict) else None
        try:
            slot = _with_defaults(project, prov, kind, [cls.model_validate(raw)])[0]
        except ValidationError as e:
            items.append({"id": sid, "error": "; ".join(err["msg"] for err in e.errors())})
            continue
        changes = _take(project, kind, slot)
        item: dict[str, Any] = {"id": slot.id, "hints": [*_lint(project, slot),
                                                         *filter(None, [takes.hint(changes)])],
                                "changes": changes, "cached": _is_cached(project, prov, kind, slot)}
        if name == "hf":
            from opencine.gen.providers.hf_space import slot_call

            try:
                role, need = slot_call(kind, slot, getattr(prov, "s", None))
                item["reserve_s"] = round(need)
                if not item["cached"]:
                    calls += [(slot.id, role, need)] * getattr(slot, "variants", 1)
            except ProviderError as e:
                item["error"] = str(e)
        items.append(item)
    out = {"provider": name, "items": items,
           "reserve_s": sum(i.get("reserve_s", 0) for i in items if not i.get("cached"))}
    if calls and hasattr(prov, "fit"):
        try:
            out["quota"] = prov.fit(calls)
        except ProviderError as e:  # no token, Space not set: the run would say it too
            out["quota"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ---- enhance: sharper, larger copies of finished clips ------------------------------------
ENHANCED = re.compile(r"\.\d{3,4}p$")  # an enhanced clip's stem ends in .1080p / .1440p
KEEP_EXTRA = ("mode", "source", "window", "window_frames", "refs")  # what the solver reads


def _enhance_source(project: Project, clip: str) -> tuple[Path, str | None]:
    """``(file, slot id)`` of a clip to enhance: a slot id (its approved clip, else its latest)
    or a path (a slot alias ``gen/clips/3.mp4`` = its variant); the slot of a variant is its
    name before ``~``."""
    if not clip.lower().endswith(".mp4") and not re.search(r"[/\\]", clip):
        rel = slot_state.approved(project, "video", clip) or _read_json(
            project.clips_dir / f"{clip}.json").get("path")
        if not rel or not (project.root / rel).is_file():
            raise FileNotFoundError(f"slot {clip}: no generated clip yet")
        path = project.root / rel
    else:
        path = _resolve(project, clip)
    in_clips = path.parent.resolve() == project.clips_dir.resolve()
    if in_clips and "~" not in path.stem and (rel := _read_json(path.with_suffix(".json")).get("path")):
        path = project.root / rel  # an alias: the variant it links
    return path, path.stem.split("~")[0] if in_clips and "~" in path.stem else None


def _enhance_one(project: Project, prov: GenProvider, clip: str, short: int,
                 man: Manifest) -> ManifestItem:
    src, slot = _enhance_source(project, clip)
    if ENHANCED.search(src.stem):
        raise ValueError(f"{_rel(project, src)} is an enhanced clip already")
    fp = prov.fingerprint("enhance")
    key = cache_key(prov.name, "enhance", sha256_file(src), short, fp, PIPELINE_VERSION)
    out = src.with_name(f"{src.stem}.{short}p.mp4")
    prev = _read_json(out.with_suffix(".json"))
    if prev.get("key") == key and out.exists():
        return ManifestItem(**{**prev, "status": "cached", "quota_used": 0.0, "usd_est": 0.0,
                               "latency_s": 0.0})
    from opencine.gen.post import exact_duration, frames_at
    from opencine.gen.qc import video_stream

    if not man.warm:  # wake the Space once, when a clip really has to run
        man.warm = prov.prepare("enhance", [])
    res = prov.enhance(src, out=out, short=short)
    side = _read_json(src.with_suffix(".json"))
    extra = {k: v for k, v in (side.get("extra") or {}).items() if k in KEEP_EXTRA}
    extra.update(res.extra, enhanced_from=_rel(project, src))
    flags = list(extra.pop("flags", []))
    item = ManifestItem(slot=slot or clip, status="ok", path=_rel(project, out),
                        provider=prov.name, model=res.model, prompt=side.get("prompt", ""),
                        seed=side.get("seed"), quota_used=res.quota_used,
                        quota_unit=res.quota_unit, usd_est=res.usd_est, latency_s=res.latency_s,
                        key=key, fp=cache_key(fp), flags=flags,
                        frames=frames_at(exact_duration(video_stream(out)), project.rate),
                        extra=extra)
    out.with_suffix(".json").write_text(json.dumps(item.model_dump(mode="json"), indent=2,
                                                   ensure_ascii=False), "utf-8")
    return item


def _enhanced_takes_over(project: Project, item: ManifestItem) -> None:
    """The enhanced (or face-refined) clip joins its slot's variants and replaces its source
    wherever the source stood: the slot's latest, its approved clip (same gate)."""
    src = item.extra.get("enhanced_from") or item.extra.get("face_from")
    if item.status != "ok" or not item.path or "~" not in Path(item.path).name:
        return
    slot_state.add_variant(project, "video", item.slot, item.path)
    approved = (slot_state.load(project)["video"].get(item.slot) or {}).get("approved") or {}
    if approved.get("path") == src:
        slot_state.approve(project, {item.slot: item.path}, approved.get("gate", "G3"))
    if approved.get("path") == src or _read_json(
            project.clips_dir / f"{item.slot}.json").get("path") == src:
        _alias(project, "video", item.slot, item.path)


def gen_enhance(project: Project, clips: list[str], short: int | None = None,
                provider: str | None = None) -> Manifest:
    """Sharper, larger copies of finished clips (``GenProvider.enhance``: FlashVSR+ on hf): each
    becomes ``gen/clips/<stem>.<short>p.mp4``, same frames, timing, colours and sound, and takes
    its source's place in the slot (``_enhanced_takes_over``). Cached by the source's content and
    the settings; quota exhaustion stops the batch."""
    from opencine.gen.config import get_gen_settings

    name = provider or "hf"
    prov = get_provider(name, project)
    short = short or get_gen_settings().hf_enhance_short
    ledger = Ledger(project)
    man = Manifest(project=project.pid, kind="enhance", provider=name,
                   created=datetime.now().isoformat(timespec="seconds"))
    for i, clip in enumerate(clips):
        try:
            item = _enhance_one(project, prov, clip, short, man)
            _enhanced_takes_over(project, item)
        except Exception as e:
            err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
            item = ManifestItem(slot=clip, status="error", provider=name, prompt="",
                                error=f"{type(e).__name__}: {e}", quota_used=err.quota_used,
                                quota_unit=err.quota_unit, usd_est=err.usd_est)
            if isinstance(e, STOP_ERRORS):
                done = sum(x.status in ("ok", "cached") for x in man.items)
                man.stopped, man.message = True, _stop_message(name, e, done, len(clips) - i - 1)
                man.reset_at = getattr(e, "reset_at", None)
        man.items.append(item)
        _record(ledger, name, "enhance", item)
        if man.stopped:
            man.items += [ManifestItem(slot=c, status="skipped", provider=name, prompt="")
                          for c in clips[i + 1:]]
            break
    out = project.root / "gen" / "manifests" / f"enhance-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    man.path = _rel(project, out)
    out.write_text(man.model_dump_json(indent=2), "utf-8")
    return man


# ---- face pass: a far, waxy face redrawn at a high resolution ------------------------------
FACED = ".face"  # a face-refined clip's stem ends in it


def _face_refs(project: Project, side: dict[str, Any], ref: str | None) -> list[str]:
    """The identity picture of a clip: ``ref`` when given; else its first ref, a character
    sheet's close-up panel (``#face``) when that ref is a sheet panel; none otherwise (the clip's
    own clearest face is used then)."""
    if ref:
        return [ref]
    refs = (side.get("extra") or {}).get("refs") or []
    if not refs:
        return []
    first = str(refs[0]).strip()
    if m := _PANEL.match(first):
        return [f"{m['path']}#face"]
    return [first] if not _FRAME_REF.match(first) and not VIDEO_FRAME.match(first) else []


def _face_audio(project: Project, src: Path, side: dict[str, Any]) -> Path | None:
    """What the clip's mouth should follow: its own sound, else the footage window it was made
    from (a swap is silent; its window holds the sound over the same span)."""
    from opencine.gen.qc import video_stream

    if video_stream(src).get("has_audio"):
        return src
    window = (side.get("extra") or {}).get("window")
    path = project.root / window if window else None
    return path if path and path.is_file() and video_stream(path).get("has_audio") else None


def _face_one(project: Project, prov: GenProvider, clip: str, ref: str | None, details: str,
              force: bool, man: Manifest) -> ManifestItem:
    import tempfile

    from opencine.gen import face
    from opencine.gen.config import get_gen_settings
    from opencine.gen.control import pad_frames, read_rgb
    from opencine.gen.post import exact_duration, frames_at
    from opencine.gen.providers import h3face
    from opencine.gen.qc import fps_of, video_stream

    src, slot = _enhance_source(project, clip)
    if src.stem.endswith(FACED):
        raise ValueError(f"{_rel(project, src)} is face-refined already")
    s = get_gen_settings()
    side = _read_json(src.with_suffix(".json"))
    refs = [_resolve(project, r) for r in _face_refs(project, side, ref)]
    extra_src = side.get("extra") or {}
    details = details or (side.get("prompt", "") if extra_src.get("mode") == "swap" else "")
    fp = prov.fingerprint("face")
    key = cache_key(prov.name, "face", sha256_file(src), [sha256_file(r) for r in refs], details,
                    force, s.hf_face_denoise, s.hf_face_crop, s.hf_face_max_px, fp, face.VERSION,
                    PIPELINE_VERSION)
    out = src.with_name(f"{src.stem}{FACED}.mp4")
    prev = _read_json(out.with_suffix(".json"))
    if prev.get("key") == key and out.exists():
        return ManifestItem(**{**prev, "status": "cached", "quota_used": 0.0, "usd_est": 0.0,
                               "latency_s": 0.0})
    v = video_stream(src)
    frames = read_rgb(src, int(v["width"]), int(v["height"]))
    p = face.plan(frames, face.yunet(), size=s.hf_face_size, base=s.hf_face_denoise,
                  crop=s.hf_face_crop, max_px=s.hf_face_max_px, force=force)
    if p.skip:
        return ManifestItem(slot=slot or clip, status="skipped", provider=prov.name, prompt="",
                            error=p.skip, extra={"face": p.info, "source": _rel(project, src)})
    prompt = h3face.prompt(details)
    n_call, fps = h3face.frames(p.n), fps_of(v)
    seed = int(key[:8], 16) % MAX_SEED
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        orig = face.crops(frames, p)
        crop = face.write(pad_frames(orig, n_call), h3face.FPS, work / "crop.mp4", crf=8)
        if not refs:  # the clip's own clearest face
            refs = [face.self_reference(frames, p, work / "self.png")]
        sound = _face_audio(project, src, side)
        wav = face.soundtrack(sound, n_call / h3face.FPS, work / "sound.wav",
                              tempo=float(h3face.FPS / fps)) if sound else None
        if not man.warm:  # wake the Space once, when a clip really has to run
            man.warm = prov.prepare("face", [])
        res = prov.face(crop, refs, prompt, [*p.strength.tolist(), *[0.0] * (n_call - p.n)],
                        out=work / "redrawn.mp4", audio=wav, seed=seed)
        try:
            redrawn = read_rgb(res.path, s.hf_face_size, s.hf_face_size)[: p.n]
            if len(redrawn) < p.n:
                raise ProviderError(f"face: {len(redrawn)} frames back, {p.n} sent")
            pasted, info = face.paste(frames, redrawn, p, orig)
            face.write(pasted, fps, out, audio_from=src)
        except Exception as e:  # the call was paid for: the error carries its GPU time
            err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
            raise err.spent(res.quota_used, res.quota_unit or "gpu_s") from e
    extra = {k: v for k, v in extra_src.items() if k in KEEP_EXTRA}
    extra.update(res.extra, face={**p.info, **info, "refs": [_rel(project, r) for r in refs],
                                  "audio": _rel(project, sound) if sound else None},
                 face_from=_rel(project, src))
    item = ManifestItem(slot=slot or clip, status="ok", path=_rel(project, out),
                        provider=prov.name, model=res.model, prompt=prompt, seed=res.seed,
                        quota_used=res.quota_used, quota_unit=res.quota_unit, usd_est=res.usd_est,
                        latency_s=res.latency_s, key=key, fp=cache_key(fp),
                        flags=list(side.get("flags") or []),
                        frames=frames_at(exact_duration(video_stream(out)), project.rate),
                        extra=extra)
    out.with_suffix(".json").write_text(json.dumps(item.model_dump(mode="json"), indent=2,
                                                   ensure_ascii=False), "utf-8")
    return item


def gen_face(project: Project, clips: list[str], ref: str | None = None, details: str = "",
             force: bool = False, provider: str | None = None) -> Manifest:
    """Redraw the subject's face of finished clips at a high resolution (``GenProvider.face``:
    the MiniMax-H3 face pass on hf): each becomes ``gen/clips/<stem>.face.mp4``, same
    frames, timing and sound, and takes its source's place in the slot like an enhanced clip.
    A clip whose face is already large (median ≥ ``hf_face_max_px``) or rarely seen is
    ``skipped`` with the reason, unless ``force``. Cached by the source's content, the identity
    picture and the settings; quota exhaustion stops the batch."""
    name = provider or "hf"
    prov = get_provider(name, project)
    ledger = Ledger(project)
    man = Manifest(project=project.pid, kind="face", provider=name,
                   created=datetime.now().isoformat(timespec="seconds"))
    for i, clip in enumerate(clips):
        try:
            item = _face_one(project, prov, clip, ref, details, force, man)
            _enhanced_takes_over(project, item)
        except Exception as e:
            err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
            item = ManifestItem(slot=clip, status="error", provider=name, prompt="",
                                error=f"{type(e).__name__}: {e}", quota_used=err.quota_used,
                                quota_unit=err.quota_unit, usd_est=err.usd_est)
            if isinstance(e, STOP_ERRORS):
                done = sum(x.status in ("ok", "cached") for x in man.items)
                man.stopped, man.message = True, _stop_message(name, e, done, len(clips) - i - 1)
                man.reset_at = getattr(e, "reset_at", None)
        man.items.append(item)
        _record(ledger, name, "face", item)
        if man.stopped:
            man.items += [ManifestItem(slot=c, status="skipped", provider=name, prompt="")
                          for c in clips[i + 1:]]
            break
    out = project.root / "gen" / "manifests" / f"face-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    man.path = _rel(project, out)
    out.write_text(man.model_dump_json(indent=2), "utf-8")
    return man


def _per_provider(provider: str | None, fn: str) -> dict[str, Any]:
    names = [provider] if provider else list(dict.fromkeys(
        [select_provider("image"), select_provider("video")]))
    out: dict[str, Any] = {}
    for n in names:
        try:
            out[n] = getattr(get_provider(n), fn)()
        except ProviderError as e:
            out[n] = {"error": f"{type(e).__name__}: {e}"}
    return out


def quota(provider: str | None = None) -> dict[str, Any]:
    """Quota left per provider (the image and video providers when ``provider`` is None)."""
    return _per_provider(provider, "quota")


def warm(provider: str | None = None) -> dict[str, Any]:
    """Wake remote backends before a batch (HF Spaces); no quota is spent."""
    return _per_provider(provider, "warm")
