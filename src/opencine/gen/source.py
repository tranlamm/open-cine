"""Video inputs for video-to-video: footage windows, cut and framed, and frames of video files.

``cut_window`` renders a window through the renderer so the model gets what the edit would show
(full resolution, tone-mapped like the proxy, cropped to 1080×1920 / 1920×1080 at the plan rate,
with sound), cached as ``gen/sources/<scene>_<in>-<out>_<key>.mp4``. ``video_frame`` grabs one
frame of a video file (``gen/clips/7~1.mp4@0.4``, ``…@last``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opencine.core import ffmpeg
from opencine.core.cache import cache_key, sha256_file
from opencine.core.project import Project
from opencine.core.time import to_frame
from opencine.plan.schema import CropBox

MAX_WINDOW_S = 15.0
DELIVERABLE = {"9:16": "1080x1920", "16:9": "1920x1080"}
SOURCE_VERSION = 1
VIDEO_FRAME = re.compile(r"^(?P<path>.+\.(?:mp4|mov|mkv|webm|m4v))@(?P<t>last|\d+(?:[.,]\d+)?)$",
                         re.IGNORECASE)


class SourceWindow(BaseModel):
    """A stretch of footage (``scene``) or of a generated/rendered clip (``file``) to edit."""

    model_config = ConfigDict(extra="forbid")

    scene: str | None = Field(None, description="Scene id, e.g. 'S03'; in_s/out_s are seconds into it.")
    file: str | None = Field(None, description="A video file instead (e.g. 'gen/clips/7~12.mp4').")
    in_s: float = Field(0.0, ge=0, description="Start, seconds into the scene/file.")
    out_s: float = Field(gt=0, description="End (exclusive), seconds into the scene/file.")
    crop: CropBox | None = Field(None, description="Window of the frame to keep (fractions); wins over reframe.")
    reframe: Literal["track", "center"] = Field("track", description="Wider footage: track = follow the subject, center = fixed centred crop.")

    @model_validator(mode="after")
    def _check(self) -> SourceWindow:
        if (self.scene is None) == (self.file is None):
            raise ValueError("source: give scene or file")
        if self.out_s <= self.in_s:
            raise ValueError("source: out_s must be after in_s")
        if self.out_s - self.in_s > MAX_WINDOW_S + 1e-6:
            raise ValueError(f"source: {self.out_s - self.in_s:.2f} s is longer than "
                             f"{MAX_WINDOW_S:g} s; split it into several slots")
        return self

    @property
    def seconds(self) -> float:
        return self.out_s - self.in_s

    def label(self) -> str:
        where = self.scene.upper() if self.scene else Path(self.file or "").stem
        return f"{where}@{self.in_s:g}-{self.out_s:g}"


def _item(project: Project, w: SourceWindow, aspect: str) -> dict:
    """The one EditPlan item showing the window (seconds, crop, centred / tracked framing)."""
    if w.file is not None:
        item: dict = {"source": {"gen": w.file}, "in_s": w.in_s, "out_s": w.out_s}
    else:
        item = {"source": {"scene": w.scene.strip().upper()}, "in_s": w.in_s, "out_s": w.out_s}
    item |= {"color": "none", "gain": 0.0}
    if w.crop is not None:
        item["crop"] = w.crop.model_dump()
    elif w.scene is not None and w.reframe == "track" and aspect == "9:16":
        from opencine.index.model import load_index
        from opencine.plan.reframe import reframe

        sc = load_index(project).scene(w.scene.strip().upper())
        a, b = (to_frame(t, project.rate, "nearest") for t in (w.in_s, w.out_s))
        crop = reframe(project, sc.clip, sc.start_frame, a, b, b - a)
        if crop is not None:
            item["crop"] = crop
    return item


def _plan(project: Project, w: SourceWindow, aspect: str):
    """``(EditPlan of the window, key)``: the key names the picture (the footage's content and
    where the window sits in it, the framing), not the encoded bytes, so a re-cut file or an
    unrelated re-ingest never looks like a new input to the generation cache."""
    from opencine.plan.schema import EditPlan

    if aspect not in DELIVERABLE:
        raise ValueError(f"source windows are cut at {' or '.join(DELIVERABLE)}, not {aspect}")
    item = _item(project, w, aspect)
    plan = EditPlan.model_validate({"rate": str(project.rate), "deliverable": DELIVERABLE[aspect],
                                    "items": [item]})
    if w.file is not None:
        dep: list = [sha256_file(project.root / w.file)]
    else:
        from opencine.index.model import load_index

        index = load_index(project)
        sc = index.scene(w.scene.strip().upper())
        if to_frame(w.out_s, project.rate, "nearest") > sc.end_frame - sc.start_frame:
            raise ValueError(f"source {w.label()} runs past the end of {sc.id} "
                             f"({(sc.end_frame - sc.start_frame) / float(project.rate):.2f} s)")
        dep = [index.clip(sc.clip).sha256, sc.start_frame]
    return plan, cache_key("source", SOURCE_VERSION, plan.model_dump(mode="json"), dep)


def window_key(project: Project, w: SourceWindow, aspect: str = "9:16") -> str:
    return _plan(project, w, aspect)[1]


def cut_window(project: Project, w: SourceWindow, aspect: str = "9:16") -> Path:
    """The window as an mp4 at the deliverable size of ``aspect`` and the project rate.

    Raises ``ValueError`` for an aspect other than 9:16 / 16:9 or a window past the scene's end."""
    from opencine.plan.validate import errors, validate
    from opencine.render.compile import render

    plan, key = _plan(project, w, aspect)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", w.label().replace("@", "_"))
    out = project.root / "gen" / "sources" / f"{stem}_{key[:8]}.mp4"
    if out.is_file():
        return out
    if bad := errors(validate(plan, project)):
        raise ValueError("; ".join(f"{i.path}: {i.message}" for i in bad))
    out.parent.mkdir(parents=True, exist_ok=True)
    render(project, plan, "final", out=out)
    out.with_suffix(".json").write_text(json.dumps(
        {"window": w.model_dump(mode="json"), "aspect": aspect, "key": key,
         "item": plan.items[0].model_dump(mode="json", by_alias=True)}, indent=2), "utf-8")
    return out


def edit_record(project: Project | None, rel: str) -> dict | None:
    """``{"mode", "source": {scene, in_s, out_s, …}}`` of a video-to-video clip (its sidecar),
    else None: what the solver and the checks need to keep it in step with the footage."""
    if project is None:
        return None
    path = Path(rel) if Path(rel).is_absolute() else project.root / rel
    try:
        extra = json.loads(path.with_suffix(".json").read_text("utf-8")).get("extra") or {}
    except (OSError, ValueError, AttributeError):
        return None
    src = extra.get("source")
    if not (isinstance(src, dict) and extra.get("mode") in ("swap", "restyle")):
        return None
    return {"mode": extra["mode"], "source": src}


def video_frame(project: Project, ref: str) -> Path:
    """``<video>@<seconds>`` or ``<video>@last`` → that frame as a PNG under ``gen/refs/frames/``
    (cached by the file's content). The path is relative to the project root or absolute."""
    m = VIDEO_FRAME.match(ref.strip())
    if not m:
        raise ValueError(f"bad video frame ref {ref!r}; use gen/clips/x.mp4@1.2 or …@last")
    src = Path(m["path"]) if Path(m["path"]).is_absolute() else project.root / m["path"]
    if not src.is_file():
        raise FileNotFoundError(f"input not found: {m['path']}")
    t = m["t"].lower().replace(",", ".")
    key = cache_key("video_frame", sha256_file(src), t)
    out = project.root / "gen" / "refs" / "frames" / f"{src.stem}_{t.replace('.', 'p')}_{key[:8]}.png"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    seek = ["-sseof", "-1"] if t == "last" else ["-ss", t]
    # -update 1 keeps overwriting one image: with -sseof it ends on the very last frame
    ffmpeg.run(["-y", *seek, "-i", str(src), "-vf", "format=rgb24",
                *([] if t == "last" else ["-frames:v", "1"]), "-update", "1", str(out)])
    if not out.is_file():
        raise ValueError(f"{ref}: no frame at that time")
    return out
