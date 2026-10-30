"""Data contract for ``index/scenes.json``: written by ingest, read by plan/render/MCP.

Frames are at the **project rate** on each clip's CFR proxy (``proxy/<clip>.mp4``), so a
scene's ``[start_frame, end_frame)`` indexes proxy frames directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from opencine.core.project import Project

INDEX_VERSION = 3  # 3: scene motion directions, colour, audio peak (optional fields)


class ClipInfo(BaseModel):
    id: str  # C01
    source: str  # relative to project root
    proxy: str  # relative, CFR at project rate, 720p
    audio: str | None  # relative WAV 48 kHz, None if the source has no audio
    sha256: str
    proxy_key: str = ""  # cache key of the proxy encode; keys every image derived from it
    # from probe (source file)
    codec: str
    fps_nominal: str  # r_frame_rate as fraction string
    fps_avg: str  # avg_frame_rate
    vfr: bool
    hdr: str | None  # "hlg" | "pq" | None
    rotation: int  # degrees, as displayed
    src_width: int  # display orientation (after rotation)
    src_height: int
    # proxy
    width: int
    height: int
    frames: int  # proxy frame count at project rate
    duration_s: float
    shot_at: str | None = None  # ISO local time the recording started; None = unknown
    shot_from: str | None = None  # "tag" (container metadata) | "name" (date-time in the file name)
    same_as: str | None = None  # the same footage as this (better) clip: sent twice (index.dupes)


class Scene(BaseModel):
    id: str  # S01 … numbered across the whole project in clip order
    clip: str  # C01
    start_frame: int  # inclusive, proxy frame at project rate
    end_frame: int  # exclusive
    start_s: float
    end_s: float
    motion: float  # 0–1
    brightness: float  # 0–1
    sharpness: float  # 0–1
    keyframes: list[float] = []  # representative times, seconds within the clip (display, names)
    motion_peak_s: float | None = None
    keyframe_frames: list[int] = []  # the same moments as exact proxy frames
    motion_peak_frame: int | None = None
    # planning without images (index v3; None in older indexes)
    cam_motion: float | None = None  # 0–1 how much the camera moves (shake included)
    cam_dir: str | None = None  # camera move: → ← ↑ ↓ (pan/tilt), zoom+ / zoom-, "shake"; None = still
    subject_motion: float | None = None  # 0–1 how fast things move against the camera
    head_dir: str | None = None  # on-screen motion over the first ~1 s: arrow, zoom±, "·"
    tail_dir: str | None = None  # … over the last ~1 s
    colour: str | None = None  # dominant colour name (red, teal, grey…)
    audio_peak_s: float | None = None  # strongest sound onset, seconds within the clip

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class SceneIndex(BaseModel):
    version: int = INDEX_VERSION
    rate: str
    clips: list[ClipInfo]
    scenes: list[Scene]

    def clip(self, cid: str) -> ClipInfo:
        for c in self.clips:
            if c.id == cid:
                return c
        raise KeyError(f"no clip {cid!r}")

    def shot_order(self) -> list[str]:
        """Clip ids in shooting order (timed clips by time, then the rest by id): the default
        story order."""
        timed = sorted((c for c in self.clips if c.shot_at), key=lambda c: (_utc(c.shot_at), c.id))
        return [c.id for c in timed] + [c.id for c in self.clips if not c.shot_at]

    def shown_scenes(self) -> list[Scene]:
        """Scenes of the clips that are not a duplicate of another (``ClipInfo.same_as``)."""
        dup = {c.id for c in self.clips if c.same_as}
        return [s for s in self.scenes if s.clip not in dup]

    def scene(self, sid: str) -> Scene:
        for s in self.scenes:
            if s.id == sid:
                return s
        raise KeyError(f"no scene {sid!r}")


def _utc(iso: str) -> str:
    from datetime import UTC, datetime

    return datetime.fromisoformat(iso).astimezone(UTC).isoformat()  # naive = this machine's zone


def index_path(project: Project):
    return project.index_dir / "scenes.json"


def load_index(project: Project) -> SceneIndex:
    """The scene index; empty for a project with no videos (a film from a prompt, stills only)."""
    p = index_path(project)
    if not p.exists():
        if not project.meta.sources:
            return SceneIndex(rate=project.meta.rate, clips=[], scenes=[])
        raise FileNotFoundError(f"project {project.pid} is not ingested yet (no {p})")
    return SceneIndex.model_validate_json(p.read_text("utf-8"))


def save_index(project: Project, index: SceneIndex) -> None:
    index_path(project).write_text(index.model_dump_json(indent=2), "utf-8")


def samples_path(project: Project, clip: str):
    """``index/samples/<clip>.json``: one ``frames.sample_row`` every 0.5 s (``f`` = proxy frame
    at the project rate)."""
    return project.index_dir / "samples" / f"{clip}.json"


def load_samples(project: Project, clip: str) -> list[dict] | None:
    """A clip's frame samples, or None when its ingest predates them."""
    import json

    p = samples_path(project, clip)
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def subjects_path(project: Project, clip: str):
    """``index/subjects/<clip>.json``: ``[{f, box: [x, y, w, h], kind, conf}]`` every 0.5 s
    (``index.subjects``; ``f`` = proxy frame at the project rate)."""
    return project.index_dir / "subjects" / f"{clip}.json"


def load_subjects(project: Project, clip: str) -> list[dict] | None:
    """A clip's subject rows, or None when its ingest predates them."""
    import json

    p = subjects_path(project, clip)
    return json.loads(p.read_text("utf-8")) if p.exists() else None
