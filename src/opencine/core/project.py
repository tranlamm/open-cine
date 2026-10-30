"""Project layout (``projects/<pid>/``), id and small key-value state."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from opencine.core.config import get_settings
from opencine.core.time import Rate

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# edit: a short cut from the user's footage · film: every shot generated from a prompt (footage
# and stills are references only) · swap: the user's clip remade with another character
Mode = Literal["edit", "film", "swap"]
FILM_RATE = "24"  # MiniMax-H3's own rate: clips play 1:1, multi-shot cut times stay true

# relative dirs created for every project
LAYOUT = [
    "source", "proxy", "audio", "index/frames", "index/sheets", "plans",
    "gen/stills", "gen/clips", "renders", "review", "cache",
]


class SourceRef(BaseModel):
    id: str  # C01, C02, … (videos) · I01, I02, … (still images)
    file: str  # path relative to the project root, e.g. source/C01.mov
    original: str  # absolute path the user gave


class ProjectMeta(BaseModel):
    pid: str
    created: str
    rate: str
    mode: Mode = "edit"
    sources: list[SourceRef] = []
    images: list[SourceRef] = []  # stills the user sent (EditPlan `{"image": file}`, gen refs)


class ProjectNotFound(LookupError):
    pass


def _expand(paths: list[str | Path], exts: set[str] = VIDEO_EXTS) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in exts))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def expand_media(paths: list[str | Path]) -> list[Path]:
    """Files as given; folders → their videos and stills."""
    return _expand(paths, VIDEO_EXTS | IMAGE_EXTS)


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Project:
    def __init__(self, root: Path, meta: ProjectMeta):
        self.root = root
        self.meta = meta

    # ---- construction -------------------------------------------------
    @classmethod
    def create(
        cls,
        paths: list[str | Path],
        *,
        name: str | None = None,
        rate: str | None = None,
        mode: Mode = "edit",
        projects_dir: Path | None = None,
    ) -> Project:
        """A new project from videos and stills (none is fine: a film, or footage added later).

        A film project runs at ``FILM_RATE`` unless ``rate`` is given."""
        files = expand_media(paths)
        base = Path(projects_dir or get_settings().projects_dir)
        pid = name or f"p{datetime.now():%y%m%d}-{secrets.token_hex(2)}"
        root = base / pid
        if root.exists():
            raise FileExistsError(f"project {pid} already exists at {root}")
        for d in LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        meta = ProjectMeta(
            pid=pid,
            created=datetime.now().isoformat(timespec="seconds"),
            rate=str(Rate(rate or (FILM_RATE if mode == "film" else get_settings().default_rate))),
            mode=mode,
        )
        proj = cls(root, meta)
        for f in files:
            proj.add_image(f) if is_image(f) else proj.add_source(f)
        proj.save()
        return proj

    @classmethod
    def load(cls, pid: str, *, projects_dir: Path | None = None) -> Project:
        root = Path(projects_dir or get_settings().projects_dir) / pid
        meta_path = root / "project.json"
        if not meta_path.exists():
            raise ProjectNotFound(f"no project {pid!r} under {root.parent}")
        return cls(root, ProjectMeta.model_validate_json(meta_path.read_text("utf-8")))

    def add_source(self, path: Path) -> SourceRef:
        cid = f"C{len(self.meta.sources) + 1:02d}"
        dst = self.source_dir / f"{cid}{path.suffix.lower()}"
        _link_or_copy(path, dst)
        ref = SourceRef(id=cid, file=dst.relative_to(self.root).as_posix(), original=str(path.resolve()))
        self.meta.sources.append(ref)
        return ref

    def add_image(self, path: Path) -> SourceRef:
        iid = f"I{len(self.meta.images) + 1:02d}"
        dst = self.source_dir / f"{iid}{path.suffix.lower()}"
        _link_or_copy(path, dst)
        ref = SourceRef(id=iid, file=dst.relative_to(self.root).as_posix(), original=str(path.resolve()))
        self.meta.images.append(ref)
        return ref

    def add_sources(self, paths: list[str | Path]) -> list[SourceRef]:
        """Add videos as the next clips (C04…) and stills as images (I01…); folders expand to both."""
        refs = [self.add_image(f) if is_image(f) else self.add_source(f) for f in expand_media(paths)]
        self.save()
        return refs

    def save(self) -> None:
        (self.root / "project.json").write_text(self.meta.model_dump_json(indent=2), "utf-8")

    # ---- layout -------------------------------------------------------
    @property
    def pid(self) -> str:
        return self.meta.pid

    @property
    def rate(self) -> Rate:
        return Rate(self.meta.rate)

    @property
    def film(self) -> bool:
        return self.meta.mode == "film"

    @property
    def swap(self) -> bool:
        return self.meta.mode == "swap"

    source_dir = property(lambda self: self.root / "source")
    proxy_dir = property(lambda self: self.root / "proxy")
    audio_dir = property(lambda self: self.root / "audio")
    index_dir = property(lambda self: self.root / "index")
    frames_dir = property(lambda self: self.root / "index" / "frames")
    sheets_dir = property(lambda self: self.root / "index" / "sheets")
    plans_dir = property(lambda self: self.root / "plans")
    stills_dir = property(lambda self: self.root / "gen" / "stills")
    clips_dir = property(lambda self: self.root / "gen" / "clips")
    renders_dir = property(lambda self: self.root / "renders")
    review_dir = property(lambda self: self.root / "review")
    cache_dir = property(lambda self: self.root / "cache")
    gates_path = property(lambda self: self.root / "review" / "gates.json")
    costs_path = property(lambda self: self.root / "costs.jsonl")
    state_path = property(lambda self: self.root / "state.sqlite")

    def source_path(self, cid: str) -> Path:
        for s in self.meta.sources:
            if s.id == cid:
                return self.root / s.file
        raise KeyError(f"no source {cid!r} in project {self.pid}")

    # ---- key-value state ------------------------------------------------
    def _db(self) -> closing[sqlite3.Connection]:
        con = sqlite3.connect(self.state_path, timeout=30)
        con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return closing(con)

    def state_get(self, key: str, default=None):
        with self._db() as db, db as con:

            row = con.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def state_set(self, key: str, value) -> None:
        with self._db() as db, db as con:

            con.execute(
                "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (key, json.dumps(value)),
            )


def list_projects(projects_dir: Path | None = None) -> list[str]:
    base = Path(projects_dir or get_settings().projects_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "project.json").exists())
