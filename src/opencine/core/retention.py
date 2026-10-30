"""Retention: keep the newest renders (and their cache), plan versions, sheets and raw gen
clips; store gen aliases as hard links."""

from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencine.core.project import Project

KEEP_RENDERS = 5
KEEP_PLANS = 5  # per kind; plan_solve's newest draft and its ShotPlan stay too (opencine.plan.store)
KEEP_SHEETS = 2  # per kind: critic_compare lays two versions side by side
KEEP_RAW = 5  # provider files (gen/clips/*.raw.mp4); the post-processed clips all stay
_SHEET = re.compile(r"^(editplan|shotplan|scribble|gen)-([^-.]+)(?:-\d+)?\.jpg$")
_LISTED = re.compile(r"^file '(.+)'$", re.MULTILINE)  # a concat list's segments
_CUBE = re.compile(r"([0-9a-f]{16})\.cube")  # a segment graph's LUTs


def link(src: Path, dst: Path) -> None:
    """``dst`` as a hard link to ``src`` (a copy where the volume has none). Replaced, not
    written through: writing into an old link would change the file it shares."""
    tmp = dst.with_name(f"{dst.name}.{secrets.token_hex(4)}.tmp")
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def remove(paths: Iterable[Path]) -> list[Path]:
    """Delete ``paths``; a file another process holds open (Windows) is left for next time."""
    gone = []
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            continue
        gone.append(p)
    return gone


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def stale_renders(project: Project, keep: int = KEEP_RENDERS) -> list[Path]:
    """``renders/*.mp4`` past the newest ``keep``, with their sidecars (a scribble's ``.key``)."""
    d = project.renders_dir
    if not d.is_dir():
        return []
    videos = sorted((p for p in d.glob("*.mp4") if not p.name.endswith(".tmp.mp4")),
                    key=_mtime, reverse=True)
    return [s for p in videos[keep:] for s in d.glob(f"{glob.escape(p.stem)}.*")
            if s == p or (s.suffix != ".mp4" and ".tmp" not in s.name)]


def _rendering(project: Project) -> bool:
    """Another process runs a render or scribble job: its segments are in no render yet."""
    from opencine.core.jobs import pid_alive

    for f in (project.root / "jobs").glob("*.json"):
        try:
            job = json.loads(f.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if (job.get("kind") in ("render", "motion", "scribble") and job.get("status") in ("queued", "running")
                and job.get("pid") != os.getpid() and pid_alive(job.get("pid"))):
            return True
    return False


def stale_render_cache(project: Project) -> list[Path]:
    """``cache/render`` segments, mixes and concat lists and ``cache/lut`` cubes that no render
    in ``renders/`` uses (after ``stale_renders``); none while another render job runs."""
    root = project.cache_dir / "render"
    if not root.is_dir() or _rendering(project):
        return []
    gone = set(stale_renders(project))
    kept = {p.name.split("-")[0] for p in project.renders_dir.glob("*.mp4")
            if p not in gone and ".tmp" not in p.name}
    segs: set[str] = set()
    mixes: set[str] = set()
    lists: set[str] = set()
    unrecorded = False
    for lst in (root / "concat").glob("*.txt"):
        if lst.stem[:10] not in kept:
            continue
        lists.add(lst.stem)
        segs |= {Path(f).stem for f in _LISTED.findall(lst.read_text("utf-8"))}
        try:
            mixes.add(json.loads(lst.with_suffix(".json").read_text("utf-8"))["audio"].split(".")[0])
        except (OSError, ValueError, KeyError, AttributeError):
            unrecorded = True
    if unrecorded:  # a render without a recorded mix: keep the newest mixes
        wavs = sorted((p for p in (root / "audio").glob("*.wav") if p.name.count(".") == 1),
                      key=_mtime, reverse=True)
        mixes |= {p.stem for p in wavs[:KEEP_RENDERS]}
    cubes = {c for g in (root / "segment").glob("*.filtergraph") if g.stem in segs
             for c in _CUBE.findall(g.read_text("utf-8"))}

    def unused(d: Path, used: set[str]) -> list[Path]:
        return [p for p in d.iterdir() if p.is_file() and p.name.split(".")[0] not in used
                and ".tmp" not in p.name] if d.is_dir() else []

    return [*unused(root / "segment", segs), *unused(root / "audio", mixes),
            *unused(root / "concat", lists), *unused(project.cache_dir / "lut", cubes)]


def stale_raws(project: Project, keep: int = KEEP_RAW) -> list[Path]:
    """``gen/clips/*.raw.mp4`` past the newest ``keep``."""
    d = project.clips_dir
    raws = sorted(d.glob("*.raw.mp4"), key=_mtime, reverse=True) if d.is_dir() else []
    return raws[keep:]


def stale_sheets(project: Project, keep: int = KEEP_SHEETS) -> list[Path]:
    """``review/self`` sheets past the newest ``keep`` of each kind; one sheet may span several
    files (``editplan-v031-1.jpg``, ``-2.jpg``)."""
    d = project.review_dir / "self"
    kinds: dict[str, dict[str, list[Path]]] = {}
    for p in d.glob("*.jpg") if d.is_dir() else []:
        if m := _SHEET.match(p.name):
            kinds.setdefault(m[1], {}).setdefault(m[2], []).append(p)
    out: list[Path] = []
    for sheets in kinds.values():
        ranked = sorted(sheets.values(), key=lambda ps: max(map(_mtime, ps)), reverse=True)
        out += [p for ps in ranked[keep:] for p in ps]
    return out


def copied_aliases(project: Project) -> list[tuple[Path, Path]]:
    """``(variant, alias)`` pairs where ``gen/<kind>/<slot>.png|mp4`` is a byte copy of the
    variant its sidecar names, not a link to it (older projects copied instead of linking)."""
    pairs = []
    for d, ext in ((project.stills_dir, ".png"), (project.clips_dir, ".mp4")):
        for alias in d.glob(f"*{ext}") if d.is_dir() else []:
            if "~" in alias.name or not alias.with_suffix(".json").is_file():
                continue
            try:
                rel = json.loads(alias.with_suffix(".json").read_text("utf-8")).get("path")
            except (OSError, ValueError):
                continue
            src = project.root / rel if isinstance(rel, str) else None
            if src is None or not src.is_file() or src.resolve() == alias.resolve():
                continue
            if os.path.samefile(src, alias) or src.stat().st_size != alias.stat().st_size:
                continue
            if _same_bytes(src, alias):
                pairs.append((src, alias))
    return pairs


def _same_bytes(a: Path, b: Path) -> bool:
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def _freed(paths: Iterable[Path]) -> int:
    """Bytes freed by deleting ``paths`` (a file with another hard link frees nothing)."""
    total = 0
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        total += st.st_size if st.st_nlink == 1 else 0
    return total


def clean(project: Project, dry_run: bool = False) -> dict[str, Any]:
    """Apply the retention policy to an existing project (``opencine project clean``)."""
    from opencine.plan.store import stale_plans

    aliases = copied_aliases(project)
    stale = {"renders": stale_renders(project), "cache": stale_render_cache(project),
             "raw": stale_raws(project), "plans": stale_plans(project),
             "sheets": stale_sheets(project)}
    freed = sum(a.stat().st_size for _, a in aliases) + sum(_freed(ps) for ps in stale.values())
    if not dry_run:
        for src, alias in aliases:
            link(src, alias)
        stale = {k: remove(ps) for k, ps in stale.items()}
    rel = [p.relative_to(project.root).as_posix() for ps in stale.values() for p in ps]
    return {"pid": project.pid, "dry_run": dry_run, "freed_mb": round(freed / 2**20, 1),
            "relinked": [a.relative_to(project.root).as_posix() for _, a in aliases],
            **{k: len(ps) for k, ps in stale.items()}, "removed": rel}
