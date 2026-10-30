"""Per-slot generation state ``gen/slots.json``, kept outside the versioned plans.

``{"image": {"3": {...}}, "video": {"3": {...}}}``, one entry per gen slot id (the ShotPlan
slot number as a string, e.g. ``"3"``)::

    {"attempts": 2,                      # batches in which the slot really ran (not cached)
     "variants": ["gen/clips/3~11.mp4", …],
     "approved": {"path": …, "gate": "G3", "ts": …}}

With ``OPENCINE_GEN_MAX_REROLLS`` set, a slot may run ``1 + OPENCINE_GEN_MAX_REROLLS`` batches;
``refused`` names the slots past that so the batch skips them. Unset (default): no limit.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from opencine.core.config import get_settings

if TYPE_CHECKING:
    from opencine.core.project import Project

Kind = Literal["image", "video"]
KINDS: tuple[Kind, ...] = ("image", "video")
_lock = threading.Lock()


class RerollLimit(RuntimeError):
    pass


def path(project: Project) -> Path:
    return project.root / "gen" / "slots.json"


def load(project: Project) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        data = json.loads(path(project).read_text("utf-8"))
    except (OSError, ValueError):
        data = {}
    return {k: dict(data.get(k) or {}) for k in KINDS}


def _save(project: Project, data: dict[str, Any]) -> None:
    p = path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    os.replace(tmp, p)


def max_batches() -> int | None:
    """Batches a slot may run; None = no limit."""
    n = get_settings().gen_max_rerolls
    return None if n is None else 1 + max(0, n)


def refused(project: Project, kind: Kind, slot_ids: set[str]) -> dict[str, str]:
    """``{slot id: reason}`` for the slots of ``slot_ids`` that used up their batches."""
    state, most = load(project)[kind], max_batches()
    if most is None:
        return {}
    return {sid: (f"slot {sid} already ran {state[sid]['attempts']} {kind} batches "
                  f"(1 + {most - 1} re-rolls): ask the human before spending more, or pick "
                  "one of its variants (OPENCINE_GEN_MAX_REROLLS raises the limit)")
            for sid in sorted(slot_ids)
            if (state.get(sid) or {}).get("attempts", 0) >= most}


def record_runs(project: Project, kind: Kind, runs: dict[str, list[str]]) -> None:
    """One batch ran each slot of ``runs`` (``{slot id: new variant paths}``)."""
    if not runs:
        return
    with _lock:
        data = load(project)
        for sid, paths in runs.items():
            e = data[kind].setdefault(sid, {"attempts": 0, "variants": []})
            e["attempts"] = e.get("attempts", 0) + 1
            e["variants"] = list(dict.fromkeys([*e.get("variants", []), *paths]))
        _save(project, data)


def add_variant(project: Project, kind: Kind, slot_id: str, rel: str) -> None:
    """``rel`` joins the slot's variants without counting a batch (an enhanced copy)."""
    with _lock:
        data = load(project)
        e = data[kind].setdefault(str(slot_id), {"attempts": 0, "variants": []})
        e["variants"] = list(dict.fromkeys([*e.get("variants", []), rel]))
        _save(project, data)


def kind_of(project: Project, rel: str) -> Kind:
    """``gen/stills/…`` → image, ``gen/clips/…`` → video."""
    p = (project.root / rel).resolve()
    for kind, d in (("image", project.stills_dir), ("video", project.clips_dir)):
        if p.is_relative_to(d.resolve()):
            return kind
    raise ValueError(f"{rel}: an approved variant lives under gen/stills/ or gen/clips/")


def approve(project: Project, picks: dict[str, str], gate: str) -> dict[str, dict[str, Any]]:
    """Record ``{slot id: variant path}`` as approved at ``gate``; returns the new entries."""
    out: dict[str, dict[str, Any]] = {}
    with _lock:
        data = load(project)
        for sid, rel in picks.items():
            rel = Path(rel).as_posix()
            if not (project.root / rel).is_file():
                raise FileNotFoundError(f"slot {sid}: {rel} does not exist")
            kind = kind_of(project, rel)
            e = data[kind].setdefault(str(sid), {"attempts": 0, "variants": []})
            e["approved"] = {"path": rel, "gate": gate,
                             "ts": datetime.now().isoformat(timespec="seconds")}
            out[str(sid)] = {"kind": kind, **e["approved"]}
        _save(project, data)
    return out


def approved(project: Project, kind: Kind, slot_id: str) -> str | None:
    """Path of the variant approved for ``slot_id``, if it still exists."""
    e = (load(project)[kind].get(str(slot_id)) or {}).get("approved") or {}
    rel = e.get("path")
    return rel if rel and (project.root / rel).is_file() else None


def summary(project: Project) -> dict[str, Any]:
    """Compact view for ``project_status``: ``{kind: {slot: "2/3 batches · approved x"}}``."""
    most = max_batches()
    return {kind: {sid: {"attempts": f"{e.get('attempts', 0)}" + (f"/{most}" if most else ""),
                         "variants": len(e.get("variants", [])),
                         **({"approved": e["approved"]["path"]} if e.get("approved") else {})}
                   for sid, e in sorted(entries.items())}
            for kind, entries in load(project).items() if entries}
