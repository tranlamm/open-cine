"""Review gates in the agent's chat (MCP ``review_open`` / ``review_state``).

Opening a gate records a new round and returns the message and numbered files the agent
shows the human. The human answers in chat; the agent records the reading with
``review_decide``. G1–G3 messages end with the engine's cost line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opencine.core.project import Project
from opencine.review import gates
from opencine.review.status import COST_GATES, cost_line

HINT = 'Reply with e.g. "all ok" · "ok 1,3; 2 lower camera angle" · "cancel".'


def _rel(project: Project, p: Path) -> str:
    return (p.relative_to(project.root) if p.is_relative_to(project.root) else p).as_posix()


def message_text(project: Project, gate: str, rnd: int, summary: str, n_items: int) -> str:
    head = f"{gate} · {project.pid} · round {rnd}"
    items = f"\n\n{n_items} numbered items below." if n_items else ""
    return f"{head}\n{summary}{items}\n\n{HINT}"


def open_review(project: Project, gate: str, summary: str,
                media_paths: list[str | Path]) -> dict[str, Any]:
    """Open the next round of ``gate``; returns ``{pid, gate, round, status, message, items}``."""
    gate = gate.upper()
    rnd = gates.check_can_open(project, gate)
    srcs = [Path(p) if Path(p).is_absolute() else project.root / p for p in media_paths]
    for s in srcs:
        if not s.exists():
            raise FileNotFoundError(s)
    if gate in COST_GATES:
        summary = f"{summary.rstrip()}\n{cost_line(project)}"
    items = [{"n": n, "path": _rel(project, s), "abs": str(s.resolve())}
             for n, s in enumerate(srcs, 1)]
    g = gates.open_round(project, gate, [{"n": i["n"], "src": i["path"]} for i in items])
    return {"pid": project.pid, "gate": gate, "round": rnd, "status": g["status"],
            "message": message_text(project, gate, rnd, summary, len(items)),
            "items": items}


def newest_final(project: Project) -> Path:
    """The most recent ``renders/*-final*.mp4``."""
    rdir = project.renders_dir
    found = sorted(rdir.glob("*-final*.mp4"), key=lambda p: p.stat().st_mtime) if rdir.exists() else []
    if not found:
        raise FileNotFoundError('no final render yet: render(preset="final") first')
    return found[-1]


def review_state(project: Project) -> dict[str, Any]:
    return {"pid": project.pid, "gates": gates.load(project)}
