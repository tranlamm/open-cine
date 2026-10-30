"""Review gate state (``review/gates.json``): ``{gate, round, items, status, feedback}`` per gate.

Each opening starts a new round; a gate takes as many rounds as the human wants.
Statuses: ``pending`` (opened, waiting) → ``approved`` | ``changes`` | ``cancelled``.
Approved/cancelled gates are closed. Interpreting the human reply is the Director's job;
this module only records it.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencine.core.project import Project

GATES = ("G1", "G2", "G3", "G4")
STATUSES = ("pending", "approved", "changes", "cancelled")
CLOSED = ("approved", "cancelled")


class GateError(ValueError):
    pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _check_gate(gate: str) -> str:
    gate = gate.upper()
    if gate not in GATES:
        raise GateError(f"unknown gate {gate!r}; expected one of {', '.join(GATES)}")
    return gate


def load(project: Project) -> dict[str, dict[str, Any]]:
    p = project.gates_path
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _save(project: Project, gates: dict[str, dict[str, Any]]) -> None:
    p = project.gates_path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(gates, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(p)


def check_can_open(project: Project, gate: str) -> int:
    """Return the round number the next opening would start, or raise ``GateError``."""
    gate = _check_gate(gate)
    g = load(project).get(gate)
    if g is None:
        return 1
    if g["status"] in CLOSED:
        raise GateError(f"{gate} is {g['status']}; it cannot be opened again")
    return g["round"] + 1


def open_round(project: Project, gate: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Start the next round of ``gate`` with ``items``; status becomes ``pending``."""
    rnd = check_can_open(project, gate)
    gate = _check_gate(gate)
    gates = load(project)
    prev = gates.get(gate, {})
    g = {"gate": gate, "round": rnd, "items": items, "status": "pending",
         "feedback": prev.get("feedback", []), "sent": _now()}
    gates[gate] = g
    _save(project, gates)
    return g


def record_feedback(project: Project, gate: str, text: str, status: str | None = None) -> dict[str, Any]:
    """Append feedback for the current round; ``status`` (if given) changes the gate status."""
    gate = _check_gate(gate)
    if status is not None and status not in STATUSES:
        raise GateError(f"unknown status {status!r}; expected one of {', '.join(STATUSES)}")
    gates = load(project)
    if gate not in gates:
        raise GateError(f"{gate} has not been opened yet")
    g = gates[gate]
    g["feedback"].append({"round": g["round"], "text": text, "ts": _now(), "status": status})
    if status is not None:
        g["status"] = status
    _save(project, gates)
    return g
