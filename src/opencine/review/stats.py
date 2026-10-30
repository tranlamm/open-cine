"""Revision-round statistics across projects.

Per gate: approvals and average rounds; the share of G4 approved in round 1; how often the critic
agreed with the human. Compare a run with ``OPENCINE_CRITIC=none OPENCINE_TASTE=false`` against one without.
"""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencine.core.project import Project


def review_stats(projects: list[Project], since: str | None = None) -> dict[str, Any]:
    """``since``: ISO date; projects whose gates were all opened before it are left out."""
    from opencine.review import critic, gates

    rounds: dict[str, list[int]] = {g: [] for g in gates.GATES}
    first: list[bool] = []
    used = []
    for p in projects:
        gs = gates.load(p)
        if not gs or (since and not any((v.get("sent") or "") >= since for v in gs.values())):
            continue
        used.append(p)
        for g, v in gs.items():
            if v["status"] == "approved":
                rounds[g].append(v["round"])
        if gs.get("G4", {}).get("status") == "approved":
            first.append(gs["G4"]["round"] == 1)
    return {
        "projects": len(used),
        "gates": {g: {"approved": len(r), "avg_rounds": round(mean(r), 2) if r else None}
                  for g, r in rounds.items()},
        "g4_first_round": round(sum(first) / len(first), 2) if first else None,
        "critic": {k: v for k, v in critic.stats(used).items() if k != "rows"},
    }
