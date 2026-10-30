"""Versioned plan storage: ``plans/<kind>.v<NNN>.json``. A save that changes the plan is a new
version; the newest ``KEEP_PLANS`` of each kind stay (``opencine.core.retention``)."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from opencine.core.retention import KEEP_PLANS, remove

if TYPE_CHECKING:
    from opencine.core.project import Project

Kind = Literal["brief", "shotplan", "editplan"]
KINDS: tuple[Kind, ...] = ("brief", "shotplan", "editplan")
_MODEL = {"brief": "Brief", "shotplan": "ShotPlan", "editplan": "EditPlan"}


class PlanInvalid(ValueError):
    def __init__(self, kind: str, issues: list[dict[str, Any]]):
        lines = "; ".join(f"{i['path']}: {i['message']}" for i in issues)
        super().__init__(f"{kind} rejected: {lines}")
        self.issues = issues


def model_for(kind: Kind):
    if kind not in KINDS:
        raise ValueError(f"unknown plan kind {kind!r}; expected one of {KINDS}")
    return getattr(importlib.import_module("opencine.plan.schema"), _MODEL[kind])


def versions(project: Project, kind: Kind) -> list[int]:
    pat = re.compile(rf"^{kind}\.v(\d+)\.json$")
    found = (pat.match(p.name) for p in project.plans_dir.glob(f"{kind}.v*.json"))
    return sorted(int(m.group(1)) for m in found if m)


def save_plan(project: Project, kind: Kind, data: dict[str, Any] | str) -> dict[str, Any]:
    """Parse, validate (EditPlan against the project's media) and store as the next version.

    Raises ``PlanInvalid`` if the schema or any error-level validation issue fails;
    warnings are returned alongside the saved version.
    """
    from pydantic import ValidationError

    raw = json.loads(data) if isinstance(data, str) else data
    try:
        plan = model_for(kind).model_validate(raw)
    except ValidationError as e:
        issues = [{"level": "error", "path": ".".join(map(str, err["loc"])), "message": err["msg"]}
                  for err in e.errors()]
        raise PlanInvalid(kind, issues) from None
    issues: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}
    if kind in ("editplan", "shotplan"):
        from opencine.plan.validate import validate, validate_shotplan

        check = validate if kind == "editplan" else validate_shotplan
        issues = [i.model_dump() for i in check(plan, project)]
        errors = [i for i in issues if i["level"] == "error"]
        if errors:
            raise PlanInvalid(kind, errors)
    if kind in ("editplan", "shotplan"):
        extra["hints"] = _hints(project, plan)
    text = plan.model_dump_json(indent=2, by_alias=True)
    have = versions(project, kind)
    latest = project.plans_dir / f"{kind}.v{have[-1]:03d}.json" if have else None
    if latest is not None and latest.read_text("utf-8") == text:  # nothing changed: no new version
        return {"kind": kind, "version": have[-1], "path": str(latest), "unchanged": True,
                "warnings": issues, **extra}
    version = (have or [0])[-1] + 1
    path = project.plans_dir / f"{kind}.v{version:03d}.json"
    path.write_text(text, "utf-8")
    remove(stale_plans(project))
    return {"kind": kind, "version": version, "path": str(path), "warnings": issues, **extra}


def _sent_versions(project: Project) -> set[int]:
    """EditPlan versions whose render (or a scribble over it: same key prefix) a gate's current
    round shows: the user's notes are about them, and ``critic_compare`` reads them as ``a``."""
    from opencine.review import gates

    keys = {Path(i["src"]).name.split("-")[0] for g in gates.load(project).values()
            for i in g.get("items", []) if Path(i.get("src") or "").parent.name == "renders"}
    out: set[int] = set()
    for f in (project.root / "jobs").glob("*.json") if keys else []:
        try:
            res = json.loads(f.read_text("utf-8")).get("result")
        except (OSError, ValueError, AttributeError):
            continue
        if (isinstance(res, dict) and isinstance(res.get("version"), int)
                and Path(res.get("path") or "").name.split("-")[0] in keys):
            out.add(res["version"])
    return out


def stale_plans(project: Project, keep: int = KEEP_PLANS) -> list[Path]:
    """Plan files past the newest ``keep`` versions of each kind (an EditPlan with its solve
    record), except the newest solver draft and its ShotPlan (``taste.learn`` and
    ``plan_solve --lock`` read them) and EditPlans a gate's current round shows."""
    from opencine.plan.solve import record_path

    drafts = [v for v in versions(project, "editplan") if record_path(project, v).exists()]
    pinned: dict[str, set] = {k: set() for k in KINDS}
    pinned["editplan"] |= _sent_versions(project)
    if drafts:
        pinned["editplan"].add(drafts[-1])
        try:
            pinned["shotplan"].add(json.loads(record_path(project, drafts[-1]).read_text("utf-8"))
                                   .get("shotplan"))
        except (OSError, ValueError):
            pass
    out: list[Path] = []
    for kind in KINDS:
        for v in versions(project, kind)[:-keep]:
            if v in pinned[kind]:
                continue
            out.append(project.plans_dir / f"{kind}.v{v:03d}.json")
            if kind == "editplan" and record_path(project, v).exists():
                out.append(record_path(project, v))
    return out


def _hints(project: Project, plan) -> list[dict[str, Any]]:
    """Craft hints for a valid EditPlan or ShotPlan (against the latest Brief); they never
    block a save."""
    from opencine.plan.schema import EditPlan
    from opencine.plan.validate import craft_hints, shotplan_hints

    try:
        brief = load_plan(project, "brief")[1] if versions(project, "brief") else None
        found = (craft_hints(plan, project, brief) if isinstance(plan, EditPlan)
                 else shotplan_hints(plan, brief))
        return [i.model_dump() for i in found]
    except Exception as e:  # a hint bug must not cost the Director its save
        return [{"level": "hint", "path": "", "message": f"craft hints unavailable: {type(e).__name__}: {e}"}]


def load_plan(project: Project, kind: Kind, version: int | None = None):
    """Return ``(version, model)``; latest version when ``version`` is None."""
    have = versions(project, kind)
    if not have:
        raise FileNotFoundError(f"project {project.pid} has no {kind} yet")
    v = have[-1] if version is None else version
    path = project.plans_dir / f"{kind}.v{v:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"{kind} v{v} not found; have {have}")
    return v, model_for(kind).model_validate_json(path.read_text("utf-8"))
