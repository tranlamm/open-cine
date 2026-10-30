"""Taste memory across projects, in ``OPENCINE_TASTE_DIR`` (default ``~/.opencine/taste``).

Rules (``rules.json``): preferences the Director distils from feedback ("no flash", "hook text ≤ 4
words"), weighted by how often they recur. Numbers (``profile.json``, per template): the solver's
draft vs each approved EditPlan, folded with an EMA (α = ``ALPHA``); ``plan_solve`` applies them,
clamped, once ``MIN_PROJECTS`` approved projects agree.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from opencine.core.config import get_settings
from opencine.plan.templates import Template

if TYPE_CHECKING:
    from opencine.core.project import Project

ALPHA = 0.3
MIN_PROJECTS = 2
MAX_RULES = 30
SCALE_RANGE = (0.75, 1.3)  # role length scale
DB_RANGE = 6.0  # ± gain offset
AVOID_AT = 2  # net times a kind was taken out before the solver stops using it
SUMMARY_RULES = 8
SUMMARY_CHARS = 600  # ~300 tokens


def _dir() -> Path:
    return get_settings().taste_dir


def enabled() -> bool:
    return get_settings().taste


def _load(name: str, default: Any) -> Any:
    p = _dir() / name
    return json.loads(p.read_text("utf-8")) if p.exists() else default


def _save(name: str, data: Any) -> None:
    p = _dir() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(p)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- rules ------------------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def load_rules() -> list[dict[str, Any]]:
    return _load("rules.json", [])


def note(text: str = "", scope: str = "all", weight: int = 1, evidence: str = "",
         rule_id: str | None = None) -> dict[str, Any]:
    """Add a rule, or strengthen it (same words, same scope; or ``rule_id``) by ``weight``; a
    negative weight weakens it, and at 0 it is forgotten. → ``{rule, action}``."""
    if not enabled():
        return {"action": "skipped", "reason": "OPENCINE_TASTE=off"}
    rules = load_rules()
    scope = (scope or "all").strip().lower()
    if rule_id:
        rule = next((r for r in rules if r["id"] == rule_id), None)
        if rule is None:
            raise KeyError(f"no taste rule {rule_id!r}; see taste()")
    else:
        if not text.strip():
            raise ValueError("give the rule's text (or rule_id to change one)")
        rule = next((r for r in rules if r["scope"] == scope and _norm(r["text"]) == _norm(text)), None)
    if rule is None:
        if weight <= 0:
            return {"action": "none", "reason": "no such rule to weaken"}
        n = max((int(r["id"][1:]) for r in rules), default=0) + 1
        rule = {"id": f"r{n}", "text": " ".join(text.split())[:200], "scope": scope, "weight": 0,
                "evidence": []}
        rules.append(rule)
        action = "added"
    else:
        action = "strengthened" if weight > 0 else "weakened"
    rule["weight"] += weight
    if evidence and evidence not in rule["evidence"]:
        rule["evidence"] = (rule["evidence"] + [evidence])[-5:]
    rule["updated"] = _now()
    if rule["weight"] <= 0:
        rules.remove(rule)
        action = "forgotten"
    while len(rules) > MAX_RULES:  # the weakest, least recently confirmed goes
        rules.remove(min(rules, key=lambda r: (r["weight"], r.get("updated", ""))))
    _save("rules.json", rules)
    return {"action": action, "rule": rule}


def forget(rule_id: str) -> bool:
    rules = load_rules()
    kept = [r for r in rules if r["id"] != rule_id]
    _save("rules.json", kept)
    return len(kept) != len(rules)


def rules_for(template: str | None) -> list[dict[str, Any]]:
    t = (template or "").strip().lower()
    return sorted((r for r in load_rules() if r["scope"] in ("all", t)),
                  key=lambda r: (-r["weight"], r.get("updated", "")))


# ---- numbers ------------------------------------------------------------------------------
def load_profile() -> dict[str, Any]:
    return _load("profile.json", {})


def _ema(old: float | None, obs: float) -> float:
    return obs if old is None else old + ALPHA * (obs - old)


def _key(template: str | None) -> str:
    return (template or "").strip().lower() or "none"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def observe(draft: dict[str, Any], final: dict[str, Any], roles: dict[int, str | None]) -> dict[str, Any]:
    """What the approved ``final`` EditPlan changed against the solver's ``draft`` (both dumps):
    per-role length ratios, gain offsets, transitions/sfx taken out (+1) or put in (−1)."""
    def by_slot(plan):
        return {it["slot"]: it for it in plan.get("items", []) if it.get("slot") is not None}

    def length(it):
        from opencine.plan.schema import EditItem

        return EditItem.model_validate(it).length

    d, f = by_slot(draft), by_slot(final)
    ratios: dict[str, list[float]] = {}
    gains: list[float] = []
    trans: dict[str, int] = {}
    for sid in d.keys() & f.keys():
        a, b = d[sid], f[sid]
        role = roles.get(sid) or "none"
        ratios.setdefault(role, []).append(length(b) / max(1, length(a)))
        if "scene" in a["source"] and a.get("gain") is not None and b.get("gain") is not None:
            gains.append(b["gain"] - a["gain"])
        ka = (a.get("transition_in") or {}).get("kind", "cut")
        kb = (b.get("transition_in") or {}).get("kind", "cut")
        if ka != kb:
            if ka != "cut":
                trans[ka] = trans.get(ka, 0) + 1
            if kb != "cut":
                trans[kb] = trans.get(kb, 0) - 1
    obs: dict[str, Any] = {"role_scale": {r: median(v) for r, v in ratios.items()}, "transitions": trans}
    if gains:
        obs["footage_db"] = median(gains)
    dm, fm = draft.get("music"), final.get("music")
    if dm and fm:
        obs["music_db"] = fm.get("gain", 0.0) - dm.get("gain", 0.0)

    def sounds(plan):
        return {Path(s["file"]).stem: s.get("gain", 0.0) for s in plan.get("sfx", [])}

    ds, fs = sounds(draft), sounds(final)
    obs["sfx"] = {n: 1 for n in ds.keys() - fs.keys()} | {n: -1 for n in fs.keys() - ds.keys()}
    if both := ds.keys() & fs.keys():
        obs["sfx_db"] = median(fs[n] - ds[n] for n in both)
    return obs


def fold(profile: dict[str, Any], template: str | None, obs: dict[str, Any], pid: str) -> dict[str, Any]:
    t = profile.setdefault(_key(template), {"projects": 0, "seen": [], "role_scale": {},
                                            "transitions": {}, "sfx": {}})
    if pid in t["seen"]:
        return t
    t["seen"] = (t["seen"] + [pid])[-50:]
    t["projects"] += 1
    for role, r in obs["role_scale"].items():
        t["role_scale"][role] = round(_ema(t["role_scale"].get(role), r), 3)
    for k in ("music_db", "footage_db", "sfx_db"):
        if k in obs:
            t[k] = round(_ema(t.get(k), obs[k]), 2)
    for group in ("transitions", "sfx"):
        for name, n in obs[group].items():
            t[group][name] = t[group].get(name, 0) + n
    t["updated"] = _now()
    return t


def _draft_version(project: Project, upto: int) -> int | None:
    from opencine.plan.solve import record_path
    from opencine.plan.store import versions

    return next((v for v in sorted(versions(project, "editplan"), reverse=True)
                 if v <= upto and record_path(project, v).exists()), None)


def learn(project: Project) -> dict[str, Any]:
    """Fold the approved (latest) EditPlan vs the solver draft it came from into the profile."""
    if not enabled():
        return {"learned": False, "reason": "OPENCINE_TASTE=off"}
    from opencine.plan.solve import record_path
    from opencine.plan.store import load_plan

    final_v, final = load_plan(project, "editplan")
    draft_v = _draft_version(project, final_v)
    if draft_v is None:
        return {"learned": False, "reason": "no plan_solve draft to compare with"}
    _, draft = load_plan(project, "editplan", draft_v)
    rec = json.loads(record_path(project, draft_v).read_text("utf-8"))
    _, sp = load_plan(project, "shotplan", rec.get("shotplan"))
    roles = {s.slot: s.role for s in sp.slots}
    obs = observe(draft.model_dump(mode="json", by_alias=True),
                  final.model_dump(mode="json", by_alias=True), roles)
    profile = load_profile()
    t = fold(profile, sp.template, obs, project.pid)
    _save("profile.json", profile)
    return {"learned": True, "template": _key(sp.template), "draft": draft_v, "approved": final_v,
            "projects": t["projects"]}


def adjust(tpl: Template) -> tuple[Template, list[str]]:
    """``tpl`` with the learned numbers applied (once ``MIN_PROJECTS`` agree) + report notes."""
    if not enabled():
        return tpl, []
    t = load_profile().get(_key(tpl.name))
    if not t or t.get("projects", 0) < MIN_PROJECTS:
        return tpl, []
    notes = []
    weights = dict(tpl.role_weight)
    for role, r in t.get("role_scale", {}).items():
        s = _clamp(r, *SCALE_RANGE)
        if role in weights and abs(s - 1) >= 0.05:
            weights[role] = round(weights[role] * s, 3)
            notes.append(f"{role} ×{s:.2f}")
    changes: dict[str, Any] = {"role_weight": weights}
    for k, label in (("music_db", "music"), ("footage_db", "original audio"), ("sfx_db", "sfx")):
        if (v := t.get(k)) is not None and abs(v) >= 0.5:
            v = _clamp(v, -DB_RANGE, DB_RANGE)
            changes[k] = getattr(tpl, k) + v
            notes.append(f"{label} {v:+.1f} dB")
    avoid_tr = {k for k, n in t.get("transitions", {}).items() if n >= AVOID_AT}
    if tpl.chapter_transition in avoid_tr:
        changes["chapter_transition"] = None
        notes.append(f"drop {tpl.chapter_transition} between chapters")
    avoid_sfx = {k for k, n in t.get("sfx", {}).items() if n >= AVOID_AT}
    if dropped := {e: n for e, n in tpl.auto_sfx.items() if n in avoid_sfx}:
        changes["auto_sfx"] = {e: n for e, n in tpl.auto_sfx.items() if e not in dropped}
        notes.append("drop sfx " + ", ".join(sorted(set(dropped.values()))))
    out = replace(tpl, **changes)
    return out, ([f"taste ({t['projects']} projects): " + " · ".join(notes)] if notes else [])


# ---- what the Director reads ------------------------------------------------------------------
def summary(template: str | None = None) -> str:
    """≤ ~300 tokens: the strongest rules for ``template`` + one line of learned numbers."""
    if not enabled():
        return "Taste: off (OPENCINE_TASTE=off)."
    lines = []
    for r in rules_for(template)[:SUMMARY_RULES]:
        scope = "" if r["scope"] == "all" else f" ({r['scope']})"
        lines.append(f"- [{r['id']} ×{r['weight']}]{scope} {r['text']}")
    t = load_profile().get(_key(template)) if template else None
    if t:
        _, notes = adjust(Template(_key(template)))
        state = notes[0] if notes else f"learned from {t['projects']} projects (applies from {MIN_PROJECTS})"
        lines.append(f"Numbers ({_key(template)}): {state}")
    if not lines:
        return ("Taste: nothing yet (record one with taste_note when the user states a general "
                "preference in a review).")
    text = "Taste remembered:\n" + "\n".join(lines)
    return text if len(text) <= SUMMARY_CHARS else text[:SUMMARY_CHARS - 1] + "…"
