"""Where a project stands, read from disk: ``project_status`` for the Director and the ``Cost:``
line every G1–G3 message carries (quota left, reset time, spend/cap)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from opencine.core.config import get_settings
from opencine.core.ledger import IMAGE_REVIEW, Ledger

if TYPE_CHECKING:
    from opencine.core.project import Project

COST_GATES = ("G1", "G2", "G3")  # gates before (or between) the spending


def _money(x: float) -> str:
    return f"${x:.2f}"


def _when(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{t:%H:%M}" if t.date() == datetime.now().date() else f"{t:%d/%m %H:%M}"


def quota_text(q: dict[str, Any]) -> str:
    """One provider's ``quota()`` as ``hf 1820/2400 gpu_s left (resets 07:00)``; several HF
    accounts: ``hf 1000/4800 gpu_s left [a 500 · b 500] (resets 07:00)``."""
    name = q.get("provider", "?")
    if "error" in q:
        return f"{name}: quota unreadable ({str(q['error'])[:80]})"
    left, unit = q.get("left"), q.get("unit", "")
    if left is None:
        txt = f"{name} unlimited" if q.get("limit") is None else f"{name} ?"
    else:
        lim = q.get("limit")
        txt = f"{name} {left:g}" + (f"/{lim:g}" if lim is not None else "") + f" {unit} left"
        if accounts := q.get("accounts"):  # one call runs on one account: each one's share
            txt += " [" + " · ".join(f"{a.get('account', '?')} {a['left']:g}" for a in accounts) + "]"
    if reset := _when(q.get("reset_at")):
        txt += f" (resets {reset})"
    return txt


def providers_quota(names: list[str]) -> list[dict[str, Any]]:
    from opencine.gen.service import quota

    out = []
    for n in dict.fromkeys(names):
        try:
            q = quota(n).get(n) or {}
        except Exception as e:  # a status line must never fail a gate
            q = {"error": f"{type(e).__name__}: {e}"}
        out.append({"provider": n, **q})
    return out


def cost_line(project: Project) -> str:
    """``Cost: hf 1820/2400 gpu_s left (resets 07:00) · project $0.40/$5.00``."""
    from opencine.gen.service import select_provider

    qs = providers_quota([select_provider("image"), select_provider("video")])
    spend = Ledger(project).total_usd()
    return ("Cost: " + " · ".join(quota_text(q) for q in qs)
            + f" · project {_money(spend)}/{_money(get_settings().cost_cap_usd)}")


def project_status(project: Project) -> dict[str, Any]:
    """Gates, plan versions, slot state, spend and image budgets — what a resumed turn needs."""
    from opencine.gen import slots
    from opencine.plan.store import KINDS, versions
    from opencine.review import gates

    s = get_settings()
    led = Ledger(project)
    (n, tok), (rn, rtok) = led.image_usage(), led.image_usage(IMAGE_REVIEW)
    gs = {g: {"status": v["status"], "round": v["round"],
              **({"last_feedback": v["feedback"][-1]["text"][:200]} if v.get("feedback") else {})}
          for g, v in sorted(gates.load(project).items())}
    plans = {k: vs[-1] for k in KINDS if (vs := versions(project, k))}
    rdir = project.renders_dir
    renders = sorted(rdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime) if rdir.exists() else []
    return {
        "pid": project.pid,
        "gates": gs or "no gate opened yet",
        "plans": plans or "no plan yet",
        "slots": slots.summary(project) or "no slot generated yet",
        "spend_usd": round(led.total_usd(), 4),
        "cost_cap_usd": s.cost_cap_usd,
        "images": f"{n}/{s.image_budget_count} ({tok}/{s.image_budget_tokens} tok)",
        "review_images": f"{rn}/{s.image_review_count} ({rtok}/{s.image_review_tokens} tok)",
        "renders": [p.relative_to(project.root).as_posix() for p in renders[-4:]],
    }
