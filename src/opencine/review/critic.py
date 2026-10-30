"""Optional second opinion before a gate: a judge from another model family compares two versions.

The judge sees each side's ``review_sheet`` (or ``gen_sheet``) and table, the brief's goal and the
human's feedback on the earlier side, never the Director's reasoning (it would tend to agree).
Pairwise, asked twice with the order swapped: disagreement is a ``tie``, and a feedback point is
fixed only if both answers say so. ``codex`` runs ``codex exec`` (ChatGPT subscription, read-only
sandbox, JSON schema, no saved session); ``none`` skips. Answers are cached and logged to
``review/critic.jsonl`` for ``stats``. The verdict advises; it never blocks a gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opencine.core.cache import cache_key
from opencine.core.config import get_settings
from opencine.core.proc import NO_WINDOW

if TYPE_CHECKING:
    from opencine.core.project import Project

VERSION = 1
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["winner", "fixed", "regressions", "notes"],
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "fixed": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["point", "ok", "why"],
            "properties": {"point": {"type": "string"}, "ok": {"type": "boolean"},
                           "why": {"type": "string"}}}},
        "regressions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
}
RUBRIC = ("hook in 0–2 s (is the first frame compelling), pacing (shot lengths, freezes), framing "
          "(subject in frame, off-centre crop, pixelated zoom), transitions, text (readable, not covering "
          "the subject), consistent colour across shots, obvious errors (black frames, deformed hands/objects)")
# a film project is judged as a short film, not a short-form video
FILM_RUBRIC = ("opening (an establishing shot, draws you into the story), storytelling pace (shots long "
               "enough to read, the climax cuts faster, no needless freezes), continuity between shots "
               "(character, costume, props, lighting, eyeline and direction of motion), cinematic "
               "composition, transitions, text, a consistent colour/look, obvious errors (black frames, "
               "deformed hands/objects, a character's face changing)")


class CriticError(RuntimeError):
    pass


# ---- the judge ---------------------------------------------------------------------
def _command(binary: str) -> list[str]:
    if binary.endswith(".py"):  # a Python script standing in for the judge
        return [sys.executable, binary]
    from opencine.core.proc import find_exe

    return [find_exe(binary)]


def ask_codex(images: list[Path], prompt: str) -> dict[str, Any]:
    """One ``codex exec`` answer, parsed (raises ``CriticError``)."""
    s = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        schema, out = Path(tmp) / "schema.json", Path(tmp) / "answer.json"
        schema.write_text(json.dumps(SCHEMA), "utf-8")
        cmd = [*_command(s.codex_bin), "exec", "-", "--sandbox", "read-only", "--skip-git-repo-check",
               "--ephemeral", "--output-schema", str(schema), "-o", str(out),
               "--image", ",".join(str(p.resolve()) for p in images)]
        if s.critic_model:
            cmd += ["-m", s.critic_model]
        if s.critic_effort:
            cmd += ["-c", f'model_reasoning_effort="{s.critic_effort}"']
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", cwd=tmp, timeout=s.critic_timeout_s,
                                  **NO_WINDOW)
        except subprocess.TimeoutExpired as e:
            raise CriticError(f"codex timed out after {s.critic_timeout_s} s") from e
        except OSError as e:
            raise CriticError(f"cannot run {s.codex_bin!r}: {e}") from e
        if proc.returncode != 0:
            raise CriticError(f"codex exited {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        try:
            answer = json.loads(out.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise CriticError(f"codex gave no JSON answer: {e}") from e
    if answer.get("winner") not in ("A", "B", "tie"):
        raise CriticError(f"codex answer has no winner: {answer!r}")
    return answer


# ---- what the judge sees -------------------------------------------------------------
def _side(project: Project, ref: int | str) -> tuple[list[Path], str]:
    """Sheet image(s) + text of one side: an EditPlan version (its rendered preview) or a
    generated file."""
    from opencine.review.selfcheck import gen_sheet, review_sheet

    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        refs, table = review_sheet(project, int(ref))
    else:
        refs, table = gen_sheet(project, [str(ref)])
    return [Path(r.path) for r in refs], table


def _goal(project: Project) -> str:
    from opencine.plan.store import load_plan

    try:
        _, brief = load_plan(project, "brief", None)
    except (FileNotFoundError, KeyError, ValueError):
        return ""
    return brief.goal


def prompt(goal: str, first: tuple[str, str], second: tuple[str, str], feedback: str,
           old_label: str, film: bool = False) -> str:
    """The question for one ordering: ``first``/``second`` = (label, text) as shown;
    ``film``: judged as a short film (``FILM_RUBRIC``)."""
    new_label = "B" if old_label == "A" else "A"
    lines = [
        "You are a film editor (a short film built from generated shots), judging independently." if film else
        "You are an editor of short vertical 9:16 videos (TikTok/Reels), judging independently.",
        f"Video goal: {goal}" if goal else "",
        f"Two edits, A and B. Attached images: first edit {first[0]}, then edit {second[0]} "
        "(each image is a grid of frames: the first frame = the hook, then the middle frame of each shot, "
        "labelled #n with its duration).",
        f"Edit {old_label} is the old version, edit {new_label} the revision.",
        f"## Edit {first[0]}\n{first[1]}", f"## Edit {second[0]}\n{second[1]}",
    ]
    if feedback.strip():
        lines += [f"## The reviewer's feedback on edit {old_label}\n{feedback.strip()}",
                  f"For each feedback point: did edit {new_label} fix it (fixed: point, ok, why)."]
    else:
        lines.append("No feedback: leave fixed empty.")
    lines += [f"List new errors edit {new_label} has that edit {old_label} does not (regressions).",
              f"Criteria: {FILM_RUBRIC if film else RUBRIC}.",
              "winner: the edit that is better to publish (A, B, or tie if equal). notes: ≤ 2 sentences. "
              "Judge only from the images and tables; answer in English, matching the schema."]
    return "\n\n".join(x for x in lines if x)


def _digest(paths: list[Path]) -> list[str]:
    return [hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in paths]


def _cached(project: Project, key: str, fn) -> dict[str, Any]:
    p = project.cache_dir / "critic" / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    answer = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(answer, ensure_ascii=False), "utf-8")
    return answer


def _merge(run1: dict[str, Any], run2: dict[str, Any]) -> dict[str, Any]:
    """Run 1 shows (a, b) as (A, B); run 2 shows (b, a). → winner in a/b terms."""
    w1 = {"A": "a", "B": "b", "tie": "tie"}[run1["winner"]]
    w2 = {"A": "b", "B": "a", "tie": "tie"}[run2["winner"]]
    agree = w1 == w2
    second = {f["point"].strip().lower(): f for f in run2.get("fixed", [])}
    fixed = []
    for k, f in enumerate(run1.get("fixed", [])):
        other = second.get(f["point"].strip().lower())
        if other is None and k < len(run2.get("fixed", [])):
            other = run2["fixed"][k]  # same point, worded differently
        both = bool(f["ok"]) and bool(other and other["ok"])
        fixed.append({"point": f["point"], "ok": both, "why": f["why"] if both or not other else
                      (other["why"] if f["ok"] else f["why"])})
    regressions = list(dict.fromkeys([*run1.get("regressions", []), *run2.get("regressions", [])]))
    notes = " / ".join(x for x in (run1.get("notes"), run2.get("notes")) if x)
    return {"winner": w1 if agree else "tie", "agree": agree, "fixed": fixed,
            "regressions": regressions, "notes": notes}


def ok_verdict(v: dict[str, Any]) -> bool:
    """The critic's call on ``b``: better (or as good), every feedback point fixed, nothing new broken."""
    return (v.get("winner") in ("b", "tie") and all(f["ok"] for f in v.get("fixed", []))
            and not v.get("regressions"))


def compare(project: Project, a: int | str, b: int | str, feedback: str = "",
            gate: str | None = None) -> dict[str, Any]:
    """Compare ``a`` (the earlier version / variant) with ``b``. ``feedback``: what the human
    said about ``a`` (or what to judge, for variants). ``gate``: the gate ``b`` goes to next
    (logged with the round it will open, for ``stats``)."""
    s = get_settings()
    base = {"a": a, "b": b, "judge": s.critic}
    if s.critic == "none":
        return base | {"winner": "skipped", "reason": "OPENCINE_CRITIC=none"}
    (ia, ta), (ib, tb) = _side(project, a), _side(project, b)
    goal = _goal(project)
    model = f"{s.critic_model or 'default'}/{s.critic_effort or 'default'}"
    try:
        runs = []
        for first, second, old in (((ia, ("A", ta)), (ib, ("B", tb)), "A"),
                                   ((ib, ("A", tb)), (ia, ("B", ta)), "B")):
            q = prompt(goal, first[1], second[1], feedback, old, film=project.film)
            imgs = first[0] + second[0]
            key = cache_key("critic", _digest(imgs), q, model, VERSION)
            runs.append(_cached(project, key, lambda imgs=imgs, q=q: ask_codex(imgs, q)))
    except CriticError as e:
        return base | {"winner": "skipped", "reason": str(e)}
    verdict = base | _merge(*runs) | {"model": model}
    verdict["ok"] = ok_verdict(verdict)
    _log(project, verdict, gate)
    return verdict


# ---- calibration -----------------------------------------------------------------------
def log_path(project: Project) -> Path:
    return project.review_dir / "critic.jsonl"


def _log(project: Project, verdict: dict[str, Any], gate: str | None) -> None:
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "a": verdict["a"], "b": verdict["b"],
           "winner": verdict["winner"], "agree": verdict["agree"], "ok": verdict["ok"],
           "fixed": [f["ok"] for f in verdict["fixed"]], "regressions": len(verdict["regressions"])}
    if gate:
        from opencine.review import gates

        rec["gate"] = gate.upper()
        try:
            rec["round"] = gates.check_can_open(project, gate)
        except gates.GateError:
            rec["round"] = None
    p = log_path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _records(project: Project) -> list[dict[str, Any]]:
    p = log_path(project)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()]


def stats(projects: list[Project]) -> dict[str, Any]:
    """How often the critic's call on ``b`` matched the human's decision on ``b``'s round
    (approved ↔ ok)."""
    from opencine.review import gates

    total = agreed = 0
    rows = []
    for p in projects:
        state = gates.load(p)
        for r in _records(p):
            if not r.get("gate") or not r.get("round"):
                continue
            fb = [f for f in state.get(r["gate"], {}).get("feedback", [])
                  if f.get("round") == r["round"] and f.get("status") in ("approved", "changes")]
            if not fb:
                continue
            human_ok = fb[-1]["status"] == "approved"
            total += 1
            agreed += human_ok == r["ok"]
            rows.append({"pid": p.pid, "gate": r["gate"], "round": r["round"], "critic_ok": r["ok"],
                         "human_ok": human_ok})
    return {"compared": total, "agreed": agreed,
            "agreement": round(agreed / total, 2) if total else None, "rows": rows}

