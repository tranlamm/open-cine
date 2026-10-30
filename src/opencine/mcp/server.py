"""MCP server ``opencine`` (stdio): the Director agent's door into the engine.

Tools return fast. Slow work (ingest, render, generation) runs as a background job: the tool
waits up to ``SLOW_WAIT_S``, then returns ``{"job_id", "status": "running"}`` for ``job_status``.
Image tools return the images inline (plus their paths) and charge the project's image budget.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent

from opencine.core import jobs
from opencine.core.ledger import IMAGE_REVIEW, IMAGE_VIEW, Ledger
from opencine.core.project import Project

SLOW_WAIT_S = 50

mcp = MCPServer(
    name="opencine",
    instructions=(
        "Open Cine video engine. Skills: opencine-edit (cut a short from footage), opencine-film "
        "(a film from a prompt, every shot generated), opencine-swap (the user's clip remade with "
        "another character). Look at footage in order: scenes (L0 text, free) → sheet_overview "
        "(L1) → sheet_scenes / sheet_scene (L2) → frame (L3); image tools spend a per-project "
        "budget, so read L0 first and keep what you saw in scene_notes. Plans are versioned: "
        "plan_save returns the version render takes; plan_solve turns a ShotPlan into a draft "
        "EditPlan. Check your own work before each gate: storyboard_sheet (G1), gen_sheet "
        "(G2/G3), review_sheet of the rendered preview (G4). review_open opens a gate: show its "
        "message and numbered files to the user, end the turn, then record their answer with "
        "review_decide. telegram_send (optional) posts a gate or the final to the user's Telegram "
        "chat. project_status says where a project stands. hf_quota and gen_check size a "
        "generation batch before it spends HF ZeroGPU quota."
    ),
)


def tool(fn):
    """Register ``fn`` as a tool; any exception reaches the agent as a ToolError with its reason
    (MCP hides unexpected exception text, but budget refusals and bad ids are what the agent needs)."""
    def reason(e: Exception) -> ToolError:
        return ToolError(f"{type(e).__name__}: {e}")

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as e:
                raise reason(e) from e
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as e:
                raise reason(e) from e
    mcp.tool()(wrapper)
    return fn


def _project(pid: str) -> Project:
    return Project.load(pid)


async def _run_job(project: Project, kind: str, params: dict[str, Any]) -> dict[str, Any]:
    job_id = jobs.submit(project, kind, params)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SLOW_WAIT_S
    while True:
        job = jobs.status(project, job_id)
        if job["status"] in ("done", "failed") or loop.time() >= deadline:
            return _job_view(job)
        await asyncio.sleep(0.5)


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    view = {"job_id": job["id"], "kind": job["kind"], "status": job["status"]}
    if job["status"] == "done":
        view["result"] = job["result"]
    elif job["status"] == "failed":
        view["error"] = job["error"]
    else:
        view["hint"] = "still running; call job_status(pid, job_id) later"
    return view


Shown = list[TextContent | Image]


def _grant(project: Project, refs: list[Any], kind: str = IMAGE_VIEW,
           extra: dict[str, Any] | None = None) -> Shown:
    """Charge the budget (all or nothing), then return the details as JSON and the images inline."""
    led = Ledger(project)
    entries = led.grant_images([(ref.path, ref.tokens) for ref in refs], kind=kind)
    images = [{"path": ref.path, "width": ref.width, "height": ref.height, "tokens": ref.tokens}
              | ({"reused": True} if e.get("reused") else {})  # same file granted before: free
              for ref, e in zip(refs, entries, strict=True)]
    count, tokens = led.image_usage(kind)
    view = {"images": images, "budget_used": {"images": count, "tokens": tokens}} | (extra or {})
    text = json.dumps(view, ensure_ascii=False, indent=1, default=str)
    return [TextContent(type="text", text=text), *(Image(path=ref.path) for ref in refs)]


# ---- project & ingest -------------------------------------------------------
@tool
def project_create(paths: list[str] | None = None, name: str | None = None,
                   rate: str | None = None, mode: Literal["edit", "film", "swap"] = "edit") -> dict:
    """Create a project from video/still files or folders (none is fine). mode="film": a film from
    a prompt (24 fps unless `rate`; media are references). mode="swap": the clip remade with
    another character (stills are the character). Returns pid, clip ids (C01…) and still ids (I01…)."""
    proj = Project.create(paths or [], name=name, rate=rate, mode=mode)
    return {"pid": proj.pid, "rate": proj.meta.rate, "mode": proj.meta.mode,
            "sources": [{"id": s.id, "original": s.original} for s in proj.meta.sources],
            "images": [{"id": s.id, "original": s.original} for s in proj.meta.images]}


@tool
def project_add(pid: str, paths: list[str]) -> dict:
    """Add videos/stills (files or folders) the user gives later; then run ingest again (existing
    scene ids stay, new scenes are appended). The same file twice is added once."""
    proj = _project(pid)
    added = proj.add_sources(paths)
    return {"pid": proj.pid, "added": [{"id": s.id, "original": s.original} for s in added],
            "next": "ingest(pid)"}


@tool
async def ingest(pid: str, force: bool = False) -> dict:
    """Probe, proxy, scene-detect, sample frames and build L0/L1 (cached). May return job_id."""
    return await _run_job(_project(pid), "ingest", {"force": force})


@tool
def job_status(pid: str, job_id: str) -> dict:
    """Status/result of a background job (ingest, render, gen_*, motion, scribble…)."""
    return _job_view(jobs.status(_project(pid), job_id))


# ---- looking at footage (L0 → L3) ---------------------------------------------
@tool
def scenes(pid: str) -> str:
    """L0: one text line per scene (range, length, motion, peak = motion peak in seconds into the
    scene, cam, subj, colour, audio peak, brightness, sharpness), your notes and speech under it. Free."""
    from opencine.index.table import scenes_md

    return scenes_md(_project(pid))


@tool
def sheet_overview(pid: str) -> Shown:
    """L1: the overview grid, one labelled thumbnail per scene. Costs image budget."""
    from opencine.index.sheets import overview

    proj = _project(pid)
    return _grant(proj, overview(proj))


@tool
def sheet_scene(pid: str, sid: str) -> Shown:
    """L2: 4–6 timestamped frames of one scene (sid="S03"). Costs image budget."""
    from opencine.index.sheets import scene_strip

    proj = _project(pid)
    return _grant(proj, [scene_strip(proj, sid)])


@tool
def sheet_scenes(pid: str, sids: list[str]) -> Shown:
    """L2 for up to 8 scenes in ONE image: start · motion peak · end of each, labelled with refs
    (S03@00:01.25). Prefer it to one sheet_scene per candidate. Costs image budget (~1–3k tokens)."""
    from opencine.index.strips import scenes_sheet

    proj = _project(pid)
    return _grant(proj, [scenes_sheet(proj, sids)])


@tool
def frame(pid: str, ref: str) -> Shown:
    """L3: one full frame. ref = "S03@2.5" (seconds into the scene), "C01@12.4" (into the clip) or "S03"."""
    from opencine.index.sheets import frame_at

    proj = _project(pid)
    return _grant(proj, [frame_at(proj, ref)])


@tool
def scene_notes(pid: str, notes: dict[str, str]) -> dict:
    """Save one-line notes per scene, e.g. {"S03": "CU hand squeezing lemon, juice burst @2.1s;
    use 1.5–3.0s"}: shot size, subject, action/direction, usable range. L0 shows them from now on,
    in later sessions too, so you need not look again. "" deletes a note. Free."""
    from opencine.index.table import save_notes

    saved = save_notes(_project(pid), notes)
    return {"notes": len(saved), "updated": sorted(k.strip().upper() for k in notes)}


@tool
def sheet_images(pid: str) -> Shown:
    """L1 for the stills the user gave (I01…): one labelled grid. Costs image budget."""
    from opencine.index.sheets import images_sheet

    proj = _project(pid)
    return _grant(proj, [images_sheet(proj)])


# ---- speech (optional STT/TTS gateways) ---------------------------------------------
@tool
def speech_status() -> dict:
    """Whether the STT/TTS gateways are configured and answering."""
    from opencine.speech.service import status

    return status()


@tool
async def transcribe(pid: str, language: str | None = None) -> dict:
    """Speech → text with word times for every clip (cached); refreshes L0. Ingest already does it
    when STT answers. May return job_id."""
    return await _run_job(_project(pid), "transcribe", {"language": language})


@tool
def transcript(pid: str, ref: str | None = None, words: bool = False) -> str:
    """Timed transcript of a clip ("C01"), a scene ("S03": seconds into the scene, as in_s/out_s
    take) or all clips. words=True adds word@seconds (/fN = frame) for cutting on words and
    timing text. Free."""
    from opencine.speech.transcribe import transcript_text

    return transcript_text(_project(pid), ref, words)


@tool
async def tts(pid: str, lines: list[dict]) -> dict:
    """Voice-over: [{id, text, language?}] → audio/voice/<id>.wav plus <id>.words.json (word
    times that EditPlan `captions` follow). Returns length (s, frames), word count and `timing`;
    place each line as an EditPlan sfx {file, at}. Cached."""
    return await _run_job(_project(pid), "tts", {"lines": lines})


# ---- music ------------------------------------------------------------------------
@tool
def music_add(pid: str, path: str) -> dict:
    """Add a music track (an audio file, or a video with sound): copied to music/ with its beat
    grid. Returns file (use as EditPlan music.file), BPM, beat count, first beats and strong-beat
    frames; plan_save hints then check cuts against the beats."""
    from opencine.music.library import add_music

    return add_music(_project(pid), path)


@tool
async def music_gen(pid: str, prompt: str, seconds: float, bpm: int | None = None,
                    key: str | None = None, seed: int | None = None) -> dict:
    """Instrumental background music for this edit (ACE-Step 1.5; outputs usable commercially) →
    music/gen-<hash>.wav with its beat grid, like music_add. prompt: English genre, mood,
    instruments, energy curve ("upbeat lo-fi hip hop, warm Rhodes, soft kick, builds at the end");
    seconds: timeline length + ~4 s; bpm: cutting tempo (None = model picks); key e.g. "A minor".
    Same inputs = same track; another seed = another take. Reserves ~1.5 × (20 + 0.6 × seconds) s
    of HF quota."""
    return await _run_job(_project(pid), "music_gen", {"prompt": prompt, "seconds": seconds,
                                                       "bpm": bpm, "key": key, "seed": seed})


@tool
def beats(pid: str, name: str) -> str:
    """Beat grid of a track in music/, 8 per line: seconds/frame, * = strong. Put an action's
    impact on a beat and the cut just before it. Free."""
    from opencine.music.library import beats_text

    return beats_text(_project(pid), name)


# ---- plans & render -------------------------------------------------------------
PlanKind = Literal["brief", "shotplan", "editplan"]


@tool
def plan_schema(kind: PlanKind) -> dict:
    """JSON Schema for a plan kind (brief | shotplan | editplan)."""
    from opencine.plan.schema import json_schema

    return json_schema(kind)


@tool
def plan_save(pid: str, kind: PlanKind, plan: dict) -> dict:
    """Validate and store a plan as a new version (`unchanged: true` = same as the latest). On
    errors nothing is saved and the errors come back. EditPlan times may be seconds (in_s/out_s,
    sfx at_s, text start_s/end_s). A saved EditPlan returns craft `hints` (hook, rhythm, jump
    cuts, text time, length): fix them or accept them."""
    from opencine.plan.store import PlanInvalid, save_plan

    try:
        return save_plan(_project(pid), kind, plan)
    except PlanInvalid as e:
        return {"saved": False, "errors": e.issues}


@tool
def plan_get(pid: str, kind: PlanKind, version: int | None = None) -> dict:
    """Load a stored plan (latest when version is omitted)."""
    from opencine.plan.store import load_plan, versions

    proj = _project(pid)
    v, plan = load_plan(proj, kind, version)
    return {"kind": kind, "version": v, "versions": versions(proj, kind),
            "plan": plan.model_dump(mode="json", by_alias=True)}


@tool
async def render(pid: str, version: int | None = None,
                 preset: Literal["preview", "final"] = "preview",
                 aspect: Literal["9:16", "16:9"] = "9:16") -> dict:
    """Render an EditPlan version (latest by default): preview = 720×1280, final = 1080×1920.
    aspect="16:9" makes a landscape cut of the same plan (only after G4, when asked). Final
    results list notes (upscaled / interpolated items, mask wipes). May return job_id."""
    return await _run_job(_project(pid), "render", {"version": version, "preset": preset,
                                                    "aspect": aspect})


# ---- self-review (review image budget) ------------------------------------------------
@tool
def review_sheet(pid: str, version: int | None = None, cuts: list[int] | None = None) -> Shown:
    """Your rendered preview before G4: frame 0 (hook) and one mid frame per EditPlan item,
    labelled "#n 1.4s"; cuts=[n,…] adds the frames around the cut into item n. Also a timeline
    table and craft hints. Costs the review budget (~1.6k tokens per sheet)."""
    from opencine.plan.store import load_plan, versions
    from opencine.plan.validate import craft_hints
    from opencine.review.selfcheck import review_sheet as sheet

    proj = _project(pid)
    refs, table = sheet(proj, version, cuts)
    _, plan = load_plan(proj, "editplan", version)
    brief = load_plan(proj, "brief")[1] if versions(proj, "brief") else None
    hints = [f"{i.path}: {i.message}" for i in craft_hints(plan, proj, brief)]
    return _grant(proj, refs, IMAGE_REVIEW, {"timeline": table, "hints": hints})


@tool
def storyboard_sheet(pid: str, version: int | None = None) -> Shown:
    """One tile per ShotPlan slot ("#n hook GEN CU"): its footage moment or a generated slot's
    ref, plus a slot table and craft hints. Show it with G1. Costs the review budget."""
    from opencine.plan.store import load_plan, versions
    from opencine.plan.validate import shotplan_hints
    from opencine.review.selfcheck import storyboard_sheet as sheet

    proj = _project(pid)
    refs, table = sheet(proj, version)
    brief = load_plan(proj, "brief")[1] if versions(proj, "brief") else None
    hints = [f"{i.path}: {i.message}" for i in shotplan_hints(load_plan(proj, "shotplan", version)[1], brief)]
    return _grant(proj, refs, IMAGE_REVIEW, {"slots": table, "hints": hints})


@tool
def gen_sheet(pid: str, paths: list[str]) -> Shown:
    """Generated media before G2/G3: one tile per still, first · middle · last frame per clip,
    numbered in `paths` order (open the gate in the same order), with each variant's QC flags.
    Catch what CPU checks miss (extra fingers, melted objects, wrong subject). Costs the review budget."""
    from opencine.review.selfcheck import gen_sheet as sheet

    proj = _project(pid)
    refs, table = sheet(proj, paths)
    return _grant(proj, refs, IMAGE_REVIEW, {"items": table})


@tool
def plan_solve(pid: str, version: int | None = None, lock: list[int] | None = None,
               base: int | None = None, target_s: float | None = None) -> dict:
    """ShotPlan (latest by default) → draft EditPlan, saved as a new version: lengths within each
    slot's dur_s summing to the target, cuts on beats, the sharpest windows, approved generated
    media, tracked 9:16 crops, transitions/speed/text/sfx. lock=[slot,…] keeps those items from
    the latest (or `base`) EditPlan; hand-edited items are kept too. Returns version, report,
    warnings and hints."""
    from opencine.plan.solve import solve_and_save

    return solve_and_save(_project(pid), version, lock=lock, base=base, target_s=target_s)


MatchKind = Literal["color", "shape", "light", "motion", "position"]


@tool
def match_candidates(pid: str, from_scene: str | None = None, kind: MatchKind | None = None,
                     top: int = 5) -> dict:
    """Match-cut pairs, best first: {from: "S03@4.2", to: "S11@0.6", kind, score 0–1}; the cut
    carries over colour, shape, light layout, motion direction or subject position. Put a pair in
    the ShotPlan as the two slots' `at` with `match: {kind}` on the first. Needs match samples
    from ingest."""
    from opencine.plan.match import candidates

    found = candidates(_project(pid), from_scene=from_scene, kind=kind, top=max(1, min(top, 30)))
    return {"pairs": found} if found else {"pairs": [], "note": "no samples with match data: re-ingest"}


@tool
def reframe(pid: str, scene: str, in_s: float, out_s: float, speed: float = 1.0) -> dict:
    """A 9:16 crop following the scene's subject (face, else motion, else saliency) for in_s→out_s
    (seconds into the scene) at `speed`: paste it as the item's `crop`. null when the source is
    already 9:16 or nothing was tracked."""
    from opencine.core.time import to_frame
    from opencine.index.model import load_index
    from opencine.plan.reframe import reframe as follow

    project = _project(pid)
    sc = load_index(project).scene(scene.strip().upper())
    a, b = to_frame(in_s, project.rate, "nearest"), to_frame(out_s, project.rate, "nearest")
    if b <= a:
        raise ValueError("out_s must be after in_s")
    n = max(1, round((b - a) / speed))
    return {"crop": follow(project, sc.clip, sc.start_frame, a, b, n, speed=speed)}


@tool
def sfx_list(pid: str, query: str | None = None, limit: int = 40) -> dict:
    """Sounds a plan can name (slot `sfx`, EditPlan `sfx[].file`): the project's sfx/, the
    OPENCINE_SFX_DIR library (`<pack>/<name>`, with tags), then the synthesised set (whoosh,
    swish, pop, click, tick, riser, impact, slice, scribble). query: English words that must all
    match ("knife chop"); `total` counts every match."""
    from opencine.music.sfx import library

    return library(_project(pid), query, limit)


@tool
async def motion(pid: str, code: str | None = None, marks: dict[str, float | str] | None = None,
                 behind: dict[str, dict] | None = None, version: int | None = None,
                 preset: Literal["preview", "final"] = "preview", check: bool = False,
                 times: list[float] | None = None) -> dict | Shown:
    """Motion layer written as code (one TSX file) drawn over the rendered EditPlan: kinetic
    titles, counters, bursts, flashes, shockwaves, speed lines. `code` imports from
    'opencine-motion' (Text, Counter, Burst, Flash, Shock, SpeedLines, Show, Behind, useMotion,
    beatPulse, nextBeat), may export `spans = [[start_s, end_s], …]` (where it draws; omitted =
    whole video, slower) and default-exports the layer. Times are timeline seconds, x/y 0–1 of the
    frame, sizes fractions of its height. `marks`: named moments for the code ({"hit": "S03@2.1",
    "price": "say:49 dollars", "end": 12.4}). Text behind a subject: wrap it in
    `<Behind matte="me">` and pass behind={"me": {"on": "person", "from": 0.4, "to": 2.6}}.
    check=true returns a contact sheet (review budget) instead of rendering. Otherwise renders
    <render>-motion.mp4 at `preset`. code omitted = the layer saved for this version, so after G4:
    motion(pid, preset="final")."""
    proj = _project(pid)
    if check:
        from opencine.motion.service import sheet

        refs, table = await asyncio.to_thread(sheet, proj, code, version, times, marks, behind)
        return _grant(proj, refs, IMAGE_REVIEW, {"layer": table})
    return await _run_job(proj, "motion", {"code": code, "version": version, "preset": preset, "marks": marks,
                                           "behind": behind})


@tool
async def scribble(pid: str, items: list[dict] | None = None, version: int | None = None,
                   preset: Literal["preview", "final"] = "preview", check: bool = False,
                   times: list[float] | None = None) -> dict | Shown:
    """Hand-drawn marks that follow the things in the rendered EditPlan frame by frame (only when
    the user asked for scribble). Item types: outline, fill, ring, trail, doodle, scatter, frame
    (fields: see the skill's scribble guide). Common fields: on (English words naming the thing;
    found once per shot, its mask tracked to the cut), each, at (timeline seconds, "S03@2.1" or
    "say:<word>[#n]"), dur_s, look pen|marker|neon|fire, color, stroke, draw_s, exit
    fade|erase|cut; x/y are 0–1 of the frame. Items whose thing is not found are dropped (noted).
    check=true returns a contact sheet (review budget) instead of rendering. Otherwise renders
    <render>-scribble.mp4 at `preset` (over the motion layer when there is one). items omitted =
    the ones saved for this version, so after G4: scribble(pid, preset="final")."""
    proj = _project(pid)
    if check:
        from opencine.scribble.service import sheet

        refs, table = await asyncio.to_thread(sheet, proj, items, version, times)
        return _grant(proj, refs, IMAGE_REVIEW, {"items": table})
    return await _run_job(proj, "scribble", {"items": items, "version": version, "preset": preset})


# ---- review ---------------------------------------------------------------------
Gate = Literal["G1", "G2", "G3", "G4"]


@tool
def review_open(pid: str, gate: Gate, summary: str, media: list[str] | None = None) -> dict:
    """Open the next round of a review gate. Returns `message` (gate, round, your summary, the
    cost line at G1–G3, a reply hint) and the numbered `items` (paths of `media`, in order). Show
    both to the user in chat, then end your turn and wait for their reply."""
    from opencine.review.service import open_review

    return open_review(_project(pid), gate, summary, media or [])


@tool
def telegram_send(text: str = "", paths: list[str] | None = None, pid: str | None = None) -> dict:
    """Post a message and files to the user's Telegram chat (optional: needs
    OPENCINE_TELEGRAM_BOT_TOKEN and OPENCINE_TELEGRAM_CHAT_ID; errors when they are unset). For
    a gate: text = review_open's `message`, paths = the items' `abs` paths in order (several files
    are captioned 1, 2, 3…). For the final: the final path. Relative paths resolve inside pid's
    project. Photos and videos go as albums; uploads ≤ 50 MB, bigger files come back in `skipped`."""
    from opencine.review.telegram import send

    root = _project(pid).root if pid else None
    files = [Path(p) if Path(p).is_absolute() or root is None else root / p for p in paths or []]
    return send(text, files)


@tool
async def critic_compare(pid: str, a: int | str, b: int | str, feedback: str = "",
                         gate: Gate | None = None) -> dict:
    """Optional second opinion from another model family (Codex CLI, OPENCINE_CRITIC=codex):
    compares a (earlier EditPlan version or generated file) with b (your new one), both shown as
    review sheets (versions need a rendered preview). feedback = what the user said about a.
    Returns winner a|b|tie, fixed[{point, ok, why}], regressions and ok; winner "skipped" when the
    critic is off or failed. Advice only: fix at most once on it. ~1–3 min."""
    return await _run_job(_project(pid), "critic", {"a": a, "b": b, "feedback": feedback, "gate": gate})


@tool
def review_decide(pid: str, gate: Gate, status: Literal["approved", "changes", "cancelled"],
                  note: str = "", picks: dict[str, str] | None = None) -> dict:
    """Record your reading of the user's reply before acting on it: approved (gate closed),
    changes (fix, then open a new round) or cancelled. At G2/G3 pass picks={slot: variant path}
    for the approved stills/clips (e.g. {"3": "gen/clips/3~4821.mp4"}); plan_solve uses them."""
    from opencine.gen.slots import approve
    from opencine.review.gates import record_feedback

    proj = _project(pid)
    approved = approve(proj, picks, gate) if picks else {}
    out = record_feedback(proj, gate, note or status, status) | (
        {"approved": approved} if approved else {})
    if gate == "G4" and status == "approved":  # learn the numbers of what was approved
        from opencine.plan.taste import learn

        try:
            out["taste"] = learn(proj)
        except Exception as e:  # noqa: BLE001 — never lose the decision over it
            out["taste"] = {"learned": False, "reason": f"{type(e).__name__}: {e}"}
    return out


@tool
def taste(template: str | None = None) -> str:
    """What the user liked across projects: your strongest noted rules and the numbers learned
    from approved edits for this template. Read it with the Brief. Free, ≤ ~300 tokens."""
    from opencine.plan.taste import summary

    return summary(template)


@tool
def taste_note(text: str = "", scope: str = "all", weight: int = 1, evidence: str = "",
               rule_id: str | None = None) -> dict:
    """Remember a general preference (not a one-off fix), e.g. "no flash transitions", "hook ≤ 1 s",
    "fewer than 3 labels". scope: all | a template name. Noting it again strengthens it; weight −1
    (or rule_id + −1) weakens it when the user contradicts it; at 0 it is forgotten.
    evidence: "<pid>@G4"."""
    from opencine.plan.taste import note

    return note(text, scope, weight, evidence, rule_id)


@tool
def project_status(pid: str) -> dict:
    """Where the project stands: gates, latest plan versions, generated slots (batches, variants,
    approved path), spend vs cap, image budgets, recent renders. Call it when a session resumes. Free."""
    from opencine.review.status import project_status as status

    return status(_project(pid))


@tool
def review_state(pid: str) -> dict:
    """State of every review gate (round, status, feedback)."""
    from opencine.review.service import review_state as state

    return state(_project(pid))


# ---- generation & costs -----------------------------------------------------------
Prov = Literal["hf"]


def _default_models(slots: list[dict], kind: str | None = "image") -> list[dict]:
    """Allow only the Director's models (``hf_space.DIRECTOR_MODELS``) for ``kind`` (None: each
    slot's ``mode``); the other models are for manual CLI runs."""
    from opencine.gen.providers.hf_space import DIRECTOR_MODELS

    for s in slots:
        if not (isinstance(s, dict) and s.get("model")):
            continue
        allowed = DIRECTOR_MODELS.get(kind or str(s.get("mode")), ())
        if str(s["model"]) not in allowed:
            choice = f"choose one of {', '.join(allowed)}" if allowed else "leave `model` out"
            raise ValueError(f"model {s['model']}: {choice} (other models are for manual "
                             "runs: opencine gen … --model)")
    return slots


@tool
async def gen_image(pid: str, slots: list[dict], provider: Prov | None = None) -> dict:
    """Generate stills with Qwen-Image-2.1 (2K: 1536×2688 for 9:16, 2688×1536 for 16:9), one per
    slot: {id, refs?, prompt, aspect="9:16", crop?, variants=1, seed?}. refs = paths or frame refs
    ("S03@2.5", full source resolution). With refs the English prompt is one edit instruction,
    operation first (a new angle/shot size/lens goes there too); name several refs <image1>,
    <image2>…, one ref "the image". No refs = text-to-image: describe the finished picture in one
    paragraph (subject, places across the frame, one lighting sentence, a composition sentence).
    crop={x,y,w,h} (fractions of refs[0]) keeps that window. Re-roll with a new seed (files are
    kept). Items: gen/stills/<id>~<seed>.png (+ <id>.png = latest) and QC flags (soft,
    colour_shift, blank). May return job_id."""
    return await _run_job(_project(pid), "gen_image",
                          {"slots": _default_models(slots), "provider": provider})


@tool
async def gen_video(pid: str, slots: list[dict], provider: Prov | None = None) -> dict:
    """Generate clips with MiniMax-H3: {id, first?, last?, refs?, prompt, seconds=5,
    aspect="9:16", seed?, canvas?, retime="conform", match_colour=true, trim_head?, model?}.
    From an approved still (first), first+last frame, or text alone; first/last may be frame refs.
    model: h3-turbo (default, 7 steps) | h3-turbo-hq (9 steps, ~+10 % quota) | h3 (28 steps,
    ~2–2.5× quota). refs = character/place pictures (≤ 3 with a first frame; "<sheet>#face" works):
    Ref2VA keeps them in every shot (h3-ref, 8 steps, ~1.5× quota | h3-ref-hq 16 steps); the
    prompt is then H3's six-field Ref2VA text, and with `first` that frame is <Picture 1>.
    seconds 4–15 (snapped to 17n+5 frames at 24 fps); canvas fast|mid|full (default full).
    Prompts in English, best as H3 Context-IR (integrated_multimodal_description with [Shot N]
    blocks, overall_soundscape, non_diegetic_music); several [Shot N] = a multi-shot clip.
    Output is conformed to the project rate, BT.709, colour-matched to `first`, muted unless the
    EditPlan item sets gain. Check flags: garbage or frozen → new seed; unwanted cut/multishot →
    rewrite as one shot. Items carry prompt hints and what this take changed. May return job_id."""
    return await _run_job(_project(pid), "gen_video",
                          {"slots": _default_models(slots, "video"), "provider": provider})


@tool
def gen_check(pid: str, kind: Literal["image", "video"], slots: list[dict],
              provider: Prov | None = None) -> dict:
    """Dry-run a gen_image / gen_video batch (same slots; nothing generated, free): per slot its
    prompt hints, what it changes from the slot's last take (change one axis per take), whether
    it is cached, and reserve_s (HF quota it reserves). `quota` places the calls on the daily
    quota left, account by account as the run will: fits, over (slots no account holds),
    left_after_s. Size the batch by it, not by hf_quota's left_s."""
    from opencine.gen.service import check_slots

    return check_slots(_project(pid), kind, _default_models(slots, kind), provider)


@tool
async def gen_edit(pid: str, slots: list[dict], provider: Prov | None = None) -> dict:
    """Video-to-video on a footage window (same length, motion, camera, timing): {id, source,
    mode, refs, prompt, subject="person", aspect="9:16", seed?, retime="interp", match_colour?,
    model?}. source = "S03@1.2-6.2", {scene, in_s, out_s, crop?, reframe} (≤ 15 s) or
    {file, in_s, out_s}. mode "swap": the person becomes the character of refs[0]; mode
    "restyle": the picture takes the look of refs[0] (the window's first frame restyled with
    gen_image). prompt (English): swap = 90–140 words describing the result; restyle = the style
    and scene. model (swap): default SCAIL-2 | "h3-swap" | "h3-swap-hq" (MiniMax-H3 character swap:
    refs[0] = any full-body picture of the character; prompt = the character head to toe as ONE
    comma list starting with the gender, e.g. "Female, tall, slim build, long wavy red hair,
    green silk dress, white sneakers"; subject = who to replace; windows ≤ 5 s hold identity
    best). h3-swap: 576×1024, ~575 s quota per 5 s; h3-swap-hq: 768×1344 two-pass, ~875 s.
    Output gen/clips/<id>~<seed>.mp4 at the project rate, no sound (the EditPlan item plays the
    footage's). Flags: length, garbage, frozen. May return job_id."""
    return await _run_job(_project(pid), "gen_edit",
                          {"slots": _default_models(slots, None), "provider": provider})


@tool
async def gen_enhance(pid: str, clips: list[str], short: Literal[1080, 1440] = 1080) -> dict:
    """Sharper, larger copy of finished generated clips (FlashVSR+ one-step super-resolution):
    same frames, timing and sound. Only when the user asks. clips = slot ids ("3") or paths.
    short 1080 → 1080×1920 for 9:16; 1440 → 1440 short side (~1.6× quota). Output
    gen/clips/<stem>.1080p.mp4 replaces its source as the slot's latest (and approved) clip; a
    saved EditPlan still names the old file. Quota at 1080: 5 s ≈ 108 s reserved, 10 s ≈ 182 s.
    Cached. May return job_id."""
    return await _run_job(_project(pid), "gen_enhance", {"clips": clips, "short": short})


@tool
async def gen_face(pid: str, clips: list[str], ref: str | None = None, details: str = "",
                   force: bool = False) -> dict:
    """Redraw the small, waxy face of finished generated clips at high resolution (MiniMax-H3
    face pass): the main face is tracked, redrawn from a square close-up (mouth held to the
    sound) and pasted back. For wide shots with faces under ~120 px. clips = slot ids or paths;
    ref = identity picture (default: the character sheet close-up, else the clip's clearest
    face); details = the person in a few English words. Clips whose face is already large
    (OPENCINE_HF_FACE_MAX_PX) are skipped unless force=true. Output gen/clips/<stem>.face.mp4
    replaces its source like gen_enhance; run it before gen_enhance. Quota: 5 s ≈ 440 s,
    10 s ≈ 640 s. Cached. May return job_id."""
    return await _run_job(_project(pid), "gen_face",
                          {"clips": clips, "ref": ref, "details": details, "force": force})


@tool
def gen_quota(provider: Prov | None = None) -> dict:
    """Quota used and remaining (+ reset time) for the image/video providers."""
    from opencine.gen.service import quota

    return quota(provider)


@tool
def hf_quota() -> dict:
    """HF ZeroGPU quota now (source "hub", or "local book" when the Hub does not answer): seconds
    left, reset time, and per call type (image, 5/10 s clip, Ref2VA clip, music, swap, restyle,
    enhance, face) the quota it needs and how many still fit. not_set lists calls whose Space is
    not set up (do not offer them). With several HF accounts the numbers are summed; one call
    never spans accounts, so use max_call_s and calls.*.fits. `overquota` says whether calls past
    the quota run on credits or are refused. Read it before a batch. Free."""
    from opencine.gen.providers.hf_space import quota_summary

    return quota_summary()


_setup: dict[str, Any] = {"status": "idle"}


def _run_setup(refresh: bool) -> None:
    from opencine.core.config import ENV_FILE, get_settings
    from opencine.gen.config import get_gen_settings
    from opencine.gen.providers.hf_setup import setup_spaces

    try:
        report = setup_spaces(None, refresh=refresh, env_file=ENV_FILE)
        _setup.update(status="done", report=report)
    except Exception as e:  # noqa: BLE001 — reported to the agent
        _setup.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        get_settings.cache_clear()
        get_gen_settings.cache_clear()


@tool
def hf_setup(refresh: bool = False) -> dict:
    """Create the user's private ZeroGPU Space duplicates (once per HF account, from HF_TOKEN /
    OPENCINE_HF_TOKENS) and save their ids to ~/.opencine/.env. Runs in the background (a few
    minutes); call again for progress. Needed once, when hf_quota lists calls under not_set.
    refresh=True re-applies the engine's patches to existing duplicates."""
    import threading

    if _setup["status"] == "running":
        return _setup
    _setup.clear()
    _setup.update(status="running")
    threading.Thread(target=_run_setup, args=(refresh,), daemon=True).start()
    return _setup


@tool
def costs(pid: str) -> dict:
    """Ledger summary: spend, quota per provider, images/tokens shown to the agent."""
    return Ledger(_project(pid)).summary()


def serve() -> None:
    mcp.run("stdio")
