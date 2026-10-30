"""Background jobs for slow work (ingest, render, motion, scribble, gen).

A job is a JSON file under ``projects/<pid>/jobs/``; it runs in a detached
``python -m opencine job-run <pid> <job_id>`` process, so state lives on disk and survives the
MCP server restarting. Callers usually ``submit`` then ``wait`` a bounded time and hand
back the ``job_id`` if the job is still running.
"""

from __future__ import annotations

import importlib
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencine.core.project import Project

# kind → "module:function"; the function is called as fn(project, **params) and returns JSON-able
HANDLERS = {
    "ingest": "opencine.core.jobs:_ingest",
    "render": "opencine.core.jobs:_render",
    "motion": "opencine.core.jobs:_motion",
    "scribble": "opencine.core.jobs:_scribble",
    "gen_image": "opencine.core.jobs:_gen_image",
    "gen_video": "opencine.core.jobs:_gen_video",
    "gen_edit": "opencine.core.jobs:_gen_edit",
    "gen_enhance": "opencine.core.jobs:_gen_enhance",
    "gen_face": "opencine.core.jobs:_gen_face",
    "transcribe": "opencine.core.jobs:_transcribe",
    "tts": "opencine.core.jobs:_tts",
    "music_gen": "opencine.core.jobs:_music_gen",
    "critic": "opencine.core.jobs:_critic",
}


def _dir(project: Project) -> Path:
    d = project.root / "jobs"
    d.mkdir(exist_ok=True)
    return d


def _write(project: Project, job: dict[str, Any]) -> None:
    path = _dir(project) / f"{job['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2, default=str), "utf-8")
    os.replace(tmp, path)


def pid_alive(pid: int | None) -> bool:
    """Is process ``pid`` still running? (``os.kill(pid, 0)`` would *terminate* it on Windows.)"""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: exists, owned by someone else
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read(project: Project, job_id: str) -> dict[str, Any]:
    path = _dir(project) / f"{job_id}.json"
    if not path.exists():
        raise KeyError(f"no job {job_id!r} in project {project.pid}")
    return json.loads(path.read_text("utf-8"))


def status(project: Project, job_id: str) -> dict[str, Any]:
    """Job record; a queued/running job whose process has died is marked ``failed``."""
    job = _read(project, job_id)
    if job["status"] in ("queued", "running") and "pid" in job and not pid_alive(job["pid"]):
        job = _read(project, job_id)  # it may have finished between the read and the check
        if job["status"] in ("queued", "running"):
            job.update(status="failed", finished=datetime.now().isoformat(timespec="seconds"),
                       error=f"job process {job['pid']} died; see jobs/{job_id}.log")
            _write(project, job)
    return job


def submit(project: Project, kind: str, params: dict[str, Any], *, detach: bool = True) -> str:
    if kind not in HANDLERS:
        raise ValueError(f"unknown job kind {kind!r}")
    job_id = f"{kind}-{datetime.now():%H%M%S}-{secrets.token_hex(2)}"
    _write(project, {"id": job_id, "kind": kind, "params": params, "status": "queued",
                     "created": datetime.now().isoformat(timespec="seconds")})
    if not detach:
        run(project, job_id)
        return job_id
    cmd = [sys.executable, "-m", "opencine", "job-run", project.pid, job_id]
    env = {**os.environ, "OPENCINE_PROJECTS_DIR": str(project.root.parent.resolve())}
    kwargs: dict[str, Any] = {"stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env}
    if os.name == "nt":  # a hidden console, not none: a detached job's ffmpeg calls would each open a window
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    with (_dir(project) / f"{job_id}.log").open("w", encoding="utf-8") as log:  # child keeps its copy
        child = subprocess.Popen(cmd, stdout=log, **kwargs)
    # record the launcher pid so a job that dies before ``run`` starts is noticed by ``status``
    job = _read(project, job_id)
    if job["status"] == "queued":
        _write(project, job | {"pid": child.pid})
    return job_id


def run(project: Project, job_id: str) -> dict[str, Any]:
    job = _read(project, job_id)
    job.update(status="running", started=datetime.now().isoformat(timespec="seconds"), pid=os.getpid())
    _write(project, job)
    mod, fn = HANDLERS[job["kind"]].split(":")
    try:
        job["result"] = getattr(importlib.import_module(mod), fn)(project, **job["params"])
        job["status"] = "done"
    except Exception as e:  # recorded for the caller; the job process exits non-zero
        job.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    job["finished"] = datetime.now().isoformat(timespec="seconds")
    _write(project, job)
    return job


def wait(project: Project, job_id: str, timeout_s: float, poll_s: float = 0.5) -> dict[str, Any]:
    """Poll until the job finishes or ``timeout_s`` passes; returns the latest status."""
    deadline = time.monotonic() + timeout_s
    while True:
        job = status(project, job_id)
        if job["status"] in ("done", "failed") or time.monotonic() >= deadline:
            return job
        time.sleep(poll_s)


# ---- handlers ---------------------------------------------------------------
def _ingest(project: Project, force: bool = False) -> Any:
    from opencine.index.pipeline import ingest

    return _jsonable(ingest(project, force=force))


def _render(project: Project, version: int | None = None, preset: str = "preview",
            aspect: str = "9:16") -> Any:
    from opencine.plan.store import load_plan
    from opencine.render.compile import render

    v, plan = load_plan(project, "editplan", version)
    if aspect == "16:9":
        from opencine.plan.landscape import derive

        plan = derive(project, plan)
    return {"version": v, "aspect": aspect, **_jsonable(render(project, plan, preset))}


def _motion(project: Project, code: str | None = None, version: int | None = None,
            preset: str = "preview", marks: dict | None = None, behind: dict | None = None) -> Any:
    from opencine.motion.service import render_motion

    return render_motion(project, code, version, preset, marks, behind)


def _scribble(project: Project, items: list[dict] | None = None, version: int | None = None,
              preset: str = "preview") -> Any:
    from opencine.scribble.service import render_scribble

    return render_scribble(project, items, version, preset)


def _gen_image(project: Project, slots: list[dict], provider: str | None = None) -> Any:
    from opencine.gen.service import ImageSlot, gen_images

    return _jsonable(gen_images(project, [ImageSlot.model_validate(s) for s in slots], provider))


def _gen_video(project: Project, slots: list[dict], provider: str | None = None) -> Any:
    from opencine.gen.service import VideoSlot, gen_videos

    return _jsonable(gen_videos(project, [VideoSlot.model_validate(s) for s in slots], provider))


def _gen_edit(project: Project, slots: list[dict], provider: str | None = None) -> Any:
    from opencine.gen.service import EditSlot, gen_edits

    return _jsonable(gen_edits(project, [EditSlot.model_validate(s) for s in slots], provider))


def _gen_enhance(project: Project, clips: list[str], short: int | None = None,
                 provider: str | None = None) -> Any:
    from opencine.gen.service import gen_enhance

    return _jsonable(gen_enhance(project, clips, short, provider))


def _gen_face(project: Project, clips: list[str], ref: str | None = None, details: str = "",
              force: bool = False, provider: str | None = None) -> Any:
    from opencine.gen.service import gen_face

    return _jsonable(gen_face(project, clips, ref, details, force, provider))


def _transcribe(project: Project, language: str | None = None) -> Any:
    from opencine.speech.service import transcribe

    return transcribe(project, language)


def _tts(project: Project, lines: list[dict]) -> Any:
    from opencine.speech.tts import synthesize

    return synthesize(project, lines)


def _music_gen(project: Project, prompt: str, seconds: float, bpm: int | None = None,
               key: str | None = None, seed: int | None = None) -> Any:
    from opencine.music.gen import generate

    return generate(project, prompt, seconds, bpm, key, seed)


def _critic(project: Project, a: int | str, b: int | str, feedback: str = "",
            gate: str | None = None) -> Any:
    from opencine.review.critic import compare

    return compare(project, a, b, feedback, gate)


def _jsonable(obj: Any) -> Any:

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
