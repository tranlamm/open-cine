"""Entry points shared by the CLI, MCP tools, jobs and ingest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencine.core.config import get_settings
from opencine.speech.client import SpeechUnavailable, SttClient, TtsClient

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.model import SceneIndex
    from opencine.index.steps import StepCache


def status() -> dict[str, Any]:
    out: dict[str, Any] = {"language": get_settings().speech_language}
    for name, cls in (("stt", SttClient), ("tts", TtsClient)):
        try:
            out[name] = cls.from_settings().status()
        except SpeechUnavailable as e:
            out[name] = {"configured": False, "detail": str(e)}
    return out


def transcribe(project: Project, language: str | None = None) -> dict[str, Any]:
    """Transcribe every clip, then rewrite L0 so scene lines carry what is said."""
    from opencine.index.model import load_index
    from opencine.index.table import scenes_md
    from opencine.speech.transcribe import transcribe_all

    res = transcribe_all(project, load_index(project), language=language)
    scenes_md(project)
    return {"pid": project.pid, **res}


def transcribe_on_ingest(project: Project, index: SceneIndex, steps: StepCache) -> dict[str, Any] | None:
    """Ingest's speech step: skipped when STT is unset or off; a down gateway never fails ingest."""
    from opencine.speech.transcribe import transcribe_all

    s = get_settings()
    if not s.stt_url or not s.stt_on_ingest:
        return None
    try:
        return transcribe_all(project, index, steps)
    except Exception as e:  # STT is optional: report, don't fail the ingest
        return {"error": f"{type(e).__name__}: {e}", "hint": "run transcribe(pid) when STT is up"}
