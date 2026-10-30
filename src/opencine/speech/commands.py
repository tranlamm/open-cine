"""``opencine speech status`` · ``opencine transcribe`` · ``opencine transcript`` · ``opencine tts``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def register(app: typer.Typer) -> None:
    speech = typer.Typer(help="Self-hosted STT/TTS gateways.", no_args_is_help=True)
    app.add_typer(speech, name="speech")

    @speech.command("status")
    def status_cmd() -> None:
        """Health/readiness of the STT and TTS gateways."""
        from opencine.cli import echo_json
        from opencine.speech.service import status

        echo_json(status())

    @app.command("transcribe")
    def transcribe_cmd(pid: str, language: Annotated[str | None, typer.Option(
            help="en, vi… (default OPENCINE_SPEECH_LANGUAGE).")] = None) -> None:
        """Transcribe every clip (word timestamps; cached) and refresh L0."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.speech.service import transcribe

        echo_json(transcribe(Project.load(pid), language))

    @app.command("transcript")
    def transcript_cmd(
        pid: str,
        ref: Annotated[str | None, typer.Argument(help="C01 (clip) or S03 (scene); all by default.")] = None,
        words: Annotated[bool, typer.Option(help="Add word@seconds lines.")] = False,
    ) -> None:
        """Print the transcript with timestamps (clip seconds)."""
        from opencine.core.project import Project
        from opencine.speech.transcribe import transcript_text

        typer.echo(transcript_text(Project.load(pid), ref, words))

    @app.command("tts")
    def tts_cmd(
        pid: str,
        text: Annotated[str | None, typer.Option(help="One line to say.")] = None,
        id: Annotated[str, typer.Option("--id", help="Line id → audio/voice/<id>.wav.")] = "v01",
        language: str | None = None,
        lines: Annotated[Path | None, typer.Option(
            help='JSON list of {"id", "text", "language"?} (one batch request).')] = None,
    ) -> None:
        """Synthesise voice-over line(s) into audio/voice/."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.speech.tts import synthesize

        if lines:
            items = json.loads(lines.read_text("utf-8"))
        elif text:
            items = [{"id": id, "text": text, "language": language}]
        else:
            raise typer.BadParameter("give --text or --lines file.json")
        echo_json(synthesize(Project.load(pid), items))
