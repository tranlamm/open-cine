"""CLI: ``opencine scribble`` (the MCP tool from a shell; items in a JSON file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def register(app: typer.Typer) -> None:
    @app.command("scribble")
    def scribble_cmd(
        pid: str,
        items: Annotated[Path | None, typer.Argument(help='JSON file: a list of items or {"items": [...]}; default the saved ones.')] = None,
        version: Annotated[int | None, typer.Option(help="EditPlan version (default latest).")] = None,
        preset: Annotated[str, typer.Option(help="preview | final")] = "preview",
        check: Annotated[bool, typer.Option(help="A contact sheet instead of the video.")] = False,
        times: Annotated[list[float] | None, typer.Option("--at", help="Sheet time in seconds (repeatable).")] = None,
    ) -> None:
        """Draw hand-drawn scribbles over a render (see the opencine-edit skill, scribble.md)."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.scribble.service import ScribbleError, render_scribble, sheet

        if preset not in ("preview", "final"):
            raise typer.BadParameter("preset must be preview or final")
        data = json.loads(items.read_text("utf-8")) if items else None
        if isinstance(data, dict):
            data = data.get("items")
        project = Project.load(pid)
        try:
            if check:
                refs, table = sheet(project, data, version, times)
                echo_json({"sheets": [r.path for r in refs], "items": table})
            else:
                echo_json(render_scribble(project, data, version, preset))
        except ScribbleError as e:
            raise typer.BadParameter(str(e)) from None
