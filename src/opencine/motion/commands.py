"""CLI: ``opencine motion`` (the MCP tool from a shell; the layer in a .tsx file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def register(app: typer.Typer) -> None:
    @app.command("motion")
    def motion_cmd(
        pid: str,
        code: Annotated[Path | None, typer.Argument(help="Layer .tsx file; default the saved one.")] = None,
        marks: Annotated[Path | None, typer.Option(help='JSON file of named moments {"hit": "S03@2.1"}.')] = None,
        behind: Annotated[Path | None, typer.Option(help='JSON file of mattes {"me": {"on": "person", "from": 0.4, "to": 2.6}}.')] = None,
        version: Annotated[int | None, typer.Option(help="EditPlan version (default latest).")] = None,
        preset: Annotated[str, typer.Option(help="preview | final")] = "preview",
        check: Annotated[bool, typer.Option(help="A contact sheet instead of the video.")] = False,
        times: Annotated[list[float] | None, typer.Option("--at", help="Sheet time in seconds (repeatable).")] = None,
    ) -> None:
        """Draw a motion layer (kinetic text, effects) over a render (see the opencine-edit skill, motion.md)."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.motion.service import MotionError, render_motion, sheet

        if preset not in ("preview", "final"):
            raise typer.BadParameter("preset must be preview or final")
        src = code.read_text("utf-8") if code else None
        named = json.loads(marks.read_text("utf-8")) if marks else None
        mattes = json.loads(behind.read_text("utf-8")) if behind else None
        project = Project.load(pid)
        try:
            if check:
                refs, table = sheet(project, src, version, times, named, mattes)
                echo_json({"sheets": [r.path for r in refs], "layer": table})
            else:
                echo_json(render_motion(project, src, version, preset, named, mattes))
        except MotionError as e:
            raise typer.BadParameter(str(e)) from None
