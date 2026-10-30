"""CLI: ``opencine ingest`` · ``opencine scenes`` · ``opencine sheet`` · ``opencine frame``."""

from __future__ import annotations

from typing import Annotated

import typer

from opencine.core.project import Project


def register(app: typer.Typer) -> None:
    @app.command("ingest")
    def ingest_cmd(
        pid: str,
        force: Annotated[bool, typer.Option("--force", help="Ignore the cache, redo every step.")] = False,
    ) -> None:
        """Probe, proxy, detect scenes and build L0/L1 for a project."""
        from opencine.cli import echo_json
        from opencine.index.pipeline import ingest

        echo_json(ingest(Project.load(pid), force=force))

    @app.command("scenes")
    def scenes_cmd(pid: str) -> None:
        """Print the L0 scene table."""
        from opencine.index.table import scenes_md

        typer.echo(scenes_md(Project.load(pid)), nl=False)

    @app.command("sheet")
    def sheet_cmd(
        pid: str,
        scene: Annotated[str | None, typer.Option("--scene", help="Scene id for an L2 strip, e.g. S03.")] = None,
    ) -> None:
        """L1 overview sheet(s), or the L2 strip of one scene (image paths + token estimates)."""
        from opencine.cli import echo_json
        from opencine.index.sheets import overview, scene_strip

        proj = Project.load(pid)
        refs = [scene_strip(proj, scene.upper())] if scene else overview(proj)
        echo_json([r.model_dump() for r in refs])

    @app.command("frame")
    def frame_cmd(
        pid: str,
        ref: Annotated[str, typer.Argument(help="S03 · S03@2.5 · S03@00:02.5 (in scene) · C01@12.4 (in clip)")],
    ) -> None:
        """L3: one full frame (image path + token estimate)."""
        from opencine.cli import echo_json
        from opencine.index.sheets import frame_at

        echo_json(frame_at(Project.load(pid), ref).model_dump())
