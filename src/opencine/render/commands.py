"""CLI: ``opencine plan validate|solve|match|schema``, ``opencine render``, ``opencine beats``, ``opencine sfx``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer


def register(app: typer.Typer) -> None:
    plan_app = typer.Typer(help="Validate plans and export their JSON Schema.", no_args_is_help=True)
    app.add_typer(plan_app, name="plan")

    @plan_app.command("validate")
    def plan_validate(
        file: Annotated[Path, typer.Argument(help="EditPlan JSON file.")],
        pid: Annotated[str | None, typer.Option(help="Project id: also check sources and files.")] = None,
    ) -> None:
        """Validate an EditPlan; exits 1 if there are errors."""
        from pydantic import ValidationError

        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.plan.schema import EditPlan
        from opencine.plan.validate import errors, validate

        try:
            plan = EditPlan.model_validate_json(file.read_text("utf-8"))
        except ValidationError as e:
            echo_json({"ok": False, "issues": [
                {"level": "error", "path": ".".join(map(str, err["loc"])), "message": err["msg"]}
                for err in e.errors()]})
            raise typer.Exit(1) from None
        issues = validate(plan, Project.load(pid) if pid else None)
        bad = errors(issues)
        echo_json({"ok": not bad, "total_frames": plan.total_frames,
                   "duration_s": round(plan.total_frames / float(plan.fps), 3),
                   "issues": [i.model_dump() for i in issues]})
        if bad:
            raise typer.Exit(1)

    @plan_app.command("solve")
    def plan_solve(
        pid: str,
        version: Annotated[int | None, typer.Option(help="ShotPlan version (default latest).")] = None,
        lock: Annotated[list[int] | None, typer.Option(help="Slot to keep from the base EditPlan (repeatable).")] = None,
        target_s: Annotated[float | None, typer.Option(help="Length in seconds (default the Brief's).")] = None,
    ) -> None:
        """Solve a ShotPlan into a draft EditPlan (saved as a new version)."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.plan.solve import solve_and_save

        echo_json(solve_and_save(Project.load(pid), version, lock=lock, target_s=target_s))

    @plan_app.command("match")
    def plan_match(
        pid: str,
        from_scene: Annotated[str | None, typer.Option("--from", help="Only cuts out of this scene.")] = None,
        kind: Annotated[str | None, typer.Option(help="color | shape | light | motion | position")] = None,
        top: Annotated[int, typer.Option(help="How many pairs.")] = 10,
    ) -> None:
        """Rank match-cut pairs (scene@s → scene@s) from the frame samples."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.plan.match import candidates

        echo_json(candidates(Project.load(pid), from_scene=from_scene, kind=kind, top=top))

    sfx_app = typer.Typer(help="Sound effects: the synthesised set and imported packs.", no_args_is_help=True)
    app.add_typer(sfx_app, name="sfx")

    @sfx_app.command("init")
    def sfx_init(
        directory: Annotated[Path | None, typer.Option("--dir", help="Where to write (default OPENCINE_SFX_DIR).")] = None,
        force: Annotated[bool, typer.Option(help="Rewrite existing files.")] = False,
    ) -> None:
        """Write whoosh, swish, pop, click, tick, riser, impact, slice, scribble (.wav) to a folder, to
        listen to or replace with real recordings of the same name."""
        from opencine.cli import echo_json
        from opencine.core.config import get_settings
        from opencine.music.sfx import write_set

        d = directory or get_settings().sfx_dir
        if d is None:
            raise typer.BadParameter("give --dir or set OPENCINE_SFX_DIR")
        echo_json({"written": [str(p) for p in write_set(Path(d), force=force)]})

    @sfx_app.command("list")
    def sfx_list(
        pid: str,
        query: Annotated[str | None, typer.Option(help="Words that must match name or tags.")] = None,
    ) -> None:
        """Sounds a project's plans can name, and where each comes from."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.music.sfx import library

        echo_json(library(Project.load(pid), query))

    @sfx_app.command("import")
    def sfx_import(
        src: Annotated[Path, typer.Argument(help="A pack: folder or .zip of wav/mp3/ogg/flac/m4a.")],
        pack: Annotated[str, typer.Option(help="Pack name; sounds are named <pack>/<file>.")],
        license: Annotated[str, typer.Option(help="e.g. CC0, Sonniss-GDC (kept in the index).")],
        source: Annotated[str | None, typer.Option(help="Where it came from (URL).")] = None,
        directory: Annotated[Path | None, typer.Option("--dir", help="Library (default OPENCINE_SFX_DIR).")] = None,
    ) -> None:
        """Import a sound pack into the shared library: 48 kHz stereo WAV at -1 dBFS, tagged."""
        from opencine.cli import echo_json
        from opencine.music.sfx import import_pack

        echo_json(import_pack(src, pack, license=license, source=source, dest=directory))

    tools_app = typer.Typer(help="Local ncnn/Vulkan tools for final renders (ESRGAN, RIFE).",
                            no_args_is_help=True)
    app.add_typer(tools_app, name="tools")

    @tools_app.command("fetch")
    def tools_fetch(tool: Annotated[str, typer.Argument(help="esrgan | rife")]) -> None:
        """Download the pinned Windows release into OPENCINE_TOOLS_DIR (rife: ~430 MB)."""
        from opencine.cli import echo_json
        from opencine.render.ncnn import RELEASES, fetch

        if tool not in RELEASES:
            raise typer.BadParameter("tool must be esrgan or rife")
        echo_json(fetch(tool))

    @tools_app.command("list")
    def tools_list() -> None:
        """Which tools final renders will use."""
        from opencine.cli import echo_json
        from opencine.render.ncnn import RELEASES, binary

        echo_json({t: binary(t) for t in RELEASES})

    @plan_app.command("schema")
    def plan_schema(kind: Annotated[str, typer.Argument(help="brief | shotplan | editplan")]) -> None:
        """Print the JSON Schema of a plan kind."""
        from opencine.cli import echo_json
        from opencine.plan.schema import json_schema

        if kind not in ("brief", "shotplan", "editplan"):
            raise typer.BadParameter("kind must be brief, shotplan or editplan")
        echo_json(json_schema(kind))  # type: ignore[arg-type]

    @app.command("render")
    def render_cmd(
        pid: str,
        plan_file: Annotated[Path, typer.Argument(help="EditPlan JSON file.")],
        preset: Annotated[str, typer.Option(help="preview (720×1280) | final (1080×1920)")] = "preview",
        out: Annotated[Path | None, typer.Option(help="Output mp4 (default renders/<hash>-<preset>.mp4).")] = None,
    ) -> None:
        """Render an EditPlan."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.plan.schema import EditPlan
        from opencine.render.compile import PlanInvalid, render

        if preset not in ("preview", "final"):
            raise typer.BadParameter("preset must be preview or final")
        p: Literal["preview", "final"] = preset  # type: ignore[assignment]
        plan = EditPlan.model_validate_json(plan_file.read_text("utf-8"))
        try:
            res = render(Project.load(pid), plan, p, out)
        except PlanInvalid as e:
            echo_json({"ok": False, "issues": [i.model_dump() for i in e.issues]})
            raise typer.Exit(1) from None
        echo_json(res.model_dump(mode="json"))

    @app.command("beats")
    def beats_cmd(
        audio: Path,
        out: Annotated[Path | None, typer.Option(help="Output JSON (default <audio>.beats.json).")] = None,
    ) -> None:
        """Detect beats → beats.json [{t, strength}]."""
        from opencine.cli import echo_json
        from opencine.music.beats import load_beats, write_beats

        path = write_beats(audio, out)
        beats = load_beats(path)
        echo_json({"path": str(path), "count": len(beats),
                   "bpm_est": round(60 * (len(beats) - 1) / (beats[-1].t - beats[0].t), 1)
                   if len(beats) > 1 else None})
