"""``opencine`` command line. Each package registers its own commands via ``register(app)``."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from opencine.core.project import Project, list_projects

app = typer.Typer(help="Open Cine: an agent-directed video engine (ingest, plan, render, generate).",
                  no_args_is_help=True, pretty_exceptions_show_locals=False)
project_app = typer.Typer(help="Create and inspect projects.", no_args_is_help=True)
app.add_typer(project_app, name="project")


def echo_json(data) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


@project_app.command("new")
def project_new(
    paths: Annotated[list[Path] | None, typer.Argument(
        help="Videos, stills or folders of them (none: a film from a prompt).")] = None,
    name: Annotated[str | None, typer.Option(help="Project id (default p<yymmdd>-<hex>).")] = None,
    rate: Annotated[str | None, typer.Option(
        help="Project frame rate, e.g. 30 or 30000/1001 (film default 24).")] = None,
    mode: Annotated[str, typer.Option(
        help="edit (cut the footage) | film (generated shots; media are references) | "
             "swap (the clip remade with another character; stills are the character).")] = "edit",
) -> None:
    """Create a project from source videos, or an empty film project."""
    if mode not in ("edit", "film", "swap"):
        raise typer.BadParameter("--mode is edit, film or swap")
    proj = Project.create(paths or [], name=name, rate=rate, mode=mode)  # type: ignore[arg-type]
    echo_json({"pid": proj.pid, "root": str(proj.root), "rate": proj.meta.rate,
               "mode": proj.meta.mode, "sources": [s.model_dump() for s in proj.meta.sources],
               "images": [s.model_dump() for s in proj.meta.images]})


@project_app.command("list")
def project_list() -> None:
    """List project ids."""
    for pid in list_projects():
        typer.echo(pid)


@project_app.command("clean")
def project_clean(
    pid: str,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List what would go; change nothing.")] = False,
) -> None:
    """Keep the newest renders (+ their cache), plan versions, raw clips and sheets; link gen aliases."""
    from opencine.core.retention import clean

    res = clean(Project.load(pid), dry_run=dry_run)
    echo_json({k: v for k, v in res.items() if k != "removed"} | {"removed": res["removed"][:50]})


@app.command("costs")
def costs(pid: str) -> None:
    """Ledger summary for a project (spend, quota, images shown to the agent)."""
    from opencine.core.ledger import Ledger

    echo_json(Ledger(Project.load(pid)).summary())


# packages that contribute commands; each module defines ``register(app: typer.Typer)``
_COMMAND_MODULES = [
    "opencine.index.commands",
    "opencine.render.commands",
    "opencine.motion.commands",
    "opencine.scribble.commands",
    "opencine.gen.commands",
    "opencine.review.commands",
    "opencine.mcp.commands",
    "opencine.speech.commands",
    "opencine.music.commands",
    "opencine.setup",
]

for _mod in _COMMAND_MODULES:
    try:
        importlib.import_module(_mod).register(app)
    except ModuleNotFoundError as e:  # package not built yet
        if e.name != _mod:
            raise


def main() -> None:
    # Windows consoles default to cp1252; engine output is UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    app()
