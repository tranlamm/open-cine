"""CLI entry points for the MCP server and background jobs."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    mcp_app = typer.Typer(help="MCP server for coding agents.", no_args_is_help=True)
    app.add_typer(mcp_app, name="mcp")

    @mcp_app.command("serve")
    def serve() -> None:
        """Run the `opencine` MCP server on stdio."""
        from opencine.mcp.server import serve as run

        run()

    @app.command("job-run", hidden=True)
    def job_run(pid: str, job_id: str) -> None:
        """Execute a queued background job (spawned by opencine.core.jobs)."""
        from opencine.core import jobs
        from opencine.core.project import Project

        job = jobs.run(Project.load(pid), job_id)
        raise typer.Exit(0 if job["status"] == "done" else 1)

    @app.command("job-status")
    def job_status(pid: str, job_id: str) -> None:
        """Show a background job's status."""
        from opencine.cli import echo_json
        from opencine.core import jobs
        from opencine.core.project import Project

        echo_json(jobs.status(Project.load(pid), job_id))
