"""CLI: ``opencine review``, ``opencine critic`` and ``opencine taste``."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from opencine.core.project import Project


def _text(value: str) -> str:
    """``-`` reads the text from stdin (lets callers pass user text without shell quoting)."""
    return sys.stdin.read().strip() if value == "-" else value


def register(app: typer.Typer) -> None:
    review = typer.Typer(help="Review gate state and stats.",
                         no_args_is_help=True)
    critic = typer.Typer(help="Second opinion from another model family.",
                         no_args_is_help=True)
    app.add_typer(review, name="review")
    app.add_typer(critic, name="critic")
    taste = typer.Typer(help="Taste memory across projects.", no_args_is_help=True)
    app.add_typer(taste, name="taste")

    @taste.command("show")
    def taste_show(template: Annotated[str | None, typer.Option(help="Template name.")] = None) -> None:
        """Rules and learned numbers (what the Director reads), then the raw store."""
        from opencine.cli import echo_json
        from opencine.plan.taste import load_profile, load_rules, summary

        typer.echo(summary(template))
        echo_json({"rules": load_rules(), "profile": load_profile()})

    @taste.command("forget")
    def taste_forget(rule_id: str) -> None:
        """Drop one rule (r3)."""
        from opencine.plan.taste import forget

        typer.echo("forgotten" if forget(rule_id) else f"no rule {rule_id}")

    @taste.command("learn")
    def taste_learn(pid: str) -> None:
        """Fold a project's approved EditPlan vs its solver draft into the numbers (done
        automatically when G4 is approved)."""
        from opencine.cli import echo_json
        from opencine.plan.taste import learn

        echo_json(learn(Project.load(pid)))

    @critic.command("compare")
    def critic_compare(
        pid: str,
        a: Annotated[str, typer.Argument(help="Earlier EditPlan version (number) or generated file.")],
        b: Annotated[str, typer.Argument(help="Newer version / other variant.")],
        feedback: Annotated[str, typer.Option(help="What the human said about a ('-' = stdin).")] = "",
        gate: Annotated[str | None, typer.Option(help="Gate b goes to (logged for stats).")] = None,
    ) -> None:
        """Compare two versions (both need a rendered preview) or two generated media."""
        from opencine.cli import echo_json
        from opencine.review.critic import compare

        echo_json(compare(Project.load(pid), a, b, _text(feedback), gate))

    @critic.command("stats")
    def critic_stats(pids: Annotated[list[str] | None, typer.Argument(help="Projects (default: all).")] = None) -> None:
        """How often the critic's call matched the human's next decision."""
        from opencine.cli import echo_json
        from opencine.core.project import list_projects
        from opencine.review.critic import stats

        echo_json(stats([Project.load(p) for p in (pids or list_projects())]))

    @review.command("stats")
    def review_stats(
        pids: Annotated[list[str] | None, typer.Argument(help="Projects (default: all).")] = None,
        since: Annotated[str | None, typer.Option(help="ISO date: leave out older projects.")] = None,
    ) -> None:
        """Rounds per gate, G4 approved in round 1, critic agreement."""
        from opencine.cli import echo_json
        from opencine.core.project import list_projects
        from opencine.review.stats import review_stats as stats

        echo_json(stats([Project.load(p) for p in (pids or list_projects())], since))

    @review.command("state")
    def review_state(pid: str) -> None:
        """Gate state of a project."""
        from opencine.cli import echo_json
        from opencine.review.service import review_state as state

        echo_json(state(Project.load(pid)))

    @review.command("feedback")
    def review_feedback(
        pid: str,
        gate: str,
        text: str,
        status: Annotated[str | None, typer.Option(
            help="pending | approved | changes | cancelled (omit to only record the text).")] = None,
    ) -> None:
        """Record human feedback on a gate's current round."""
        from opencine.cli import echo_json
        from opencine.review.gates import record_feedback

        echo_json(record_feedback(Project.load(pid), gate, text, status))
