"""``opencine music``: add a track, make one (ACE-Step on the music Space), read a beat grid."""

from __future__ import annotations

from typing import Annotated

import typer


def register(app: typer.Typer) -> None:
    music = typer.Typer(help="Background music: add, generate, beat grid.", no_args_is_help=True)
    app.add_typer(music, name="music")

    @music.command("add")
    def music_add(pid: str, path: Annotated[str, typer.Argument(help="Audio file (a video with sound works too).")]) -> None:
        """Copy a track into the project and detect its beats."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.music.library import add_music

        echo_json(add_music(Project.load(pid), path))

    @music.command("gen")
    def music_gen(
        pid: str,
        prompt: Annotated[str, typer.Option(help="English: genre, mood, instruments, energy curve.")],
        seconds: Annotated[float, typer.Option(help="Track length (timeline + ~4 s).")] = 30.0,
        bpm: Annotated[int | None, typer.Option(help="Cutting tempo; omit to let the model pick.")] = None,
        key: Annotated[str | None, typer.Option(help='e.g. "A minor".')] = None,
        seed: Annotated[int | None, typer.Option(help="Another seed = another take.")] = None,
    ) -> None:
        """Generate an instrumental track with ACE-Step 1.5 (OPENCINE_HF_SPACE_MUSIC) into music/, beat-tracked."""
        from opencine.cli import echo_json
        from opencine.core.project import Project
        from opencine.music.gen import generate

        echo_json(generate(Project.load(pid), prompt, seconds, bpm, key, seed))

    @music.command("beats")
    def music_beats(pid: str, name: str) -> None:
        """A track's beat grid as text (seconds/frame, * = strong)."""
        from opencine.core.project import Project
        from opencine.music.library import beats_text

        typer.echo(beats_text(Project.load(pid), name))
