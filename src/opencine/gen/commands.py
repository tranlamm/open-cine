"""``opencine gen …``: stills, clips, quota, warm-up and HF Space duplicates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from opencine.gen.providers import NAMES

ProviderOpt = Annotated[str | None, typer.Option(
    "--provider", help="Override OPENCINE_GEN_IMAGE/OPENCINE_GEN_VIDEO: hf | fake.")]
SlotsOpt = Annotated[Path | None, typer.Option(
    "--slots", help="JSON list of slots (paths relative to the project root or absolute).")]


def _check(provider: str | None) -> str | None:
    if provider is not None and provider not in NAMES:
        raise typer.BadParameter(f"expected one of {', '.join(NAMES)}", param_hint="--provider")
    return provider


def _input(value: str) -> str:
    """A path (made absolute) or a frame ref such as S03@2.5 / C01@12.4 (kept as is); a frame
    of a video file (gen/clips/x.mp4@0.4) keeps its time."""
    import re

    if re.fullmatch(r"[SCsc]\d+(@[\d:.,]+)?", value):
        return value
    path, at, t = value.rpartition("@")
    if at and re.fullmatch(r"last|\d+(?:[.,]\d+)?", t, re.IGNORECASE):
        return f"{Path(path).resolve()}@{t}"
    return str(Path(value).resolve())


def _aspect(value: str | None) -> dict[str, str]:
    """``--aspect`` only when given: left out, a film project frames its own way (16:9)."""
    return {"aspect": value} if value else {}


def _load_slots(path: Path) -> list[dict]:
    data = json.loads(path.read_text("utf-8"))
    return data["slots"] if isinstance(data, dict) else data


def _emit(manifest) -> None:
    from opencine.cli import echo_json

    echo_json(manifest.model_dump(mode="json"))
    if manifest.stopped:
        typer.echo(manifest.message, err=True)
        raise typer.Exit(2)
    if not manifest.ok:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    gen = typer.Typer(help="Generate storyboard stills and clips (hf).",
                      no_args_is_help=True)
    app.add_typer(gen, name="gen")

    @gen.command("image")
    def image(
        pid: str,
        slot: Annotated[str, typer.Option(help="Slot id -> gen/stills/<slot>.png.")] = "s01",
        ref: Annotated[list[str] | None, typer.Option(
            help="Reference image or frame ref S03@2.5 / C01@12.4 (source resolution); repeatable. "
                 "None: text-to-image.")] = None,
        prompt: Annotated[str | None, typer.Option(
            help="Edit prompt (a new camera angle is written here too), or the whole picture "
                 "without --ref.")] = None,
        aspect: Annotated[str | None, typer.Option(
            help="9:16 | 16:9 … (default 9:16; a film project: its frame).")] = None,
        crop: Annotated[str | None, typer.Option(
            help="x,y,w,h (fractions of the first ref): the window to keep.")] = None,
        variants: Annotated[int, typer.Option(help="Stills to make (seeds seed, seed+1…).")] = 1,
        seed: int | None = None,
        model: Annotated[str | None, typer.Option(
            help="hf: qwen-2.1 (2K, 1536x2688, the default) | qwen-2.1-1k "
                 "(768x1344, ~4.5x less quota).")] = None,
        slots: SlotsOpt = None,
        provider: ProviderOpt = None,
    ) -> None:
        """Storyboard still(s) by image edit from real frames (new camera angles included)."""
        from pydantic import ValidationError

        from opencine.core.project import Project
        from opencine.gen.service import ImageSlot, gen_images

        try:
            if slots:
                items = [ImageSlot(**s) for s in _load_slots(slots)]
            elif ref or prompt:  # no --ref: text-to-image
                box = None
                if crop:
                    x, y, w, h = (float(v) for v in crop.split(","))
                    box = {"x": x, "y": y, "w": w, "h": h}
                items = [ImageSlot(id=slot, refs=[_input(r) for r in ref or []],
                                   prompt=prompt or "", seed=seed, crop=box,
                                   variants=variants, model=model, **_aspect(aspect))]
            else:
                raise typer.BadParameter("give --prompt (and --ref to edit), or --slots file.json")
        except ValidationError as e:
            raise typer.BadParameter("; ".join(err["msg"] for err in e.errors())) from e
        _emit(gen_images(Project.load(pid), items, _check(provider)))

    @gen.command("video")
    def video(
        pid: str,
        slot: Annotated[str, typer.Option(help="Slot id -> gen/clips/<slot>.mp4.")] = "s01",
        first: Annotated[str | None, typer.Option(
            help="First frame: still or frame ref S03@2.5. None: text-to-video.")] = None,
        last: Annotated[str | None, typer.Option(help="Last frame (first/last-frame mode).")] = None,
        ref: Annotated[list[str] | None, typer.Option(
            help="hf Ref2VA: a character / place picture the whole clip follows (<Picture N> after "
                 "the first frame); repeatable; the prompt is the six-field Ref2VA text.")] = None,
        prompt: Annotated[str | None, typer.Option(help="Motion prompt.")] = None,
        seconds: float = 5.0,
        aspect: Annotated[str | None, typer.Option(
            help="9:16 | 16:9 … (default 9:16; a film project: its frame).")] = None,
        seed: int | None = None,
        canvas: Annotated[str | None, typer.Option(
            help="hf: fast | mid | full (default OPENCINE_HF_VIDEO_QUALITY).")] = None,
        retime: Annotated[str, typer.Option(
            help="conform (24→30 fps, 1.25× faster) | interp | none.")] = "conform",
        model: Annotated[str | None, typer.Option(
            help="hf: h3-turbo (Turbo LoRA, 7 steps, the default) | h3-turbo-hq "
                 "(Turbo LoRA, 9 steps) | h3 (28 steps, ~2.5x quota) | with --ref: h3-ref (8 "
                 "steps, the default) | h3-ref-hq (16 steps).")] = None,
        slots: SlotsOpt = None,
        provider: ProviderOpt = None,
    ) -> None:
        """Clip(s) by text-to-video (no --first), image-to-video or first/last frame,
        post-processed to the project rate."""
        from pydantic import ValidationError

        from opencine.core.project import Project
        from opencine.gen.service import VideoSlot, gen_videos

        try:
            if slots:
                items = [VideoSlot(**s) for s in _load_slots(slots)]
            elif prompt:
                items = [VideoSlot(id=slot, first=_input(first) if first else None,
                                   refs=[_input(r) for r in ref or []],
                                   prompt=prompt, seconds=seconds, seed=seed,
                                   last=_input(last) if last else None, canvas=canvas,
                                   retime=retime, model=model, **_aspect(aspect))]
            else:
                raise typer.BadParameter("give --prompt (and --first for image-to-video), or "
                                         "--slots file.json")
        except ValidationError as e:
            raise typer.BadParameter("; ".join(err["msg"] for err in e.errors())) from e
        _emit(gen_videos(Project.load(pid), items, _check(provider)))

    @gen.command("edit")
    def edit(
        pid: str,
        slot: Annotated[str, typer.Option(help="Slot id -> gen/clips/<slot>.mp4.")] = "e01",
        source: Annotated[str | None, typer.Option(
            help="Footage window S03@1.2-6.2 (seconds into the scene).")] = None,
        mode: Annotated[str, typer.Option(help="swap (new character) | restyle (new look).")] = "swap",
        ref: Annotated[list[str] | None, typer.Option(
            help="swap: the new character (full body); restyle: the window's first frame "
                 "restyled. Path or frame ref; repeatable.")] = None,
        prompt: Annotated[str, typer.Option(help="The new character / the style, English.")] = "",
        subject: Annotated[str, typer.Option(
            help="swap: who to replace, a short English phrase (finds the mask).")] = "person",
        control: Annotated[str, typer.Option(
            help="restyle: depth (layout, camera) | canny (edges: manga, line art).")] = "depth",
        aspect: str = "9:16",
        seed: int | None = None,
        model: Annotated[str | None, typer.Option(
            help="hf: a choice of hf_space.MODELS[mode]; default = the mode's.")] = None,
        slots: SlotsOpt = None,
        provider: ProviderOpt = None,
    ) -> None:
        """Video-to-video clip(s) of footage windows: swap the character or restyle, keeping
        the motion, camera and length."""
        from pydantic import ValidationError

        from opencine.core.project import Project
        from opencine.gen.service import EditSlot, gen_edits

        try:
            if slots:
                items = [EditSlot(**s) for s in _load_slots(slots)]
            elif source and ref:
                items = [EditSlot(id=slot, source=source, mode=mode, refs=[_input(r) for r in ref],
                                  prompt=prompt, subject=subject, control=control, aspect=aspect,
                                  seed=seed, model=model)]
            else:
                raise typer.BadParameter("give --source and --ref, or --slots file.json")
        except ValidationError as e:
            raise typer.BadParameter("; ".join(err["msg"] for err in e.errors())) from e
        _emit(gen_edits(Project.load(pid), items, _check(provider)))

    @gen.command("enhance")
    def enhance(
        pid: str,
        clips: Annotated[list[str], typer.Argument(
            help="Slot ids (approved clip, else latest) or clip paths gen/clips/3~12.mp4.")],
        short: Annotated[int | None, typer.Option(
            help="Short side delivered: 1080 | 1440 (default OPENCINE_HF_ENHANCE_SHORT).")] = None,
        provider: ProviderOpt = None,
    ) -> None:
        """Sharper, larger copies of finished clips (FlashVSR+): same frames, timing and sound,
        as gen/clips/<stem>.<short>p.mp4 in the slot's place."""
        from opencine.core.project import Project
        from opencine.gen.service import gen_enhance

        if short not in (None, 1080, 1440):
            raise typer.BadParameter("expected 1080 or 1440", param_hint="--short")
        _emit(gen_enhance(Project.load(pid), clips, short, _check(provider)))

    @gen.command("face")
    def face_cmd(
        pid: str,
        clips: Annotated[list[str], typer.Argument(
            help="Slot ids (approved clip, else latest) or clip paths gen/clips/3~12.mp4.")],
        ref: Annotated[str | None, typer.Option(
            help="Identity picture (a path, <sheet>#face); default: the clip's character "
                 "sheet close-up, else its own clearest face.")] = None,
        details: Annotated[str, typer.Option(help="The person in a few words (English).")] = "",
        force: Annotated[bool, typer.Option(
            help="Refine a face taller than OPENCINE_HF_FACE_MAX_PX too.")] = False,
        provider: ProviderOpt = None,
    ) -> None:
        """Redraw the small, waxy face of finished clips at a high resolution (MiniMax-H3 face
        pass): same frames, timing and sound, as gen/clips/<stem>.face.mp4 in the slot's place."""
        from opencine.core.project import Project
        from opencine.gen.service import gen_face

        _emit(gen_face(Project.load(pid), clips, ref, details, force, _check(provider)))

    @gen.command("quota")
    def quota_cmd(provider: ProviderOpt = None) -> None:
        """Quota/credit left (and reset time) per provider."""
        from opencine.cli import echo_json
        from opencine.gen.service import quota

        echo_json(quota(_check(provider)))

    @gen.command("warm")
    def warm_cmd(provider: ProviderOpt = None) -> None:
        """Wake HF Spaces before a batch (no GPU quota used) and check pinned sha."""
        from opencine.cli import echo_json
        from opencine.gen.service import warm

        echo_json(warm(_check(provider)))

    @gen.command("hf-setup")
    def hf_setup(
        roles: Annotated[str | None, typer.Option(
            help="Comma-separated roles (default: the ones set already, else image, cond, "
                 "video_turbo, video_ref, swap_h3, music, enhance); what a role calls is added.")] = None,
        refresh: Annotated[bool, typer.Option(
            help="Also re-apply secrets, variables and patches on duplicates that exist.")] = False,
        dry_run: Annotated[bool, typer.Option(help="Say what would be made; change nothing.")] = False,
        write_env: Annotated[bool, typer.Option(
            help="Write the OPENCINE_HF_SPACE_* lines into ~/.opencine/.env (else only print them).")] = True,
    ) -> None:
        """Private ZeroGPU duplicates of every role on every HF account (OPENCINE_HF_TOKENS, else
        HF_TOKEN), the OPENCINE_HF_SPACE_* lines written to ~/.opencine/.env, each account's files
        compared with the first one's."""
        from opencine.cli import echo_json
        from opencine.gen.providers.hf_setup import ENV_FILE, setup_spaces

        names = [r.strip() for r in roles.split(",") if r.strip()] if roles else None
        echo_json(setup_spaces(names, refresh=refresh, dry_run=dry_run,
                               env_file=ENV_FILE if write_env else None))

    @gen.command("hf-duplicate")
    def hf_duplicate(
        role: Annotated[str, typer.Argument(help="image | cond | video_turbo | video | music | rife "
                 "| swap | swap_h3 | video_ref | restyle | enhance (cond before video_turbo / video "
                 "/ swap_h3 / video_ref)")],
        to: Annotated[str | None, typer.Option(help="Target <owner>/<name>.")] = None,
        source: Annotated[str | None, typer.Option(
            help="Duplicate this Space (one already set up, e.g. a public template) instead of "
                 "the pinned upstream.")] = None,
    ) -> None:
        """One duplicate, on the first account (all of them: hf-setup); prints the
        OPENCINE_HF_SPACE_* line."""
        from opencine.cli import echo_json
        from opencine.gen.providers.hf_space import SPACES, duplicate_space

        if role not in SPACES:
            raise typer.BadParameter(f"expected one of {', '.join(SPACES)}")
        echo_json(duplicate_space(role, to, source=source))
