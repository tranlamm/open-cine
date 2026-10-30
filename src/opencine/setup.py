"""``opencine setup``: first-run wizard.

Checks the tools, saves the HF token(s) to ``~/.opencine/.env``, creates the private ZeroGPU
Space duplicates, installs the motion/scribble renderer (``fx/``) and registers the MCP server
with the user's coding agents through ``npx add-mcp``. Optional: the Telegram tool
(``--telegram``) and Hermes Agent (``--hermes``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from typing import Any

import typer

from opencine.core.config import ENV_FILE, REPO_ROOT, get_settings
from opencine.core.proc import find_exe

FX_DIR = REPO_ROOT / "fx"
NODE_MAJOR = 22


def _version(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([find_exe(cmd[0]), *cmd[1:]], capture_output=True, text=True,
                             timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "?"


def check_tools() -> dict[str, Any]:
    """Versions of the external tools; ``missing`` lists the required ones not found."""
    s = get_settings()
    found = {"ffmpeg": _version([s.ffmpeg, "-version"]), "ffprobe": _version([s.ffprobe, "-version"]),
             "node": _version([s.node, "--version"]), "npx": _version(["npx", "--version"]),
             "uv": _version(["uv", "--version"])}
    missing = [k for k in ("ffmpeg", "ffprobe") if not found[k]]
    node = re.match(r"v?(\d+)", found["node"] or "")
    if not node or int(node.group(1)) < NODE_MAJOR:
        found["node_note"] = f"Node ≥ {NODE_MAJOR} is needed for the motion and scribble layers"
    return {"tools": found, "missing": missing}


def save_tokens(tokens: list[str]) -> None:
    """One token → ``HF_TOKEN``; several → ``OPENCINE_HF_TOKENS`` (used in order)."""
    from opencine.gen.providers.hf_setup import write_env

    values = {"HF_TOKEN": tokens[0]} if len(tokens) == 1 else {"OPENCINE_HF_TOKENS": ",".join(tokens)}
    write_env(ENV_FILE, values)
    reload_settings()


def reload_settings() -> None:
    from opencine.gen.config import get_gen_settings

    get_settings.cache_clear()
    get_gen_settings.cache_clear()


def install_fx() -> str:
    """``npm ci`` in ``fx/`` unless its dependencies are there already."""
    if (FX_DIR / "node_modules").is_dir():
        return "already installed"
    npm = shutil.which("npm")
    if not npm:
        return "skipped: npm not found"
    res = subprocess.run([find_exe("npm"), "ci", "--no-audit", "--no-fund"], cwd=FX_DIR, check=False)
    return "installed" if res.returncode == 0 else f"npm ci failed ({res.returncode})"


def mcp_command() -> str:
    """The stdio command that starts the server from this checkout."""
    return f"uv --directory {REPO_ROOT.as_posix()} run --quiet opencine mcp serve"


def register_mcp(agents: list[str], *, global_: bool = True) -> int:
    """Add the server to the user's agents with ``npx add-mcp`` (detected agents when none given)."""
    if " " in REPO_ROOT.as_posix():
        typer.echo(f"The path {REPO_ROOT} has spaces; add the server by hand:\n  {mcp_command()}")
        return 1
    cmd = [find_exe("npx"), "-y", "add-mcp", mcp_command(), "--name", "opencine", "--env", "PYTHONUTF8=1", "-y"]
    cmd += ["-g"] if global_ else []
    for a in agents:
        cmd += ["-a", a]
    return subprocess.run(cmd, check=False).returncode


def setup_telegram(token: str | None, chat: str | None, *, ask: bool) -> str:
    """Save the bot token and chat id (read from the bot's newest message when not given), then
    post a test message."""
    from opencine.gen.providers.hf_setup import write_env
    from opencine.review.telegram import Bot, TelegramError, latest_chat, send

    s = get_settings()
    token = token or s.telegram_bot_token
    if not token and ask:
        typer.echo("A bot token from @BotFather (with Hermes: the same bot as Hermes).")
        token = typer.prompt("Telegram bot token", hide_input=True, default="").strip()
    if not token:
        return "skipped: no bot token"
    chat = chat or s.telegram_chat_id
    if not chat:
        if ask:
            typer.prompt("Send any message to your bot, then press Enter", default="", show_default=False)
        try:
            chat = latest_chat(Bot(token, s.telegram_api_url))
        except TelegramError as e:  # 409 while another program polls the bot (e.g. Hermes)
            typer.echo(f"Could not read the bot's messages: {e}")
        if not chat and ask:
            chat = typer.prompt("Telegram chat id", default="").strip()
    if not chat:
        return "skipped: no chat id (pass --telegram-chat)"
    write_env(ENV_FILE, {"OPENCINE_TELEGRAM_BOT_TOKEN": token, "OPENCINE_TELEGRAM_CHAT_ID": chat})
    reload_settings()
    try:
        send("Open Cine is connected.", [])
    except TelegramError as e:
        return f"saved, but the test message failed: {e}"
    return f"connected (chat {chat})"


def setup_hermes() -> str:
    """Hermes Agent loads this checkout's skills (``skills.external_dirs``); the MCP server is
    registered by the MCP step (``add-mcp -a hermes``)."""
    if not shutil.which("hermes"):
        return "skipped: hermes not found (https://hermes-agent.nousresearch.com)"
    hermes = find_exe("hermes")
    got = subprocess.run([hermes, "config", "get", "skills.external_dirs", "--json"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=False)
    try:
        dirs = json.loads(got.stdout) if got.returncode == 0 else []
    except ValueError:
        dirs = []
    dirs = [d for d in dirs if isinstance(d, str)] if isinstance(dirs, list) else []
    skills = (REPO_ROOT / "skills").as_posix()
    if skills not in dirs:
        res = subprocess.run([hermes, "config", "set", "skills.external_dirs", json.dumps([*dirs, skills])],
                             check=False)
        if res.returncode:
            return "failed: hermes config set skills.external_dirs"
    return "skills linked; restart the gateway: hermes gateway restart"


def register(app: typer.Typer) -> None:
    @app.command("setup")
    def setup(
        hf_token: list[str] | None = typer.Option(
            None, "--hf-token", help="HF token (write access, PRO account); repeat for several accounts."),
        agent: list[str] | None = typer.Option(
            None, "--agent", "-a", help="Agent to register the MCP server with (add-mcp id, e.g. "
            "claude-code, codex, cursor, gemini-cli, opencode); repeatable. Default: detected agents."),
        spaces: bool = typer.Option(True, help="Create the private ZeroGPU Space duplicates."),
        fx: bool = typer.Option(True, help="Install the motion/scribble renderer (npm ci in fx/)."),
        mcp: bool = typer.Option(True, help="Register the MCP server with your agents (npx add-mcp)."),
        telegram: bool = typer.Option(False, "--telegram", help="Set up the Telegram tool (asks for the bot token)."),
        telegram_token: str | None = typer.Option(None, help="Telegram bot token (from @BotFather)."),
        telegram_chat: str | None = typer.Option(None, help="Telegram chat id; default: the bot's newest chat."),
        hermes: bool = typer.Option(False, "--hermes", help="Add the skills and the MCP server to Hermes Agent."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Never prompt."),
    ) -> None:
        """Set up Open Cine: tools check, HF token, private Spaces, renderer, MCP registration;
        optionally Telegram and Hermes."""
        from opencine.cli import echo_json

        tools = check_tools()
        echo_json(tools)
        if tools["missing"]:
            typer.echo(f"Install {', '.join(tools['missing'])} first (FFmpeg ≥ 7 with libass).", err=True)
            raise typer.Exit(1)

        tokens = [t.strip() for t in hf_token or [] if t.strip()]
        if not tokens and not get_settings().hf_account_tokens() and not yes and sys.stdin.isatty():
            typer.echo("A Hugging Face PRO token with write access: https://huggingface.co/settings/tokens")
            entered = typer.prompt("HF token (several: comma-separated)", hide_input=True, default="")
            tokens = [t.strip() for t in entered.split(",") if t.strip()]
        if tokens:
            save_tokens(tokens)
            typer.echo(f"Saved to {ENV_FILE}")

        if spaces:
            if not get_settings().hf_account_tokens():
                typer.echo("No HF token: skipping the Spaces (run `opencine setup` again later).")
            else:
                from opencine.gen.providers.hf_setup import setup_spaces

                typer.echo("Creating private ZeroGPU Spaces (a few minutes)…")
                echo_json(setup_spaces(None, env_file=ENV_FILE))
                reload_settings()

        if fx:
            typer.echo(f"Renderer: {install_fx()}")

        if telegram or telegram_token or telegram_chat:
            ask = not yes and sys.stdin.isatty()
            typer.echo(f"Telegram: {setup_telegram(telegram_token, telegram_chat, ask=ask)}")

        agents = list(agent or [])
        if hermes and "hermes" not in agents:
            agents.append("hermes")  # with no --agent: Hermes only
        if mcp:
            if not shutil.which("npx"):
                typer.echo(f"npx not found; add this stdio MCP server to your agent:\n  {mcp_command()}")
            elif yes or typer.confirm("Register the MCP server with your coding agents?", default=True):
                register_mcp(agents)
            else:
                typer.echo(f"MCP command: {mcp_command()}")

        if hermes:
            typer.echo(f"Hermes: {setup_hermes()}")
        typer.echo("Done. Restart your agent, then ask it to make a video.")
