---
name: opencine-setup
description: Installs, configures and updates the Open Cine video engine (the opencine MCP server) for the current coding agent. Checks FFmpeg, uv and Node, installs the engine, saves the user's Hugging Face PRO token, creates their private ZeroGPU Spaces and registers the MCP server; optionally sets up the Telegram tool and Hermes Agent. Use when the user asks to install, set up, configure, update or repair Open Cine, when the opencine MCP tools are missing, or when hf_quota lists calls under not_set. Not for making videos (use opencine-edit, opencine-film or opencine-swap).
---

# Open Cine setup

Goal: the `opencine` MCP tools answer and `hf_quota()` shows nothing under `not_set`.
Reply in the user's language; keep it brief.

## Checklist

- [ ] 1. Already installed?
- [ ] 2. Tools
- [ ] 3. Engine
- [ ] 4. Token, Spaces, MCP
- [ ] 5. Verify
- [ ] 6. Optional: Telegram, Hermes

### 1. Already installed?

Call `hf_quota()`.
- It answers and `not_set` is empty → done; tell the user they can ask for a video.
- It answers but lists `not_set` → go to step 4b.
- The tool does not exist → continue with step 2.

### 2. Tools

Run each and check the version:

| Tool | Check | Needed for | Install |
|---|---|---|---|
| uv | `uv --version` | the engine | `winget install astral-sh.uv` · `brew install uv` · `pipx install uv` · [other ways](https://docs.astral.sh/uv/getting-started/installation/) |
| FFmpeg ≥ 7 (libass, libx264, zscale) | `ffmpeg -version` | everything | `winget install Gyan.FFmpeg` · `brew install ffmpeg` · `apt install ffmpeg` |
| Node ≥ 22 | `node --version` | motion and scribble layers, MCP registration | `winget install OpenJS.NodeJS.LTS` · `brew install node` |
| git | `git --version` | the engine | system package manager |

Install what is missing (ask before installing system packages), then open a new shell so PATH updates.

### 3. Engine

The engine lives in `~/.opencine/engine` (Windows: `%USERPROFILE%\.opencine\engine`).

```bash
git clone --depth 1 https://github.com/tranlamm/open-cine ~/.opencine/engine   # first time
git -C ~/.opencine/engine pull                                                # update
uv --directory ~/.opencine/engine sync
```

Add `--extra vision` to `sync` when the user wants scribbles that find named objects (Florence-2, EdgeTAM; ~400 MB of models on first use).

### 4. Token, Spaces, MCP

Generation runs on private duplicates of Hugging Face ZeroGPU Spaces in the user's own account, so it needs a **Hugging Face PRO** account and a token with **write** access (huggingface.co/settings/tokens).

**4a. Fresh install.** Prefer that the user types the token themselves, so it never enters the chat. Ask them to run in a terminal:

```bash
uv --directory ~/.opencine/engine run opencine setup
```

It checks the tools, asks for the token (hidden input), saves it to `~/.opencine/.env`, creates the Spaces (a few minutes), installs the renderer (`npm ci` in `fx/`) and registers the MCP server with every detected agent through `npx add-mcp`.

If the user pastes the token in chat instead, run it for them:

```bash
uv --directory ~/.opencine/engine run opencine setup --hf-token <TOKEN> --yes
```

Options: `--hf-token` again for each extra PRO account (each adds its own daily ZeroGPU quota); `-a <agent>` to register only some agents (`claude-code`, `codex`, `cursor`, `gemini-cli`, `opencode`, `vscode`…); `--no-spaces`, `--no-fx`, `--no-mcp` to skip a step.

**4b. Tools answer but Spaces are missing** (e.g. the Claude Code plugin, where the token was entered when the plugin was enabled): call `hf_setup()`, then call it again every minute until `status` is `done` or `failed`. Read `report.notes` and pass any action items to the user (e.g. accept a model licence on its Hub page).

### 5. Verify

- Ask the user to restart the agent (new MCP servers load at startup).
- Call `hf_quota()`: `not_set` is empty and `left_s` shows the daily quota.
- Newly created Spaces build on Hugging Face for several minutes; the first call to each can take longer while it starts.

### 6. Optional: Telegram, Hermes

Only when the user asks. Engine path as in step 3.

**Telegram tool** (`telegram_send` posts gates and finals to the user's Telegram chat):
1. The user creates a bot with @BotFather and sends it any message.
2. Ask them to run (hidden input; the chat id is read from that message):

   ```bash
   uv --directory ~/.opencine/engine run opencine setup --telegram --no-spaces --no-fx --no-mcp
   ```

   Token pasted in chat instead: `… setup --telegram-token <TOKEN> --yes --no-spaces --no-fx --no-mcp`.
   Hermes already runs on that bot (it holds the bot's updates): add `--telegram-chat <their Telegram user id>`.
3. Done when the chat receives "Open Cine is connected."

**Hermes Agent** (the user talks to the agent from Telegram through the Hermes gateway):
1. Hermes is installed and its Telegram gateway works (`hermes gateway setup`, `hermes gateway status`).
2. Run:

   ```bash
   uv --directory ~/.opencine/engine run opencine setup --hermes --no-spaces --no-fx
   ```

   It adds `<engine>/skills` to Hermes (`skills.external_dirs`) and registers the MCP server for Hermes only (`add-mcp -a hermes`).
3. `hermes gateway restart`. Suggest `hermes config set display.busy_input_mode queue` (a new message waits instead of interrupting a long turn).
4. Done when, in Telegram, the bot answers "call hf_quota" with the quota.

## Troubleshooting

- `AuthError` or 401 → the token is wrong or read-only; create a write token.
- "not PRO" / ZeroGPU refused → the account needs Hugging Face PRO; a PRO account hosts up to 10 ZeroGPU Spaces.
- Quota runs out every day → add a second PRO account with `--hf-token` (calls move to the next account on their own).
- `npx add-mcp` unavailable → add a stdio MCP server named `opencine` by hand, command: `uv --directory <engine path> run --quiet opencine mcp serve`.
- Motion or scribble fails with a Node error → `npm ci` in `<engine>/fx` (Node ≥ 22).
- `telegram_send`: "chat not found" → the user has not messaged the bot yet, or the chat id is wrong; 409 "Conflict" during setup → another program (Hermes) polls the bot, pass `--telegram-chat`.
- Hermes does not list the opencine skills → `hermes config get skills.external_dirs` shows the engine's `skills` folder; then `hermes gateway restart`.
