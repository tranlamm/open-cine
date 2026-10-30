```text
 ██████╗ ██████╗ ███████╗███╗   ██╗     ██████╗██╗███╗   ██╗███████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██╔════╝██║████╗  ██║██╔════╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ██║     ██║██╔██╗ ██║█████╗
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██║     ██║██║╚██╗██║██╔══╝
╚██████╔╝██║     ███████╗██║ ╚████║    ╚██████╗██║██║ ╚████║███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝     ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝
```

<p align="center">
  <b>Turn your agent into a film director.</b><br>
  Cut shorts from your footage, generate films from a prompt, swap the character in a clip.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-3776AB">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-server-8A2BE2">
  <img alt="Agent Skills" src="https://img.shields.io/badge/agent%20skills-4-orange">
  <a href="https://github.com/tranlamm/open-cine/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/tranlamm/open-cine?style=flat"></a>
</p>

<p align="center">
  <a href="#sample-film">Sample film</a> ·
  <a href="#vibe-install">Vibe install</a> ·
  <a href="#install">Install</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#contributing">Contributing</a> ·
  <a href="#license">License</a> ·
  <a href="#acknowledgements">Acknowledgements</a>
</p>

https://github.com/user-attachments/assets/bbb318d6-0323-4017-a8db-fd7f4601558b

---

Open Cine gives any agent (Claude Code, Codex, Cursor, Gemini CLI, OpenCode, …) a director's playbook (**skills**) and a video engine (**MCP server**). The agent plans the edit; the engine does the deterministic work (scene detection, beat-synced cuts, tracked 9:16 crops, captions, rendering), and generation runs on **private duplicates of Hugging Face ZeroGPU Spaces in your own account**.

| Skill | What it does |
|---|---|
| `opencine-edit` | Cuts a 9:16 short (30–60 s) from your footage: hook, rhythm, match cuts, music, captions, generated extra angles, virtual multicam, character swap / restyle, motion graphics, scribbles. |
| `opencine-film` | Makes a short film from a prompt. Every shot is generated, with consistent characters and places. |
| `opencine-swap` | Remakes your clip with another character from a picture or description. Motion, timing, camera and sound stay. |
| `opencine-setup` | Installs the engine, saves your HF token, creates your Spaces and registers the MCP server. |

## Sample film

A short film made with Open Cine.

https://github.com/user-attachments/assets/c56cb3aa-8a1e-4fc8-8deb-0816c911908a

## Requirements

- A **Hugging Face PRO** account and a token with write access ([settings/tokens](https://huggingface.co/settings/tokens))
- [uv](https://docs.astral.sh/uv/), [FFmpeg](https://ffmpeg.org/) ≥ 7 (libass, libx264, zscale), git
- Node.js ≥ 22 (motion and scribble layers, MCP registration)

## Vibe install

Paste this into your agent:

```text
Set up Open Cine for me: https://github.com/tranlamm/open-cine
Follow skills/opencine-setup/SKILL.md in that repo.
```

The agent checks the tools, installs the engine, has you enter your HF token, creates your Spaces and registers the MCP server. Restart the agent when it is done.

## Install

**Any agent**

```bash
npx skills add tranlamm/open-cine
```

Then ask your agent: **“Set up Open Cine.”** The `opencine-setup` skill installs the engine, asks for your HF token, creates your Spaces and registers the MCP server.

**Claude Code plugin**

```text
/plugin install open-cine --marketplace tranlamm/open-cine
```

Enter your Hugging Face token when prompted (stored in the system keychain), then ask **“Set up Open Cine”** to create your Spaces.

**Manual**

```bash
git clone https://github.com/tranlamm/open-cine ~/.opencine/engine
uv --directory ~/.opencine/engine sync
uv --directory ~/.opencine/engine run opencine setup
```

`opencine setup` checks the tools, saves the token to `~/.opencine/.env`, creates the Spaces, installs the renderer and registers the MCP server with your agents via [`add-mcp`](https://github.com/neondatabase/add-mcp). For any other MCP client, add a stdio server named `opencine`:

```bash
uv --directory ~/.opencine/engine run --quiet opencine mcp serve
```

Optional extras: `uv --directory ~/.opencine/engine sync --extra vision` (scribbles that find named objects with Florence-2 + EdgeTAM), `--extra mask` (SAM 2.1 mask wipes).

## Usage

Talk to your agent:

```text
Cut a 30-second reel from ~/Videos/lisbon/, upbeat, end on the sunset.
Make a 20-second noir short: a detective finds a letter in the rain.
Swap the dancer in ~/clips/dance.mp4 with the character in ~/refs/robot.png.
```

The agent stops at review gates and waits for you:

| Gate | You review |
|---|---|
| G1 | Shot list and storyboard |
| G2 | Generated stills |
| G3 | Generated clips |
| G4 | The rendered preview |

Answer in plain words: `all ok` · `ok 1,3; 2 lower camera angle` · `cancel`. Your taste (pacing, transitions, text density) is remembered across projects. Projects live in `~/.opencine/projects/<pid>/`.

## Telegram and Hermes (optional)

**Review on Telegram.** Create a bot with [@BotFather](https://t.me/BotFather) and send it a message, then run:

```bash
uv --directory ~/.opencine/engine run opencine setup --telegram --no-spaces --no-fx --no-mcp
```

It asks for the bot token, reads your chat id from that message and sends a test message. The agent then posts gates and finals with the `telegram_send` tool. If Hermes already polls the same bot, add `--telegram-chat <your user id>`.

**Hermes Agent.** Talk to the agent from Telegram through the [Hermes](https://github.com/NousResearch/hermes-agent) gateway. With Hermes installed and its Telegram gateway set up (`hermes gateway setup`):

```bash
uv --directory ~/.opencine/engine run opencine setup --hermes --no-spaces --no-fx
hermes gateway restart
```

This adds `skills/` to Hermes (`skills.external_dirs`) and registers the MCP server (`add-mcp -a hermes`). Send footage to your bot (the public Bot API takes files up to 20 MB) and ask for a video. Without the Telegram tool, Hermes attaches review files through `MEDIA:` lines. Recommended: `hermes config set display.busy_input_mode queue`, so a new message waits instead of interrupting a long turn.

<details>
<summary><b>CLI</b></summary>

| Command | Does |
|---|---|
| `opencine setup` | First-run wizard |
| `opencine setup --telegram` · `--hermes` | Telegram tool, Hermes Agent |
| `opencine mcp serve` | MCP server (stdio) |
| `opencine project new <videos…>` | Create a project |
| `opencine ingest <pid>` · `opencine scenes <pid>` | Index footage, print the scene table |
| `opencine render <pid> plan.json --preset final` | Render an EditPlan |
| `opencine gen hf-setup` · `opencine gen quota` | Create Spaces, read quota |

`opencine --help` lists every command.

</details>

<details>
<summary><b>Layout</b></summary>

```text
skills/          agent skills: opencine-edit, opencine-film, opencine-swap, opencine-setup
src/opencine/    engine: ingest · index · plan · render · gen · review · mcp
fx/              Remotion renderer for the motion and scribble layers
.claude-plugin/  Claude Code plugin and marketplace
```

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The code is released under the [Apache License 2.0](LICENSE). The bundled fonts (Anton, Montserrat) are under the SIL Open Font License.

The models and Spaces the engine calls keep their own licences, and several are **non-commercial or own-use only** (marked † below). Check them before any commercial use. The `fx/` renderer uses [Remotion](https://www.remotion.dev/license), which is free for individuals and small teams; larger companies need a Remotion company license.

## Acknowledgements

Open Cine stands on the work of these model authors, Space maintainers and tool builders. Thank you.

**Generation (Hugging Face ZeroGPU Spaces)**

1. Qwen Team. *Qwen-Image-2.1*. Space [Qwen/Qwen-Image-2.1](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1). Qwen Research License †
2. MiniMax. *MiniMax-H3*. Spaces [multimodalart/minimax-h3](https://huggingface.co/spaces/multimodalart/minimax-h3), [MiniMaxAI/MiniMax-H3-Turbo-Lora](https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora), conditioner [multimodalart/qwen3vl-conditioner](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner). MiniMax-H3 Community License †
3. larryvrh. *MiniMax-H3 Turbo LoRA*, via [MiniMaxAI/MiniMax-H3-Turbo-Lora](https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora). Apache-2.0
4. Alissonerdx. *MiniMax-H3 character-swap LoRA v1.1 and LMS LoRA*. [Alissonerdx/Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI); Space [linoyts/minimax-h3-character-swap](https://huggingface.co/spaces/linoyts/minimax-h3-character-swap). BFS Character Swap licence, own use only †
5. LightX2V. *Distillation LoRAs* for fast Ref2VA sampling. [huggingface.co/lightx2v](https://huggingface.co/lightx2v)
6. LBH-123-AI. *MiniMax-H3 latent upscaler*. [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
7. *SCAIL-2*. Space [fffiloni/SCAIL-2](https://huggingface.co/spaces/fffiloni/SCAIL-2). MIT weights, Apache-2.0 code
8. Lightricks. *LTX-2.3* with Union-Control IC-LoRA. Space [linoyts/LTX-2-3-sync](https://huggingface.co/spaces/linoyts/LTX-2-3-sync). LTX-2 Community License †; Gemma 3 text encoder under the Gemma terms
9. *FlashVSR+*. Space [hugging-apps/flashvsr-plus](https://huggingface.co/spaces/hugging-apps/flashvsr-plus). Apache-2.0
10. *ACE-Step 1.5*. Space [ACE-Step/Ace-Step-v1.5](https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5). MIT

**Local vision**

11. Microsoft. *Florence-2-base-ft*, ONNX by [onnx-community](https://huggingface.co/onnx-community/Florence-2-base-ft). MIT
12. Meta. *EdgeTAM*, via [yonigozlan/EdgeTAM-hf](https://huggingface.co/yonigozlan/EdgeTAM-hf) and [jax-image-tools/edgetam-video-onnx](https://huggingface.co/jax-image-tools/edgetam-video-onnx). Apache-2.0
13. Meta. *SAM 2.1 tiny*, ONNX by [vietanhdev](https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models). Apache-2.0
14. *Depth Anything V2 Small*, ONNX by [onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-small). Apache-2.0
15. OpenCV Zoo. *YuNet* face detector. [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet). MIT
16. *Real-ESRGAN* ([xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), BSD-3-Clause) and *RIFE ncnn* ([nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan), MIT) for final upscaling and interpolation

**Tools**: [FFmpeg](https://ffmpeg.org/), [Remotion](https://www.remotion.dev/), [PySceneDetect](https://www.scenedetect.com/), [librosa](https://librosa.org/), [OpenCV](https://opencv.org/), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), [gradio_client](https://www.gradio.app/), [huggingface_hub](https://github.com/huggingface/huggingface_hub), [skills](https://github.com/vercel-labs/skills), [add-mcp](https://github.com/neondatabase/add-mcp), [Hermes Agent](https://github.com/NousResearch/hermes-agent). **Fonts**: [Anton](https://fonts.google.com/specimen/Anton), [Montserrat](https://github.com/JulietaUla/Montserrat).

## Star History

<a href="https://www.star-history.com/#tranlamm/open-cine&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=tranlamm/open-cine&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=tranlamm/open-cine&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=tranlamm/open-cine&type=Date" />
  </picture>
</a>
