# Space code

Code that runs on the user's private Space duplicates, not in the engine. `opencine setup` and `opencine gen hf-duplicate <role>` push it: whole files are uploaded as they are, and snippets go into the upstream Space's own files next to an anchor. The anchors, upload paths and pinned upstream revisions are in `SPACES` (`providers/hf_space.py`).

| Folder | Upstream Space | Role | Adds |
|---|---|---|---|
| `qwen_image/` | Qwen/Qwen-Image-2.1 | `image` | pinned `requirements.txt`, endpoint `/mc_edit` |
| `minimax_h3/` | multimodalart/minimax-h3, MiniMaxAI/MiniMax-H3-Turbo-Lora | `video`, `video_turbo` | CRF 10 BT.709 writer, a plain Blocks UI for the Turbo Space |
| `ace_step/` | ACE-Step/Ace-Step-v1.5 | `music` | endpoint `/mc_music` |
| `scail/` | fffiloni/SCAIL-2 | `swap` | endpoint `/mc_scail`, LoRA from the Hub, in-place LoRA fusion (`wan/utils/lora.py`), Euler and LCM samplers |
| `ltx/` | linoyts/LTX-2-3-sync | `restyle` | endpoint `/mc_restyle`, models built on the CPU |
| `flashvsr/` | hugging-apps/flashvsr-plus | `enhance` | `mc_api.py` with `/mc_enhance`, CRF 12 BT.709 writer |
| `h3_swap/` | linoyts/minimax-h3-character-swap | `swap_h3` | `mc_api.py` with `/mc_charswap` and `/mc_charswap_hq` |
| `h3_ref/` | linoyts/minimax-h3-character-swap | `video_ref` | `mc_api.py` with `/mc_ref2va` |
| `h3_face/` | the `video_ref` duplicate | `face` | the face pass in `video_ref`'s `mc_api.py`, `/mc_facefix` |

Conventions:

- Snippets are stored dedented. `spaces.code(name, indent=n)` indents them for the place they go.
- `{{NAME}}` is a value from the engine (`spaces.code(name, NAME=value)`), so a GPU-duration formula has one source.
- Names such as `gr`, `spaces` or `pipeline` come from the upstream file a snippet joins. Ruff skips these folders.
- The `mc_` prefix marks what Open Cine adds (`mc_api.py`, the `/mc_*` endpoints). The engine calls these names.
- The text is pushed byte for byte. `apply_patches` skips a patch whose new text is already in the file.
