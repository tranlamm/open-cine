"""MiniMax-H3 Ref2VA: clips from reference pictures (role ``video_ref``).

Runs on a second private duplicate of ``linoyts/minimax-h3-character-swap`` without its swap LoRA.
Each picture (character or place, at most ``MAX_PICTURES``) enters as a ``<Picture N>`` vision
block in the prompt embedding (the conditioner's ``/encode_ref2va``) plus a one-latent-frame
reference block at the canvas area on its own aspect. An optional first frame is ``<Picture 1>``
and is also laid on the target's first latent frame as a guide, so the shot starts on it. The
lightx2v ref2v Turbo LoRA is folded into ``transformer_ref`` (Euler, no CFG); the soundtrack is
decoded by the audio VAE. The Director's six-field Ref2VA prompt is sent unchanged.

``API`` is the Space's ``mc_api.py`` (rewritten by each ``opencine gen hf-duplicate video_ref``)
and ``PATCHES`` its ``app.py`` edits; their code is in ``opencine/gen/spaces/h3_ref/``. No imports
from ``hf_space`` (it imports this module).
"""

from __future__ import annotations

import re
from typing import Any

from opencine.gen import spaces

FPS = 24
# Pictures per clip, first frame included. The conditioner (Qwen3-VL, ~62 GB on the GPU) spends
# ~7k tokens per picture at 1344x768: 3 fit (22.8k tokens, 14 s); 4 or more run out of GPU
# memory. H3 itself accepts 9.
MAX_PICTURES = 3
VERSION = 1  # how pictures are sent (sizes, first-frame guide); part of the cache fingerprint
SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music")
_PICTURE = re.compile(r"<Picture (\d+)>")

# GPU seconds, same fit as the other H3 Spaces: s/step = B·rows + C·rows², where rows = prompt +
# per picture (vision block + one latent frame) + first-frame guide + target video and audio rows.
# TEXT_ROWS (350–500 words + labels) and VISION_ROWS are estimates; 15 % margin. GPU_MAX is the
# Space's cap: a request above it is refused before upload rather than killed mid-run.
B, C, SAFETY, PIXEL_FRAMES = 1.1745e-4, 3.8396e-9, 1.15, 960 * 544 * 124
TEXT_ROWS, VISION_ROWS, GPU_MAX = 2000, 5750, 900


def need_s(width: int, height: int, n: int, steps: int, pictures: int, guide: int = 1) -> float:
    """GPU seconds one call needs, uncapped: load + picture encodes, denoise, video and audio
    decodes (mirrors the function in ``API``)."""
    lat = (n - 5) // 17 * 5 + 2
    per = (height // 32) * (width // 32)
    audio = round(n / 24 * 40) * 2
    rows = TEXT_ROWS + pictures * (VISION_ROWS + per) + (guide + lat) * per + audio
    denoise = steps * (B * rows + C * rows * rows) * SAFETY
    return 45 + 5 * pictures + denoise + 20 + 18 * n * width * height / PIXEL_FRAMES + 10


def gpu_s(width: int, height: int, n: int, steps: int, pictures: int, guide: int = 1) -> int:
    """``@spaces.GPU`` duration the Space declares."""
    return int(min(GPU_MAX, need_s(width, height, n, steps, pictures, guide)))


def variables(turbo: str, cond: str | None) -> dict[str, Any]:
    """Space variables: conditioner duplicate, no task LoRA, the ref2v Turbo LoRA file."""
    out = {"H3_LORA_FILE": "none", "H3_TURBO_FILE": turbo}
    return {"H3_CONDITIONER": cond, **out} if cond else out


def prompt_problems(prompt: str, pictures: int) -> list[str]:
    """Why ``prompt`` cannot go to Ref2VA with ``pictures`` pictures: not the six-field format,
    a ``<Picture N>`` not given, or a picture never cited (it would not be bound)."""
    text = prompt.strip()
    out = []
    if not text.startswith("subject_definitions:"):
        out.append("a refs clip takes H3's Ref2VA format: start with 'subject_definitions:' and write "
                   f"the six sections ({', '.join(SECTIONS)}; see the film skill's prompting.md, Ref2VA)")
    cited = {int(n) for n in _PICTURE.findall(text)}
    if extra := sorted(n for n in cited if not 1 <= n <= pictures):
        out.append(f"<Picture {extra[0]}> is cited but only {pictures} picture(s) are given "
                   "(first frame first, then refs in order)")
    if missing := [n for n in range(1, pictures + 1) if n not in cited]:
        out.append(f"<Picture {missing[0]}> is given but the prompt never cites it: name each picture "
                   "in subject_definitions (<Subject N> is the … in <Picture N>)")
    return out


API = spaces.code("h3_ref/api.py")

_LOADED = 'if load_models():\n    print(LOAD_ERROR, flush=True)\n'  # app.py, after the models load
_HOOK = (_LOADED, _LOADED + "\n" + spaces.code("h3_ref/bind.py"))
_UI = spaces.code("h3_ref/ui.py", indent=4) + "\n"
_RUN = "    run.click(\n        swap,\n"
_LORA_FOR = "        for repo, filename, strength, label in LORAS:\n"
_LORA_GET = "            path = hf_hub_download(repo, filename)\n"
_LORA_LOOP = _LORA_FOR + _LORA_GET
PATCHES = {"app.py": [
    ('        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",\n',
     '        os.environ.get("H3_TURBO_FILE", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),'
     '  # mc\n'),
    (_LORA_LOOP, _LORA_FOR + spaces.code("h3_ref/lora_off.py", indent=12) + _LORA_GET),
    ("import h3_video\n", "import h3_video\nimport mc_api  # mc: the hidden one-call endpoint (mc_api.py)\n"),
    _HOOK,
    (_RUN, _UI + _RUN),
]}
