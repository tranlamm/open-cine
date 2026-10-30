"""MiniMax-H3 character swap (role ``swap_h3``).

Runs the guide-latent Ref2VA LoRA of Alissonerdx/Minimax-H3-ComfyUI on a private duplicate of
``linoyts/minimax-h3-character-swap``. The footage window is the guide, VAE-encoded on the
target's own ``(t, h, w)`` grid, so pose, expression, camera and background follow it frame by
frame; the character picture is ``<Picture 1>``, cut out onto mid grey. The swap LoRA v1.1 and
the lightx2v ref2v Turbo LoRA are folded into ``transformer_ref`` at startup (8 Euler steps, no
CFG). Optional second pass in the same GPU call: the MiniMax-H3 latent upscaler 3D (LBH-123-AI,
Apache-2.0) lifts the latents to the ``high`` size, then a short sigma tail denoises them again
with the guide re-encoded there. That pass can add the author's LMS sharpness LoRA (r64, 0.7)
unfused, via hooks on the transformer's linears so it stays out of the first pass, with its
caption appended to the prompt (endpoint ``/mc_charswap_hq``).

``API`` is the Space's ``mc_api.py`` (rewritten by each ``opencine gen hf-duplicate swap_h3``)
and ``PATCHES`` its ``app.py`` edits; their code is in ``opencine/gen/spaces/h3_swap/``. No imports
from ``hf_space`` (it imports this module).
"""

from __future__ import annotations

from typing import Any

from opencine.gen import spaces

FPS = 24
MAX_FRAMES = 243  # 10.1 s; per the LoRA authors ~5 s holds identity, ~15 s drifts
MIN_FRAMES = 22
# output (w, h) per size and aspect: the canvas the guide is sent and generated at
SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "low": {"9:16": (416, 736), "16:9": (736, 416)},  # 0.31 MP: the author's pass-1 size
    "mid": {"9:16": (576, 1024), "16:9": (1024, 576)},  # 0.59 MP: the Space's "sharp"
    "high": {"9:16": (768, 1344), "16:9": (1344, 768)},  # H3's native canvas: the second pass
}
GREY = (128, 128, 128)  # reference background (BiRefNet cutout on #808080, as the author's workflow)
GPU_MAX = 900

# The LoRA author's v1.1 template, minus the closing sentences the Space adds; {details} =
# head-to-toe attributes starting with the gender, {target} = who to replace.
_LEAD = ("body_swap:  subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose "
         "identity, face, head, hair or fur, skin, body shape, proportions and clothing define the "
         "character in the target video. The guide video provides the scene and the performance: its "
         "background, environment, lighting, camera, framing, body pose, hand and arm movement, head "
         "rotation, gaze direction, lip movement and facial expression, frame by frame; ")
_TAIL = ("moves with the same timing, body pose, gestures, head motion, gaze, speech and facial "
         "expressions, while keeping the complete look from <Picture 1>: head, face, skin, hair or "
         "fur, body shape and clothing stay stable in every frame, with no trace of the original "
         "subject's face, hair or clothing. The background, lighting, colors and camera of the guide "
         "video are preserved. Prevent identity drift, clothing changes, texture crawling and "
         "flicker. The result is a real, photographic video.")
PROMPT_MAIN = (_LEAD + "only the main subject of the guide video is replaced: its face, hair, body "
               "and clothing become <Subject 1>.  summary: [video generation + reference] Replace the "
               "guide subject with <Subject 1>, keeping the scene, camera and the complete motion and "
               "performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the "
               "exact place of the guide subject and " + _TAIL)
PROMPT_TARGETED = (_LEAD + "only {target} is replaced: their face, hair, body and clothing become "
                   "<Subject 1>.  summary: [video generation + reference] Replace {target} in the "
                   "guide video with <Subject 1>, keeping the scene, camera and the complete motion "
                   "and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> "
                   "takes the exact place of {target} and " + _TAIL)
TRIGGER = "body_swap:"
_ANY_PERSON = {"", "person", "the person", "main subject", "subject"}
# the LMS LoRA's caption, appended to the second pass's prompt
LMS_CAPTION = ("Enhance this video with sharp, crisp details while preserving a natural "
               "photorealistic appearance.")
API_HQ = "/mc_charswap_hq"  # the endpoint with the second pass's LMS inputs


def prompt(details: str, subject: str = "person") -> str:
    """The prompt to send. One already starting with ``body_swap:`` goes as is; otherwise
    ``details`` (the character head to toe, e.g. "Female, tall, slim build, long red hair, …")
    fills the template, in its targeted form when ``subject`` names who to replace (e.g. "the
    dancer in the red dress")."""
    if details.strip().startswith(TRIGGER):  # its double spaces are part of the trained format
        return details.strip()
    text = " ".join(details.split()).rstrip(".").strip() or "character"
    target = " ".join(subject.split())
    if target.lower() in _ANY_PERSON:
        return PROMPT_MAIN.format(details=text)
    if not target.lower().startswith(("the ", "a ", "an ")):
        target = f"the {target}"
    return PROMPT_TARGETED.format(details=text, target=target)


def frames(n: int) -> int:
    """Frames a call runs for ``n`` window frames: the next ``17k + 5`` (VAE grid), at least 22."""
    n = max(MIN_FRAMES, n)
    return n + (5 - n % 17) % 17


def sigmas(text: str) -> list[float]:
    """Parse the second pass's sigma tail (``"0.9035,0.6316,0.3158"``): strictly decreasing in (0, 1)."""
    vals = [float(v) for v in text.replace(" ", "").split(",") if v]
    if not vals or any(not 0 < v < 1 for v in vals) or any(b >= a for a, b in zip(vals, vals[1:], strict=False)):
        raise ValueError(f"second-pass sigmas {text!r}: decreasing values in (0, 1)")
    return vals


def dims(aspect: str, size: str, upscale: bool) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """``((w, h) of the first pass, (w, h) of the second pass or None)``."""
    first = SIZES[size][aspect]
    if not upscale:
        return first, None
    up = SIZES["high"][aspect]
    return first, (up if up[0] > first[0] else None)


# GPU seconds, same fit as the other H3 Spaces: s/step = B·rows + C·rows² (the guide doubles the
# video rows), plus model load, guide encode and one decode at the final size. TEXT_ROWS (prompt +
# <Picture 1> vision block) is an estimate; 15 % margin. LMS_COST / LMS_LOAD: estimated overhead
# of the unfused rank-64 LoRA on the second pass and of moving its 1.2 GB onto the GPU.
B, C, SAFETY, PIXEL_FRAMES = 1.1745e-4, 3.8396e-9, 1.15, 960 * 544 * 124
TEXT_ROWS, REF_ROWS = 7750, 300
LMS_COST, LMS_LOAD = 1.1, 5


def gpu_s(width: int, height: int, n: int, steps: int, up_width: int = 0, up_height: int = 0,
          up_steps: int = 0, lms: bool = False) -> int:
    """``@spaces.GPU`` duration the Space declares (mirrors the function in ``API``)."""
    lat = (n - 5) // 17 * 5 + 2
    audio = round(n / 24 * 40) * 2

    def denoise(w: int, h: int, k: int) -> float:
        rows = TEXT_ROWS + REF_ROWS + 2 * lat * (h // 32) * (w // 32) + audio
        return k * (B * rows + C * rows * rows) * SAFETY

    t = 45 + 9 * n * width * height / PIXEL_FRAMES + denoise(width, height, steps)
    w, h = width, height
    if up_steps:
        w, h = up_width, up_height
        t += 15 + 9 * n * w * h / PIXEL_FRAMES + denoise(w, h, up_steps) * (LMS_COST if lms else 1)
        t += LMS_LOAD if lms else 0
    return int(min(GPU_MAX, t + 20 + 18 * n * w * h / PIXEL_FRAMES))


def variables(lora: str, turbo: str, cond: str | None, lms: str = "none",
              upscaler: str = "") -> dict[str, Any]:
    """Space variables: conditioner duplicate, swap LoRA, Turbo LoRA, and the second pass's LMS
    LoRA (``<repo>:<file>``, "none" = off) and latent upscaler (``<repo>:<file>``)."""
    out = {"H3_LORA_FILE": lora, "H3_TURBO_FILE": turbo, "H3_LMS_FILE": lms or "none",
           **({"OPENCINE_UPSCALER": upscaler} if upscaler else {})}
    return {"H3_CONDITIONER": cond, **out} if cond else out


API = spaces.code("h3_swap/api.py")

_LOADED = 'if load_models():\n    print(LOAD_ERROR, flush=True)\n'  # app.py, after the models load
_HOOK = (_LOADED, _LOADED + "\n" + spaces.code("h3_swap/bind.py"))
_UI = spaces.code("h3_swap/ui.py", indent=4) + "\n"
_RUN = "    run.click(\n        swap,\n"
# Second hidden endpoint, anchored after the UI's own run.click and outside the first patch, so
# an already-patched duplicate gets only this addition.
_RUN_END = '        api_name="swap",\n    )\n'
_UI_HQ = "\n" + spaces.code("h3_swap/ui_hq.py", indent=4)
PATCHES = {"app.py": [
    ('        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",\n',
     '        os.environ.get("H3_TURBO_FILE", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),'
     '  # mc\n'),
    ("import h3_video\n", "import h3_video\nimport mc_api  # mc: the hidden one-call endpoint (mc_api.py)\n"),
    _HOOK,
    (_RUN, _UI + _RUN),
    (_RUN_END, _RUN_END + _UI_HQ),
]}
