"""Hugging Face ZeroGPU Spaces provider, called through ``gradio_client``.

Calls go to the user's private duplicates (``OPENCINE_HF_SPACE_<ROLE>``) with ``HF_TOKEN`` so the
PRO quota is used; with several accounts (``OPENCINE_HF_TOKENS``) each call runs on the first whose
daily quota still holds its reservation. Past every quota the Hub bills credits (``hf_overquota``);
with that off, the local quota books refuse calls the server would refuse anyway. ``SPACES`` holds
each role's upstream Space, endpoint, GPU-duration formula and patches (the code they add is in
``opencine/gen/spaces/``); ``duplicate_space`` makes a duplicate at the pinned upstream revision
with those patches. Inputs are sent at exactly the size each model works at (``fit_canvas``), so
no Space stretches or re-crops them.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from opencine.core.config import get_settings
from opencine.gen import spaces
from opencine.gen.base import (
    AuthError,
    GenProvider,
    GenResult,
    Kind,
    ProviderError,
    QuotaExhausted,
    TransientError,
    fit_canvas,
    limit_pixels,
    parse_aspect,
    save_output,
    verify_video,
)
from opencine.gen.config import SPACE_HOSTS, GenSettings, SpaceRef, get_gen_settings
from opencine.gen.providers import h3face, h3ref, h3swap
from opencine.gen.providers.hf_accounts import (  # noqa: F401
    WINDOW,
    Account,
    QuotaBook,
    make_accounts,
)

MIN_SECONDS, MAX_SECONDS = 2.0, 15.0  # Space floor (OPENCINE_HF_VIDEO_MIN_S) .. the model card's 15 s
MUSIC_MIN_S, MUSIC_MAX_S = 10.0, 600.0  # ACE-Step 1.5 track length
_sleep = time.sleep  # retry backoff
_poll = time.sleep  # job status polling
_clock = time.monotonic
POLL_S = 0.5
MAX_SEED = 2**31 - 1
PREP_VERSION = 2  # how inputs are prepared for the Spaces (fit_canvas, sizes)


# ---- per-Space call shapes ------------------------------------------------------
def _file(path: Path):
    from gradio_client import handle_file

    return handle_file(str(path))


# MiniMax-H3 canvases (label → (height, width)), copied from the Space: the label goes over the
# wire and must match the denoiser's and the conditioner's tables exactly.
H3_CANVASES: dict[str, dict[str, str]] = {  # aspect → quality tier → label
    "9:16": {"fast": "544x960 · 9:16 fast", "mid": "640x1152 · 9:16", "full": "768x1344 · 9:16 full"},
    "16:9": {"fast": "960x544 · 16:9 fast", "mid": "1152x640 · 16:9", "full": "1344x768 · 16:9 full"},
    "1:1": {"fast": "544x544 · 1:1 fast", "mid": "768x768 · 1:1 full", "full": "768x768 · 1:1 full"},
    "4:3": {"fast": "768x576 · 4:3 fast", "mid": "1024x768 · 4:3 full", "full": "1024x768 · 4:3 full"},
    "3:4": {"fast": "576x768 · 3:4 fast", "mid": "768x1024 · 3:4 full", "full": "768x1024 · 3:4 full"},
}


def h3_canvas(aspect: str, quality: str) -> tuple[str, int, int]:
    """``(label, height, width)`` of the H3 canvas for ``aspect`` at ``quality``."""
    if aspect not in H3_CANVASES:
        raise ProviderError(f"hf video: aspect {aspect} not supported; use {', '.join(H3_CANVASES)}")
    label = H3_CANVASES[aspect][quality]
    w, h = (int(x) for x in label.split(" ")[0].split("x"))
    return label, h, w


def h3_frames(seconds: float) -> int:
    """Frames H3's video VAE decodes: the next ``17n + 5`` at 24 fps (the Space's ``snap_frames``)."""
    frames = max(1, round(seconds * 24))
    while frames % 17 != 5:
        frames += 1
    return frames


def _h3_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    """The denoiser Space's ``get_duration``: steps × (b·rows + c·rows²) + decode + allowances."""
    _, h, w = h3_canvas(aspect, s.hf_video_quality)
    frames = h3_frames(seconds)
    patches = (h // 32) * (w // 32)
    rows = ((frames - 5) // 17 * 5 + 2) * patches + n_images * patches
    denoise = s.hf_video_steps * (1.1745e-4 * rows + 3.8396e-9 * rows**2)
    decode = 15 + 15 * (h * w * frames) / (960 * 544 * 124)
    return max(60, int(denoise + decode) + 12 + 10)


def _h3_side(s: GenSettings) -> float:
    """Quota the conditioner books inside a video call: encode 45 s (+ rewrite 60 s), xlarge."""
    return 2 * (45 + (60 if s.hf_video_upsample else 0))


# ---- Qwen-Image geometry ---------------------------------------------------------
QWEN_AREA = 1024 * 1024


def qwen_dims(ratio: float) -> tuple[int, int]:
    """diffusers ``calculate_dimensions(1024², ratio)`` → ``(w, h)``: the size the edit
    pipeline puts every condition image at (its VAE latent)."""
    w = math.sqrt(QWEN_AREA * ratio)
    return round(w / 32) * 32, round(w / ratio / 32) * 32


def still_size(aspect: str) -> tuple[int, int]:
    """``(w, h)`` of a still: H3's ``full`` canvas when it is a Qwen fixed point (9:16 →
    768×1344, 16:9 → 1344×768), else Qwen's own size for the aspect. At a fixed point the
    input, the latent and the output are the same pixels: no internal resize, no drift."""
    if aspect in H3_CANVASES:
        _, h, w = h3_canvas(aspect, "full")
        if qwen_dims(w / h) == (w, h):
            return w, h
    return qwen_dims(float(parse_aspect(aspect)))


# ---- Qwen-Image-2.1 -----------------------------------------------------------------
Q21_STEPS = 40  # fixed in the official Space (model card default)


def q21_size(aspect: str, quality: str) -> tuple[int, int]:
    """``(w, h)`` of a Qwen-Image-2.1 still: ``still_size`` (``1k``) or twice it (``2k``, the
    model's native ~4 MP; 9:16 → 1536×2688), so the H3 keyframe is an exact 2× downscale."""
    w, h = still_size(aspect)
    return (w * 2, h * 2) if quality == "2k" else (w, h)


def _q21_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    """The official Space's ``generation_duration`` (fitted on a large slice, capped at 300 s):
    no negative prompt, prefix KV cache on (≤ 9 condition images)."""
    w, h = q21_size(aspect, s.hf_image_quality)
    pixels, prefix = w * h, n_images * (1024 // 16) ** 2
    cached = prefix * 32 * 2 * 4096 * 2 <= 10e9
    tokens = pixels / 256 + (0.2 * prefix if cached else prefix)
    per_step = 3.4e-6 * tokens**1.363 * 1.3
    fixed = 4 + 2 * n_images + 1.5e-6 * pixels
    return int(min(300, (fixed + Q21_STEPS * per_step) * 1.25))


# ---- MiniMax-H3 prompts (official prompting guide) ------------------------------------
IR_FIELD = "integrated_multimodal_description:"
_IR_LINE = re.compile(r"^\s*(For the target video,.*?referenced\.|How the reference pictures align "
                      r"with the target video[^\n]*)\s*", re.IGNORECASE)
_SHOT = re.compile(r"\[Shot (\d+)\]")


def h3_prompt(prompt: str, seconds: float, frames: int) -> tuple[str, bool]:
    """``(final prompt, is_ir)``. A prompt written in H3's Context-IR format (it starts with
    ``integrated_multimodal_description:``, after an optional alignment line) gets the
    guide's alignment line rebuilt for this request: task (``frames`` keyframes: 0 = T2VA,
    no line; 1 = I2VA; 2 = FL2VA), effective duration (17n+5 frames at 24 fps) and final
    shot. Anything else is returned unchanged."""
    body = _IR_LINE.sub("", prompt.strip(), count=1).strip()
    if not body.startswith(IR_FIELD):
        return prompt, False
    if frames == 0:
        return body, True
    duration = h3_frames(seconds) / 24
    last = max([int(n) for n in _SHOT.findall(body)] or [1])
    if frames >= 2:
        line = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) "
                "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot "
                f"{last}) aligns with the {duration:.2f}-second mark of the target video.")
    else:
        line = ("For the target video, at 0.00 seconds into the target video, <Picture 1> "
                "(from [Shot 1]) is fully referenced.")
    return f"{line}\n\n{body}", True


def _h3_outputs(result: Any) -> tuple[str | None, dict[str, Any]]:
    """``(refined prompt, report numbers)`` from the H3 Space's ``(video, report, refined,
    panel)`` result; the report carries the real conditioner and denoise seconds."""
    if not isinstance(result, list | tuple) or len(result) < 3:
        return None, {}
    report = result[1] if isinstance(result[1], str) else ""
    refined = result[2] if isinstance(result[2], str) and result[2].strip() else None
    nums: dict[str, Any] = {}
    for name, pattern, cast in (("conditioner_s", r"conditioner (\d+(?:\.\d+)?)s", float),
                                ("generate_s", r"denoise \+ decode (\d+(?:\.\d+)?)s", float),
                                ("frames", r"(\d+) frames", int)):
        if m := re.search(pattern, report):
            nums[name] = cast(m.group(1))
    return refined, nums


@dataclass(frozen=True)
class SpaceSpec:
    upstream: str  # Space to duplicate
    api_name: str | None  # None: not called by the engine (a helper Space)
    model: str
    size: str  # ZeroGPU size: large (1×) | xlarge (2× quota)
    # (clip seconds, input images, settings, aspect) → @spaces.GPU duration declared by the Space
    declared_s: Callable[[float, int, GenSettings, str], float]
    # (files, prompt, seconds, seed, settings, aspect) → endpoint kwargs
    build: Callable[..., dict[str, Any]]
    side_s: Callable[[GenSettings], float] = lambda s: 0.0  # other bookings inside one call
    secrets: tuple[str, ...] = ()  # Space secrets set to the caller's HF token on duplicate
    variables: Callable[[GenSettings], dict[str, str]] = lambda s: {}  # Space variables on duplicate
    files: dict[str, str] | None = None  # files replaced on the duplicate (path → content)
    # edits to the duplicate's own files: path → [(old, new)]; each ``old`` must occur once
    patches: dict[str, list[tuple[str, str]]] | None = None
    # settings this Space runs with (e.g. the turbo step count), applied before every use
    tune: Callable[[GenSettings], GenSettings] = lambda s: s
    needs: tuple[str, ...] = ()  # roles it calls mid-request: woken and duplicated first
    # (aspect, settings) → output (w, h); None = ``still_size``
    out_size: Callable[[str, GenSettings], tuple[int, int]] | None = None
    # video-to-video: (settings) → the rate frames go in at; frames → the count it takes
    # (padded up); the most frames one call takes
    fps: Callable[[GenSettings], int] | None = None
    frames: Callable[[int], int] = lambda n: n
    max_frames: int | None = None
    # upstream commit the patches were written against: a duplicate is put back to it when
    # upstream has moved since (None = upstream as it is)
    revision: str | None = None
    repo: str = ""  # the duplicate's name when no account has one yet (``hf_setup``)
    gated: tuple[str, ...] = ()  # model repos whose licence each account must accept first
    # (settings) → the endpoint a call goes to, when the settings choose it (else ``api_name``)
    api_for: Callable[[GenSettings], str] | None = None


# Qwen-Image-2.1 Space as of its last build (2026-09-27): diffusers/transformers from main,
# pinned to that day's commits; the rest to the releases then, within the builder's torch<=2.13
# (torchvision 0.28) and diffusers' huggingface-hub<2.
_Q21_REQUIREMENTS = spaces.code("qwen_image/requirements.txt")

# Qwen-Image-2.1 Space: its UI is two chained events (prompt rewrite, then generate) sharing a
# gr.State. One extra endpoint runs the same safety check and GPU step in one call, without
# the rewrite, at an explicit size, and returns a PNG.
_Q21_ENDPOINT = spaces.code("qwen_image/endpoint.py", indent=4) + "\n"
_Q21_ANCHOR = "    # Generate Image button event: screen/rewrite the prompt off-GPU, then generate.\n"
_Q21_PATCHES = [(_Q21_ANCHOR, _Q21_ENDPOINT + _Q21_ANCHOR)]

# Conditioner Space: ``spaces==0.51.1`` clashes with the ``spaces`` the builder now installs
# (0.51.3, "ResolutionImpossible"); leave the version to the builder, as the denoiser Space does.
_COND_PATCHES = [("spaces==0.51.1", "spaces  # version set by the Space builder (mc)")]

# H3 Space: diffusers' ``encode_video`` runs libx264 with no options (CRF 23) and converts RGB
# with swscale's default BT.601 matrix, untagged. This writer is the same loop at CRF 10 with
# BT.709 YUV and tags, so the clip matches phone footage in the edit.
H3_WRITER = spaces.code("minimax_h3/encode_video.py")
_H3_PATCHES = [
    ("    encode_video(frames, fps=FPS, output_path=path, audio=audio, "
     "audio_sample_rate=sampling_rate)",
     "    _mc_encode_video(frames, FPS, path, audio, sampling_rate)"),
    ("def _fit_keyframe(image_path, current_canvas):",
     f"{H3_WRITER}\n\ndef _fit_keyframe(image_path, current_canvas):"),
]

# H3 Turbo Space (MiniMaxAI): the same denoiser code with a Turbo LoRA folded in at startup, but
# its UI is a gr.Workflow canvas whose endpoint is named after the graph's output node and takes
# positional in_0..in_8. Replaced by a plain Blocks UI with the H3 Space's /generate signature.
_H3_TURBO_UI = spaces.code("minimax_h3/turbo_ui.py")
_H3_TURBO_PATCHES = [
    *_H3_PATCHES,
    ('demo = gr.Workflow(graph="workflow.json", bind={"generate_video": generate_video})',
     _H3_TURBO_UI),
]
# larryvrh card: v4-step600 is the recommended checkpoint at 6-8 steps (the Space defaults to the
# older v1 ckpt850); the lightx preview LoRA is not used, so it is not loaded either
H3_TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
# the same LoRA at the top of the card's 6–8 passes (SGLang's "speed/quality balance": 9 steps)
H3_TURBO_HQ_STEPS = 9


def _h3_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    return {
        "prompt": p, "image_path": _file(f[0]) if f else None,  # none: text-to-video
        "last_image_path": _file(f[1]) if len(f) > 1 else None,
        "canvas": h3_canvas(a, s.hf_video_quality)[0], "duration": sec,
        "steps": s.hf_video_steps, "seed": seed, "upsample": s.hf_video_upsample,
    }


# ACE-Step 1.5 Space (MIT code + weights): its UI is a large multi-tab app. One extra endpoint
# makes an instrumental track from a caption, length, BPM and key with the XL-turbo DiT and the
# 5 Hz LM, under a GPU duration that grows with the track (the same formula as _ace_declared).
ACE_GPU_BASE_S, ACE_GPU_PER_S, ACE_GPU_MAX_S = 30, 0.8, 300
_ACE_IMPORT = "from acestep.gradio_ui import create_gradio_interface\n"
_ACE_TOP = _ACE_IMPORT + "\n" + spaces.code("ace_step/endpoint.py", GPU_BASE_S=ACE_GPU_BASE_S,
                                              GPU_PER_S=ACE_GPU_PER_S, GPU_MAX_S=ACE_GPU_MAX_S)
_ACE_ENDPOINT = spaces.code("ace_step/ui.py", indent=4) + "\n"
_ACE_QUEUE = "    # Enable queue for multi-user support\n"
_ACE_PATCHES = [(_ACE_IMPORT, _ACE_TOP),
                (_ACE_QUEUE, _ACE_ENDPOINT + _ACE_QUEUE)]
# flash-attn 2.8.3 wheels have no sm120 kernel and FA3 does not run there either (ZeroGPU is
# Blackwell since ~05/2026: "no kernel image is available"); without them both models use SDPA
_ACE_REQUIREMENTS = [
    ("flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
     "v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux' "
     "and python_version == '3.11'\n",
     "# mc: no flash-attn on Linux (no sm120 kernel): PyTorch SDPA\n"),
    ("# Kernels library for flash-attn3 (preferred over flash-attn when available)\nkernels\n",
     "# mc: no kernels / flash-attn3 (does not run on sm120)\n"),
]


def _ace_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "") -> float:
    """``_mc_gpu_s`` of the patched ACE-Step Space."""
    return min(ACE_GPU_MAX_S, ACE_GPU_BASE_S + ACE_GPU_PER_S * seconds)


# ---- video-to-video ---------------------------------------------------------------------
# Each Space gets one hidden endpoint taking plain files (gr.File: no re-encoding of masks or
# stills on the way in) and returning an mp4, raising on failure (their UI endpoints return a
# traceback as text). Frames go in 1:1 at the model's rate; the provider pads them to a count
# the model takes and trims the answer back to the window.

# SCAIL-2 (fffiloni/SCAIL-2 app): replacement / animation from a reference on black, its mask,
# the driving clip and the per-frame mask clip; 81-frame segments, 5 overlapping.
SCAIL_SIZES = {"low": {"9:16": (512, 896), "16:9": (896, 512)},
               "high": {"9:16": (704, 1280), "16:9": (1280, 704)}}
SCAIL_SEGMENT, SCAIL_OVERLAP, SCAIL_MAX_FRAMES = 81, 5, 157
SCAIL_GPU_MAX = 1200  # the app's GPU_DURATION_MAX


def scail_frames(n: int) -> int:
    """Frames SCAIL-2 runs without dropping any: one segment of 4k+1 (≤ 81), else 81 + 76·m
    (a trailing partial segment is dropped by the model, so the clip is padded up to whole ones)."""
    if n <= SCAIL_SEGMENT:
        return max(5, n + (-(n - 1)) % 4)
    step = SCAIL_SEGMENT - SCAIL_OVERLAP
    return SCAIL_SEGMENT + -(-(n - SCAIL_SEGMENT) // step) * step


def _scail_gpu_s(frames: int, w: int, h: int, steps: int, cfg: float) -> int:
    """``_mc_gpu_s`` of the patched Space: model passes × seconds per pass at that size (17 s at
    512×896×81 frames on the full card) + loading, 20 % margin."""
    nfe = steps * (2 if cfg > 1 else 1)
    work = frames * w * h / (81 * 512 * 896)
    return int(min(SCAIL_GPU_MAX, 45 + nfe * 17 * work ** 1.15 * 1.2))


def _scail_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    w, h = SCAIL_SIZES[s.hf_scail_size][aspect if aspect in SCAIL_SIZES["low"] else "9:16"]
    return _scail_gpu_s(scail_frames(round(seconds * s.hf_scail_fps)), w, h, s.hf_scail_steps,
                        s.hf_scail_cfg)


def _scail_lora(s: GenSettings) -> dict[str, str]:
    repo, _, file = s.hf_scail_lora.partition(":")
    return {"SCAIL_LORA_REPO": repo, "SCAIL_LORA_FILE": file} if repo and file else {}


_SCAIL_ENDPOINT = spaces.code("scail/endpoint.py") + "\n\n"
_SCAIL_UI = "\n" + spaces.code("scail/ui.py", indent=8)
_SCAIL_LORA_PATH = '    lora_path = os.getenv("SCAIL_LORA_PATH") or None\n'
# First-order flow samplers for the LightX2V distill LoRA, which was distilled with one (4 steps
# on σ 1/.94/.83/.63 at shift 5; Wan2.2-Lightning: "use euler solver because it's used in
# distillation"). The Spaces only had multistep UniPC / DPM++, with which SCAIL-2 at shift 1 left
# white specks on flat backgrounds. Same interface as their schedulers: .timesteps,
# step(v, t, x, return_dict, generator)[0].
_FIRST_ORDER = "\n\n" + spaces.code("scail/first_order.py") + "\n\n"
_UNIPC_IMPORT = "from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler\n"


def _first_order_branch(indent: str) -> tuple[str, str]:
    """The patch adding ``euler`` / ``lcm`` before a Space's ``dpm++`` scheduler branch."""
    anchor = f"{indent}elif sample_solver == 'dpm++':\n"
    return anchor, (f"{indent}elif sample_solver in ('euler', 'lcm'):  # mc\n"
                    f"{indent}    sample_scheduler = _FlowFirstOrder(\n"
                    f"{indent}        sampling_steps, shift, sample_solver == 'lcm', self.device)\n"
                    f"{indent}    timesteps = sample_scheduler.timesteps\n" + anchor)


# wan/utils/lora.py, whole. Upstream folds only the low-rank pairs, skipping a LightX2V LoRA's
# 242 `.diff` and 43 lone `.diff_b`, and rebuilds every weight as a new tensor, which with the
# fp32 DiT (~66 GB) exhausts 104 GB of RAM. This one folds a Kijai-format LoRA in place, as
# ComfyUI applies it: the low-rank pairs, every `.diff` (whole-weight deltas: QK norms, norm3)
# and `.diff_b` (bias deltas), × the strength; summed in fp32, rounded once to the weight's dtype.
_SCAIL_FUSE = spaces.code("scail/lora.py")
_SCAIL_PATCHES = {
    "app.py": [
        ("def build_ui():\n", _SCAIL_ENDPOINT + "def build_ui():\n"),
        ("\n    return demo\n", _SCAIL_UI + "\n    return demo\n"),
        (_SCAIL_LORA_PATH, _SCAIL_LORA_PATH + spaces.code("scail/lora_from_hub.py", indent=4)),
    ],
    # weights (~47 GB) take long to fetch on a cold start
    "README.md": [("app_file: app.py\n", "app_file: app.py\nstartup_duration_timeout: 1h\n")],
    "wan/scail.py": [
        ("        self.model.load_state_dict(state_dict)\n",
         "        self.model.load_state_dict(state_dict)\n"
         "        del state_dict  # mc: free the 33 GB fp16 copy before the LoRA\n"),
        (_UNIPC_IMPORT, _UNIPC_IMPORT + _FIRST_ORDER),
        _first_order_branch(" " * 16),
    ],
}


def _scail_variables(s: GenSettings) -> dict[str, str]:
    """The Comfy-Org fp16 DiT (32.8 GB; the default path converts the 65.6 GB fp32 checkpoint
    and overflows the 50 GB disk), the distill LoRA, whole 81-frame segments (no auto-raise)."""
    return {"SCAIL_SAFETENSORS_REPO_ID": "Comfy-Org/SCAIL-2",
            "SCAIL_SAFETENSORS_FILENAME": "diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors",
            "SCAIL_AUTO_AVOID_SEGMENT_TRUNCATION": "0", "SCAIL_SAMPLE_SOLVER": s.hf_scail_solver,
            **_scail_lora(s)}


def _scail_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    w, h = SCAIL_SIZES[s.hf_scail_size][a]
    return {"image": _file(f[1]), "mask_image": _file(f[2]), "video": _file(f[0]),
            "mask_video": _file(f[3]), "prompt": p, "replace": s.hf_scail_replace,
            "steps": s.hf_scail_steps, "guide_scale": s.hf_scail_cfg, "shift": s.hf_scail_shift,
            "seed": seed, "width": w, "height": h, "frames": round(sec * s.hf_scail_fps)}


# LTX-2.3 distilled + Union-Control IC-LoRA (linoyts/LTX-2-3-sync, fp8): the styled first
# frame + a control clip (depth or edges) at half the size, 24 fps, 8k+1 frames, two stages.
# Sides are multiples of 128: stage 1 runs at half size and the IC-LoRA reference at half of
# that, which the VAE (32×) must divide evenly (576 wide broke its encoder: "axis of length 9").
LTX_FPS = 24
LTX_SIDE = 128
LTX_SIZES = {"low": {"9:16": (640, 1152), "16:9": (1152, 640)},
             "high": {"9:16": (768, 1408), "16:9": (1408, 768)}}
LTX_MAX_FRAMES = 241


def ltx_frames(n: int) -> int:
    return max(9, n + (-(n - 1)) % 8)


def _ltx_gpu_s(frames: int, w: int, h: int) -> int:
    return int(min(240, max(60, 40 + w * h * frames / 1.2e6)))


def _ltx_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    w, h = LTX_SIZES[s.hf_restyle_size][aspect if aspect in LTX_SIZES["low"] else "9:16"]
    return _ltx_gpu_s(ltx_frames(round(seconds * LTX_FPS)), w, h)


_LTX_ENDPOINT = spaces.code("ltx/endpoint.py") + "\n\n"
_LTX_UI = spaces.code("ltx/ui.py", indent=4) + "\n"
_LTX_BLOCKS = 'with gr.Blocks(title="LTX 2.3 Sync", css=css, theme=purple_citrus) as demo:\n'
_LTX_LEDGER = "_ledger_1 = pipeline.stage_1_model_ledger\n"
_LTX_CPU_BUILD = spaces.code("ltx/cpu_build.py")
_LTX_PATCHES = {"app.py": [
    # the Space is a lip-sync demo: its pipeline appends this to every prompt
    ('        prompt += " synchronized lipsync"\n', '        # mc: no lip-sync suffix\n'),
    (_LTX_LEDGER, _LTX_CPU_BUILD + _LTX_LEDGER),
    (_LTX_BLOCKS, _LTX_ENDPOINT + _LTX_BLOCKS),
    ("    generate_btn.click(\n", _LTX_UI + "    generate_btn.click(\n"),
]}


def _ltx_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    w, h = LTX_SIZES[s.hf_restyle_size][a]
    return {"image": _file(f[1]), "control": _file(f[0]), "prompt": p,
            "frames": ltx_frames(round(sec * LTX_FPS)), "height": h, "width": w,
            "strength": s.hf_restyle_strength, "seed": seed}


# ---- enhance: FlashVSR+ one-step video super-resolution -------------------------------------
# hugging-apps/flashvsr-plus (FlashVSR+ turbo, Apache-2.0 code + weights): a whole clip streams
# through one GPU call, centre-cropped and resized to the output size (sides multiples of 128).
# The hidden endpoint lives in its own file (``files``: a rerun of hf-duplicate replaces it, so a
# new duration formula reaches an existing duplicate); app.py only imports and binds it. The
# writer is set to CRF 12 BT.709, tagged (imageio's default was CRF 25, BT.601 untagged). OpenCV
# decodes the input with the BT.601 matrix whatever its tags, so the clip is sent in BT.601.
ENHANCE_SIDE = 128
ENHANCE_RATE = 24  # the ``seconds`` of an enhance call are its frames at this rate
ENHANCE_ASPECT_TOL = 0.02  # canvas aspect vs the clip's: the Space crops the difference
ENHANCE_GPU_MAX = 300
ENHANCE_PREP_VERSION = 1  # how clips are sent and finished (BT.601 in, Lanczos down, CRF 12)
# GPU seconds = fixed + per latent frame and megapixel, latents = frames / compression; fitted
# on real calls at 2048x1152 (243 frames: 95 s at compression 4, 167 s at 2; 72 frames: 33 s).
# Upstream's anchors (short clips, CRF 25) said 55 / 65 s, which real calls overran.
ENHANCE_FIXED_S, ENHANCE_LATENT_MP_S = 15.0, 0.55


def enhance_canvas(aspect: str, short: int) -> tuple[int, int]:
    """``(w, h)`` FlashVSR+ runs at for a clip of ``aspect`` (``"1344:768"`` or ``"16:9"``): the
    smallest sides of 128s with the short one ≥ ``short`` and an aspect within 2 % of the clip's.
    H3's 7:4 → 2048×1152 for 1080 (16:9: 1.6 % of the height cropped), 2688×1536 (2×) for 1440."""
    ratio = float(parse_aspect(aspect))
    shape = max(ratio, 1 / ratio)
    lo = math.ceil(short / ENHANCE_SIDE)
    fits = [(s * n, n, s) for s in range(lo, lo + 9) for n in range(s, math.ceil(s * shape) + 2)
            if abs(n / s / shape - 1) <= ENHANCE_ASPECT_TOL]
    _, n, s = min(fits) if fits else (0, round(lo * shape), lo)
    n, s = n * ENHANCE_SIDE, s * ENHANCE_SIDE
    return (n, s) if ratio >= 1 else (s, n)


def enhance_size(aspect: str, short: int) -> tuple[int, int]:
    """The size delivered: the canvas brought to ``short`` by Lanczos (2048×1152 → 1920×1080)."""
    w, h = enhance_canvas(aspect, short)
    k = short / min(w, h)
    return round(w * k / 2) * 2, round(h * k / 2) * 2


def _enhance_gpu_s(w: int, h: int, frames: int, compression: int) -> int:
    """``gpu_s`` of the Space's ``mc_api.py``: the fitted cost + 25 % + 5 s, capped at
    ``ENHANCE_GPU_MAX`` (a call that would need more is refused before it is sent)."""
    cost = ENHANCE_FIXED_S + ENHANCE_LATENT_MP_S * max(frames, 17) / compression * w * h / 1e6
    return int(min(ENHANCE_GPU_MAX, cost * 1.25 + 5))


def _enhance_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "16:9") -> float:
    w, h = enhance_canvas(aspect, s.hf_enhance_short)
    return _enhance_gpu_s(w, h, round(seconds * ENHANCE_RATE), s.hf_enhance_compression)


def _enhance_call(f: list[Path], sec: float, s: GenSettings, a: str) -> dict[str, Any]:
    w, h = enhance_canvas(a, s.hf_enhance_short)
    return {"video": _file(f[0]), "width": w, "height": h, "compression": s.hf_enhance_compression,
            "color": s.hf_enhance_color, "frames": round(sec * ENHANCE_RATE)}


_FLASHVSR_API = spaces.code("flashvsr/api.py", FIXED_S=ENHANCE_FIXED_S,
                            LATENT_MP_S=ENHANCE_LATENT_MP_S, GPU_MAX_S=ENHANCE_GPU_MAX)
_FLASHVSR_HOOK = spaces.code("flashvsr/bind.py") + "\n"
_FLASHVSR_UI = spaces.code("flashvsr/ui.py", indent=4) + "\n"
_FLASHVSR_WRITER = spaces.code("flashvsr/writer.py", indent=12)
FLASHVSR_WEIGHTS = "46b79520946ac67dec94d1566490690953c7debc"  # ruchangyao/FlashVSR_plus
_FLASHVSR_PATCHES = {
    "app.py": [
        ('    allow_patterns=["turbo/*", "TCDecoder.ckpt"],\n',
         f'    allow_patterns=["turbo/*", "TCDecoder.ckpt"],\n    revision="{FLASHVSR_WEIGHTS}",  # mc\n'),
        ("WARNING_MD = (\n", _FLASHVSR_HOOK + "WARNING_MD = (\n"),
        ("    gr.Examples(\n", _FLASHVSR_UI + "    gr.Examples(\n"),
    ],
    "inference/output.py": [('            output_params=["-preset", preset],\n', _FLASHVSR_WRITER)],
}


def enhance_input(src: Path, dst: Path) -> Path:
    """``src`` as the Space's OpenCV decodes it right: BT.601 YUV (tagged so), CRF 8, no sound."""
    from opencine.core import ffmpeg
    from opencine.gen.qc import matrix_of, video_stream

    matrix, rng = matrix_of(video_stream(src))
    ffmpeg.run(["-y", "-i", str(src), "-an", "-vf",
                f"scale=in_color_matrix={matrix}:in_range={rng}:flags=accurate_rnd+full_chroma_int,"
                "format=gbrp,scale=out_color_matrix=bt601:out_range=tv:flags=accurate_rnd,"
                "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "8",
                "-colorspace", "smpte170m", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", str(dst)])
    return dst


def enhance_finish(got: Path, src: Path, fps: Any, frames: int, size: tuple[int, int],
                   out: Path) -> int:
    """The Space's clip → ``out``: Lanczos to ``size``, BT.709 CRF 12, frame N at N / ``fps``
    (the source's exact rate), at most ``frames``, ``src``'s sound copied; → frames kept."""
    from opencine.core import ffmpeg
    from opencine.gen.post import CRF, frame_count
    from opencine.gen.qc import matrix_of, video_stream

    matrix, rng = matrix_of(video_stream(got))
    w, h = size
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp{out.suffix}")
    ffmpeg.run(["-y", "-r", str(fps), "-i", str(got), "-i", str(src), "-map", "0:v:0",
                "-map", "1:a:0?", "-vf",
                f"scale={w}:{h}:in_color_matrix={matrix}:in_range={rng}"
                ":flags=lanczos+accurate_rnd+full_chroma_int,format=gbrp,"
                "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd,format=yuv420p,setsar=1",
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
                "-bf", "0", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range", "tv", "-c:a", "copy",
                "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)
    return frame_count(video_stream(out))


EDIT_PREP_VERSION = 2  # how video-to-video inputs are made (rates, sizes, masks, clicks, controls)


# MiniMax-H3 character swap (Space code, prompt and sizes in ``h3swap``)
def _h3swap_dims(aspect: str, s: GenSettings) -> tuple[tuple[int, int], tuple[int, int] | None]:
    return h3swap.dims(aspect if aspect in h3swap.SIZES["low"] else "9:16", s.hf_h3swap_size,
                       s.hf_h3swap_upscale)


def _h3swap_lms(s: GenSettings) -> float:
    """The LMS strength the second pass runs with (0: no second pass, or LMS off)."""
    off = s.hf_h3swap_lms_file.strip().lower() in ("", "none")
    return 0.0 if off or not s.hf_h3swap_upscale else max(0.0, s.hf_h3swap_lms)


def _h3swap_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    (w, h), up = _h3swap_dims(aspect, s)
    tail = len(h3swap.sigmas(s.hf_h3swap_up_sigmas)) if up else 0
    return h3swap.gpu_s(w, h, h3swap.frames(round(seconds * h3swap.FPS)), s.hf_h3swap_steps,
                        *(up or (0, 0)), tail, lms=bool(up and _h3swap_lms(s)))


def _h3swap_side(s: GenSettings) -> float:
    """Image-only ``/encode_ref2va`` calls on the conditioner (xlarge): one, two with the LMS
    caption (the second pass's prompt)."""
    caption = bool(_h3swap_lms(s) and s.hf_h3swap_lms_caption and _h3swap_dims("9:16", s)[1])
    return 2 * 45 * (2 if caption else 1)


def _h3swap_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    (w, h), up = _h3swap_dims(a, s)
    out = {"video": _file(f[0]), "image": _file(f[1]), "prompt": p,
           "frames": round(sec * h3swap.FPS), "width": w, "height": h,
           "steps": s.hf_h3swap_steps, "seed": seed, "up_width": up[0] if up else 0,
           "up_height": up[1] if up else 0, "up_sigmas": s.hf_h3swap_up_sigmas if up else ""}
    if up:  # the endpoint with the LMS inputs (h3swap.API_HQ)
        out.update(up_lms=_h3swap_lms(s), up_caption=int(s.hf_h3swap_lms_caption))
    return out


def _h3swap_api(s: GenSettings) -> str:
    return h3swap.API_HQ if _h3swap_dims("9:16", s)[1] else "/mc_charswap"


def _h3swap_out(aspect: str, s: GenSettings) -> tuple[int, int]:
    first, up = _h3swap_dims(aspect, s)
    return up or first


H3SWAP_COLOUR_EVERY = 3  # frames measured for the window colour match (masks on these only)


def h3swap_reference(src: Path, work: Path, cutout: bool = True) -> tuple[Path, bool]:
    """The character picture as sent: at most 1 MP, cut out onto grey when the local vision
    models find it (the LoRA author's workflow); else as it is (the Space's own fallback)."""
    from PIL import Image

    from opencine.gen.control import grey_cutout
    from opencine.gen.qc import load_rgb

    ref = limit_pixels(src, work / "character.png", 1024 * 1024)
    if not cutout:
        return ref, False
    try:
        cut = grey_cutout(load_rgb(ref), grey=h3swap.GREY)
    except ProviderError:  # no local vision models
        return ref, False
    if cut is None:
        return ref, False
    Image.fromarray(cut).save(grey := work / "character_grey.png")
    return grey, True


def h3swap_colour(clip: Path, window: Any, subject: str, strength: float, work: Path) -> dict[str, Any]:
    """``post.match_window`` on the window's background: the person found and followed on every
    ``H3SWAP_COLOUR_EVERY``-th frame (the whole frame when the vision models are missing)."""
    from opencine.gen.control import person_masks
    from opencine.gen.post import match_window

    every = H3SWAP_COLOUR_EVERY
    try:
        _, masks = person_masks(window[::every], subject)
    except Exception:  # noqa: BLE001 — no models / nobody found: measure the whole frame
        masks = None
    return match_window(clip, window, masks, strength, work, every=every)


# MiniMax-H3 Ref2VA (Space code, GPU duration and prompt checks in ``h3ref``)
REF_MAX_PIXELS = 1344 * 768  # pictures are sent at most at the full canvas area (the Space's ``match``)


def _ref_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "9:16") -> float:
    """The Space's duration for ``n_images`` pictures, a first-frame guide counted (its 1 frame)."""
    _, h, w = h3_canvas(aspect, s.hf_video_quality)
    return h3ref.gpu_s(w, h, h3_frames(seconds), s.hf_video_steps, n_images)


def _ref_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    label, h, w = h3_canvas(a, s.hf_video_quality)
    return {"images": [_file(x) for x in f], "first": None, "prompt": p, "frames": h3_frames(sec),
            "width": w, "height": h, "steps": s.hf_video_steps, "seed": seed, "canvas": label}


# MiniMax-H3 face pass (Space code, sizes and GPU duration in ``h3face``, the CPU side in
# ``opencine.gen.face``): on the Ref2VA Space, files = the crop clip, then the identity pictures
FACE_VERSION = 1  # how crops are made, sent and pasted back (opencine.gen.face): part of the cache key


def _face_declared(seconds: float, n_images: int, s: GenSettings, aspect: str = "1:1") -> float:
    """The Space's duration: ``n_images`` counts the files sent, the crop clip first."""
    return h3face.gpu_s(s.hf_face_size, h3face.frames(round(seconds * h3face.FPS)), s.hf_face_steps,
                        max(1, n_images - 1))


def _face_call(f: list[Path], p: str, sec: float, seed: int, s: GenSettings, a: str) -> dict[str, Any]:
    return {"crop": _file(f[0]), "images": [_file(x) for x in f[1:]], "prompt": p,
            "frames": h3face.frames(round(sec * h3face.FPS)), "size": s.hf_face_size,
            "steps": s.hf_face_steps, "seed": seed}


def edit_knobs(role: str, s: GenSettings) -> dict[str, Any]:
    """Settings that change a video-to-video output of ``role`` (its cache fingerprint)."""
    if role == "swap_h3":
        knobs = {"size": s.hf_h3swap_size, "steps": s.hf_h3swap_steps,
                 "upscale": s.hf_h3swap_up_sigmas if s.hf_h3swap_upscale else None,
                 "lora": s.hf_h3swap_lora, "turbo": s.hf_h3swap_turbo,
                 "colour": s.hf_h3swap_colour, "cutout": s.hf_h3swap_cutout, "letterbox": True}
        if s.hf_h3swap_upscale:  # what the second pass runs with
            lms = _h3swap_lms(s)
            knobs.update(upscaler=s.hf_h3swap_upscaler, lms=[lms, s.hf_h3swap_lms_file,
                                                            s.hf_h3swap_lms_caption] if lms else None)
        return knobs
    if role == "swap":
        return {"size": s.hf_scail_size, "fps": s.hf_scail_fps, "steps": s.hf_scail_steps,
                "cfg": s.hf_scail_cfg, "shift": s.hf_scail_shift, "lora": s.hf_scail_lora,
                "solver": s.hf_scail_solver, "replace": s.hf_scail_replace, "grow": s.hf_scail_grow}
    return {"size": s.hf_restyle_size, "strength": s.hf_restyle_strength}


def trim_frames(path: Path, n: int) -> int:
    """Cut ``path`` to its first ``n`` frames when it has more (a model padded to its own
    count); → the frames it keeps."""
    from opencine.core import ffmpeg
    from opencine.gen.post import frame_count
    from opencine.gen.qc import video_stream

    have = frame_count(video_stream(path))
    if have <= n:
        return have
    tmp = path.with_name(f"{path.stem}.trim{path.suffix}")
    ffmpeg.run(["-y", "-i", str(path), "-frames:v", str(n), "-c:v", "libx264", "-preset",
                "medium", "-crf", "10", "-c:a", "copy", str(tmp)])
    tmp.replace(path)
    return n


def _rife_disabled(*a: Any) -> dict[str, Any]:
    # Space signature is interpolate(video, factor="2x frames"), ≤1280 px, ≤6.5 s
    raise ProviderError("RIFE role is not wired (Space call shape unverified); clips are "
                        "retimed on CPU by opencine.gen.post instead")


SPACES: dict[str, SpaceSpec] = {
    # Official Qwen-Image-2.1 Space, bf16, 40 steps: top open-weights editor (Artificial
    # Analysis, 2026-10); Qwen Research License (non-commercial). Condition images at ~1 MP,
    # output up to 2K.
    "image": SpaceSpec(
        "Qwen/Qwen-Image-2.1", "/mc_edit", "Qwen-Image-2.1", "large", _q21_declared,
        lambda f, p, sec, seed, s, a: {
            "images": [{"image": _file(x), "caption": None} for x in f], "prompt": p,
            "seed": seed, "width": q21_size(a, s.hf_image_quality)[0],
            "height": q21_size(a, s.hf_image_quality)[1],
        },
        files={"requirements.txt": _Q21_REQUIREMENTS},
        patches={"app.py": _Q21_PATCHES},
        out_size=lambda a, s: q21_size(a, s.hf_image_quality),
        revision="5694809f4c56a4e8c40787b5392cecac36493c5d", repo="qwen-image-2-1"),
    # MiniMax-H3 (33B, bf16 AdaLN-pruned = released weights folded, CFG-distilled, 28 steps):
    # first frame, or first + last frame. The prompt is encoded by the conditioner Space.
    "video": SpaceSpec(
        "multimodalart/minimax-h3", "/generate", "MiniMax-H3", "xlarge", _h3_declared,
        lambda f, p, sec, seed, s, a: _h3_call(f, p, sec, seed, s, a),
        side_s=_h3_side,
        secrets=("HF_TOKEN",),  # to reach the private conditioner duplicate
        variables=lambda s: {"H3_CONDITIONER": s.hf_space("cond").id} if s.hf_space("cond") else {},
        patches={"app.py": _H3_PATCHES},
        needs=("cond",),
        revision="ca7ac4974e2a1838798b6a875bcff5144b9f8c9c", repo="minimax-h3"),
    # MiniMaxAI's Space: MiniMax-H3 (released bf16 weights) + larryvrh Turbo LoRA (Apache-2.0)
    # folded at startup; same duration formula and call as "video", ~1/4 of the steps
    "video_turbo": SpaceSpec(
        "MiniMaxAI/MiniMax-H3-Turbo-Lora", "/generate", "MiniMax-H3+Turbo-LoRA", "xlarge",
        _h3_declared,
        lambda f, p, sec, seed, s, a: _h3_call(f, p, sec, seed, s, a),
        side_s=_h3_side,
        secrets=("HF_TOKEN",),
        variables=lambda s: ({"H3_CONDITIONER": s.hf_space("cond").id, "H3_LORA": H3_TURBO_LORA,
                              "H3_LIGHTX": "off"} if s.hf_space("cond") else {}),
        patches={"app.py": _H3_TURBO_PATCHES, "requirements.txt": _COND_PATCHES},
        tune=lambda s: s.model_copy(update={"hf_video_steps": s.hf_video_turbo_steps}),
        needs=("cond",),
        revision="9a76d679e4197339c32b115d3eebe435d43b373a", repo="minimax-h3-turbo"),
    "cond": SpaceSpec(
        "multimodalart/qwen3vl-conditioner", None, "MiniMax-H3 conditioner (Qwen3-VL)", "xlarge",
        lambda sec, n, s, a: 45, lambda *a: {},
        patches={"requirements.txt": _COND_PATCHES},
        revision="71cd9793a2dde2c195ae99043368993f8391002b", repo="h3-conditioner"),
    "rife": SpaceSpec(
        "Upsampler/rife-frame-interpolation", None, "RIFE-4.25-x2", "large",
        lambda sec, n, s, a: 13, _rife_disabled, repo="rife"),
    # SCAIL-2 (MIT weights, Apache code; 14B on Wan2.1-I2V): the person of a clip replaced by
    # the character of a reference on black, background and camera kept; fp32 DiT → xlarge
    "swap": SpaceSpec(
        "fffiloni/SCAIL-2", "/mc_scail", "SCAIL-2", "xlarge", _scail_declared,
        lambda f, p, sec, seed, s, a: _scail_call(f, p, sec, seed, s, a),
        variables=_scail_variables, patches=_SCAIL_PATCHES,
        files={"wan/utils/lora.py": _SCAIL_FUSE},
        out_size=lambda a, s: SCAIL_SIZES[s.hf_scail_size][a],
        fps=lambda s: s.hf_scail_fps, frames=scail_frames, max_frames=SCAIL_MAX_FRAMES,
        revision="325cdada0cc6e78ed37fe62c9e3beaf188e19989", repo="scail-2"),
    # MiniMax-H3 Ref2VA (H3 Community License) + Alissonerdx character-swap LoRA v1.1 (BFS Character
    # Swap licence: no hosted service; own use only) + lightx2v ref2v Turbo, folded; the window is
    # an aligned guide, the character <Picture 1>; the prompt is encoded by the conditioner Space
    "swap_h3": SpaceSpec(
        "linoyts/minimax-h3-character-swap", "/mc_charswap", "MiniMax-H3+CharacterSwap-v1.1",
        "xlarge", _h3swap_declared, lambda f, p, sec, seed, s, a: _h3swap_call(f, p, sec, seed, s, a),
        side_s=_h3swap_side,  # image-only /encode_ref2va on the conditioner, xlarge
        secrets=("HF_TOKEN",),  # to reach the private conditioner duplicate
        variables=lambda s: h3swap.variables(
            s.hf_h3swap_lora, s.hf_h3swap_turbo,
            s.hf_space("cond").id if s.hf_space("cond") else None, s.hf_h3swap_lms_file,
            s.hf_h3swap_upscaler),
        files={"mc_api.py": h3swap.API}, patches=h3swap.PATCHES, needs=("cond",),
        out_size=_h3swap_out, fps=lambda s: h3swap.FPS, frames=h3swap.frames,
        max_frames=h3swap.MAX_FRAMES,
        revision="0f4a2f75a73c3e780512ad22eac1e39de01a4a1d", repo="h3-charswap",
        api_for=_h3swap_api),
    # MiniMax-H3 Ref2VA from reference pictures: a second duplicate of the character-swap
    # Space with its swap LoRA off, the ref2v Turbo LoRA folded; character / place pictures as
    # <Picture N> + native reference blocks, an optional first frame also as a guide on frame 0
    "video_ref": SpaceSpec(
        "linoyts/minimax-h3-character-swap", "/mc_ref2va", "MiniMax-H3-Ref2VA+Turbo", "xlarge",
        _ref_declared, lambda f, p, sec, seed, s, a: _ref_call(f, p, sec, seed, s, a),
        side_s=lambda s: 2 * 45,  # one /encode_ref2va on the conditioner, xlarge
        secrets=("HF_TOKEN",),  # to reach the private conditioner duplicate
        variables=lambda s: h3ref.variables(
            s.hf_video_ref_turbo, s.hf_space("cond").id if s.hf_space("cond") else None),
        files={"mc_api.py": h3ref.API + h3face.API},
        patches={"app.py": h3ref.PATCHES["app.py"] + h3face.PATCHES}, needs=("cond",),
        tune=lambda s: s.model_copy(update={"hf_video_steps": s.hf_video_ref_steps}),
        revision="0f4a2f75a73c3e780512ad22eac1e39de01a4a1d", repo="h3-ref"),
    # MiniMax-H3 face pass: a tracked face crop redrawn from its own latent on the Ref2VA
    # Space above (no duplicate of its own: config.SPACE_HOSTS), the identity picture as
    # <Picture 1>, the clip's soundtrack locked into the audio rows
    "face": SpaceSpec(
        "linoyts/minimax-h3-character-swap", h3face.API_NAME, "MiniMax-H3-Ref2VA+Turbo face pass",
        "xlarge", _face_declared, lambda f, p, sec, seed, s, a: _face_call(f, p, sec, seed, s, a),
        side_s=lambda s: 2 * 45,  # one image-only /encode_ref2va on the conditioner, xlarge
        needs=("cond",), out_size=lambda a, s: (s.hf_face_size, s.hf_face_size),
        fps=lambda s: h3face.FPS, frames=h3face.frames, max_frames=h3face.MAX_FRAMES,
        revision="0f4a2f75a73c3e780512ad22eac1e39de01a4a1d"),
    # LTX-2.3 distilled (fp8) + Union-Control IC-LoRA fused (LTX-2 Community License): restyle
    # from a styled first frame, structure from a depth / edge clip; 22B + Gemma → xlarge
    "restyle": SpaceSpec(
        "linoyts/LTX-2-3-sync", "/mc_restyle", "LTX-2.3+Union-Control", "xlarge", _ltx_declared,
        lambda f, p, sec, seed, s, a: _ltx_call(f, p, sec, seed, s, a),
        # Gemma 3 (the text encoder) is gated: the token's account must accept its licence
        secrets=("HF_TOKEN",),
        patches=_LTX_PATCHES, out_size=lambda a, s: LTX_SIZES[s.hf_restyle_size][a],
        fps=lambda s: LTX_FPS, frames=ltx_frames, max_frames=LTX_MAX_FRAMES,
        revision="43260a4dedca79a01fd27adc48314f1c36f51770", repo="ltx-restyle",
        gated=("google/gemma-3-12b-it-qat-q4_0-unquantized",)),
    # FlashVSR+ turbo (Apache-2.0 code + weights; 1.3B on Wan2.1, ~7 GB → large): one-step video
    # super-resolution of a whole generated clip in one call, same frames and timing
    "enhance": SpaceSpec(
        "hugging-apps/flashvsr-plus", "/mc_enhance", "FlashVSR+-turbo", "large", _enhance_declared,
        lambda f, p, sec, seed, s, a: _enhance_call(f, sec, s, a),
        files={"mc_api.py": _FLASHVSR_API}, patches=_FLASHVSR_PATCHES,
        out_size=lambda a, s: enhance_canvas(a, s.hf_enhance_short),
        revision="a1673237bcbae9ad5dba060d9661d05486cd4071", repo="flashvsr-enhance"),
    # ACE-Step 1.5 (MIT code + weights; outputs usable commercially): instrumental background
    # music, 10–600 s, BPM and key settable
    "music": SpaceSpec(
        "ACE-Step/Ace-Step-v1.5", "/mc_music", "ACE-Step-1.5-XL-turbo", "large", _ace_declared,
        lambda f, p, sec, seed, s, a: {"caption": p, "seconds": sec, "seed": seed, "bpm": 0,
                                       "keyscale": "", "thinking": True},
        patches={"app.py": _ACE_PATCHES, "requirements.txt": _ACE_REQUIREMENTS},
        revision="7403460e9b34972f760317b56048f6cb9d4a3a11", repo="ace-step-music"),
}


# `--model` choices per kind → (role, settings it overrides)
MODELS: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {
    "image": {
        "qwen-2.1-1k": ("image", {"hf_image_quality": "1k"}),  # 768x1344: 48 s reserved
        "qwen-2.1": ("image", {"hf_image_quality": "2k"}),  # 1536x2688: 215 s
    },
    "video": {
        "h3-turbo": ("video_turbo", {}),  # 7 steps: 274 s for 5 s `full`
        "h3-turbo-hq": ("video_turbo", {"hf_video_turbo_steps": H3_TURBO_HQ_STEPS}),  # 314 s
        "h3": ("video", {}),  # 28 steps: 700 s
        # Ref2VA from character / place pictures: 8 Euler steps over the 8-step ref2v Turbo;
        # 2 pictures, 5 s full: ~556 s reserved; hq = 16 steps (ComfyUI: Turbo barely applies refs)
        "h3-ref": ("video_ref", {}),
        "h3-ref-hq": ("video_ref", {"hf_video_ref_steps": 16}),
    },
    # video-to-video, by edit mode
    "swap": {
        "scail2": ("swap", {}),  # 512x896, 16 fps, 8 steps + distill LoRA: ~420 s for 5 s
        "scail2-hq": ("swap", {"hf_scail_size": "high"}),  # 704x1280: ~1200 s (the cap)
        # MiniMax-H3 guide-latent swap: new face, body and clothing, background regenerated
        # along the guide; 576x1024, 8 steps: 574 s reserved for 5 s (conditioner included)
        "h3-swap": ("swap_h3", {}),
        # the LoRA author's two passes: 416x736, then latent upscale to 768x1344 + 3 steps with
        # the LMS LoRA (0.7, captioned prompt): 874 s
        "h3-swap-hq": ("swap_h3", {"hf_h3swap_size": "low", "hf_h3swap_upscale": True}),
    },
    "restyle": {
        "ltx-union": ("restyle", {}),  # 640x1152, 24 fps: 228 s for 5 s
        "ltx-union-hq": ("restyle", {"hf_restyle_size": "high"}),  # 768x1408: 298 s
    },
}
# What runs when a slot names no model. The MCP tools let the Director pick only the choices
# in ``DIRECTOR_MODELS`` (film mode asks the user first); the rest are for manual runs
# (``opencine gen ... --model``).
DEFAULT_MODELS = {"image": "qwen-2.1", "video": "h3-turbo", "swap": "scail2",
                  "restyle": "ltx-union"}
DIRECTOR_MODELS = {"video": ("h3-turbo", "h3-turbo-hq", "h3", "h3-ref", "h3-ref-hq"),
                   "swap": ("h3-swap", "h3-swap-hq")}
REF_MODEL = "h3-ref"  # what a clip with refs runs when it names no model


def resolve_model(use: str, choice: str | None,
                  settings: GenSettings | None = None) -> tuple[str, str, GenSettings]:
    """``(model name, role, settings the call runs with)`` for a ``--model`` choice of ``use``
    (image | video); None = ``DEFAULT_MODELS[use]``."""
    s = settings or get_gen_settings()
    name = choice or (s.hf_swap_model if use == "swap" else DEFAULT_MODELS[use])
    options = MODELS[use]
    if name not in options:
        raise ProviderError(f"hf {use} model {name!r}: choose one of {', '.join(options)}")
    role, over = options[name]
    return name, role, SPACES[role].tune(s.model_copy(update=over) if over else s)


def reservation_s(role: str, seconds: float = 5.0, n_images: int = 1,
                  settings: GenSettings | None = None, aspect: str = "9:16") -> float:
    """Quota the server reserves before the main booking of a call: declared × 1.5 (large)
    or × 2 (xlarge)."""
    spec = SPACES[role]
    s = spec.tune(settings or get_gen_settings())
    return spec.declared_s(seconds, n_images, s, aspect) * (2.0 if spec.size == "xlarge" else 1.5)


# ---- error strings ----------------------------------------------------------
class ServerQuotaExceeded(QuotaExhausted):
    """Server-side quota refusal; carries the server's ``left_s`` for the local book."""

    def __init__(self, message: str, reset_at: datetime | None, left_s: float):
        super().__init__(message, reset_at)
        self.left_s = left_s


class GpuAborted(ProviderError):
    """The task ran past its declared duration; the reservation was spent."""


_QUOTA = re.compile(r"exceeded your (Pro |free )?ZeroGPU quota \((\d+(?:\.\d+)?)s requested "
                    r"vs\. (\d+(?:\.\d+)?)s left\)\. Try again in ([^.]+)\.")
_GUEST = re.compile(r"Falling back to IP-based quotas|Authenticate with a Hugging Face token")
_GUEST_HINT = "HF token not recognised (counted as a guest); check HF_TOKEN / PRO status"
_RULES: list[tuple[re.Pattern[str], type[ProviderError], str]] = [
    (re.compile(r"exceeded your ZeroGPU runs limit"), QuotaExhausted,
     "ZeroGPU runs limit reached (unpublished count); wait and retry later"),
    (re.compile(r"requested GPU duration \(\d+s\) is larger than the maximum allowed"),
     ProviderError, "declared @spaces.GPU duration too long; lower it on the duplicate"),
    (re.compile(r"No GPU was available"), TransientError, "no GPU available / queue expired"),
    (re.compile(r"too many ZeroGPU credits allocated to running tasks"), TransientError,
     "other ZeroGPU tasks still hold quota; retry when they finish"),
    (re.compile(r"GPU task aborted"), GpuAborted,
     "GPU task aborted (ran past its declared duration); raise duration on the duplicate"),
]


def parse_wait(text: str) -> timedelta | None:
    """``"8:36:04"`` / ``"1 day, 2:03:04"`` → timedelta."""
    m = re.fullmatch(r"\s*(?:(\d+) days?, )?(\d+):(\d{2}):(\d{2})\s*", text)
    if not m:
        return None
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return timedelta(days=d, hours=h, minutes=mi, seconds=s)


def classify_hf_error(message: str, now: datetime | None = None) -> ProviderError:
    """Map a Space/ZeroGPU error message to a typed error.

    A guest fallback (or the *free* quota, which a PRO token never sees) is an auth problem,
    checked first so its guest numbers never reach the local quota book.
    """
    now = now or datetime.now()
    m = _QUOTA.search(message)
    if _GUEST.search(message) or (m and m.group(1) == "free "):
        return AuthError(f"{_GUEST_HINT}: {message.strip()[:300]}")
    if m:
        wait = parse_wait(m.group(4))
        return ServerQuotaExceeded(
            f"HF ZeroGPU quota exhausted ({m.group(2)}s requested vs {m.group(3)}s left)",
            reset_at=now + wait if wait else None, left_s=float(m.group(3)))
    for pattern, cls, hint in _RULES:
        if pattern.search(message):
            return cls(f"{hint}: {message.strip()[:300]}")
    return ProviderError(message.strip()[:500] or "unknown Space error")


# ---- remote handles -----------------------------------------------------------
def _make_client(space_id: str, token: str | None):
    from gradio_client import Client

    return Client(space_id, token=token, verbose=False)


def _make_api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _whoami(token: str) -> dict[str, Any]:
    """Who a token belongs to (``hf_accounts.identify``); a token the Hub refuses is an
    ``AuthError``."""
    try:
        return _make_api(token).whoami()
    except Exception as e:
        cls = AuthError if re.search(r"\b401\b|[Ii]nvalid", str(e)) else TransientError
        raise cls(f"HF token …{token[-4:]}: whoami failed: {type(e).__name__}: "
                  f"{str(e)[:200]}") from e


def _stage(runtime) -> str:
    return str(getattr(runtime.stage, "value", runtime.stage))


def _find_file(obj: Any) -> str | None:
    """First existing file path inside a Gradio result (str, dict, list/tuple)."""
    if isinstance(obj, str | Path):
        return str(obj) if Path(obj).is_file() else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list | tuple):
        for x in obj:
            if found := _find_file(x):
                return found
    return None


def _job_stage(job) -> str | None:
    try:
        code = job.status().code
    except Exception:  # status is best effort
        return None
    return str(getattr(code, "value", code))


class CallHung(TimeoutError):
    """A Space call outlived its deadline; ``gpu_s`` = seconds seen in ``PROCESSING``."""

    def __init__(self, waited_s: float, gpu_s: float | None):
        super().__init__(f"no answer after {waited_s:.0f}s; cancelled")
        self.gpu_s = gpu_s


def _wait_job(job, timeout_s: float) -> float | None:
    """Block until a ``gradio_client`` Job is done → seconds seen in ``PROCESSING`` (the
    GPU part, without upload/queue), or None when that stage was never observed. Past
    ``timeout_s`` the job is cancelled and ``CallHung`` raised."""
    t0, t_proc = _clock(), None
    while not job.done():
        if t_proc is None and _job_stage(job) == "PROCESSING":
            t_proc = _clock()
        if _clock() - t0 > timeout_s:
            with contextlib.suppress(Exception):  # cancelling is best effort
                job.cancel()
            raise CallHung(_clock() - t0, _clock() - t_proc if t_proc is not None else None)
        _poll(POLL_S)
    return _clock() - t_proc if t_proc is not None else None


def _find_seed(obj: Any) -> int | None:
    if isinstance(obj, list | tuple):
        ints = [int(x) for x in obj if isinstance(x, int | float) and not isinstance(x, bool)
                and x == int(x)]  # Slider outputs come back as floats
        return ints[-1] if ints else None
    return None


# ---- provider ---------------------------------------------------------------
class HfSpaceProvider(GenProvider):
    name = "hf"
    max_parallel = 1  # one call per Space at a time
    ref_model = REF_MODEL

    def __init__(self, settings: GenSettings | None = None, token: str | None = None,
                 now: Callable[[], datetime] = datetime.now, tokens: list[str] | None = None):
        """``tokens``: the accounts, in order (default ``OPENCINE_HF_TOKENS``, else ``HF_TOKEN``);
        ``token``: a single one."""
        self.s = settings or get_gen_settings()
        self.now = now
        if tokens is None:
            tokens = [token] if token is not None else get_settings().hf_account_tokens()
        self._tokens = tokens
        self._accounts: list[Account] | None = None  # made on first use (may ask the Hub who)
        self._active: Account | None = None
        self._clients: dict[tuple[str, str], Any] = {}

    # ---- accounts ------------------------------------------------------------
    @property
    def accounts(self) -> list[Account]:
        if self._accounts is None:
            self._accounts = make_accounts(self._tokens, self.s, _whoami, self.now)
            self._active = self._accounts[0]
        return self._accounts

    @property
    def active(self) -> Account:
        """The account calls go to until a quota runs out."""
        return self._active if self._accounts is not None else self.accounts[0]

    @property
    def token(self) -> str | None:
        return self.active.token or None

    @token.setter
    def token(self, value: str | None) -> None:
        self._tokens, self._accounts = [value] if value else [], None

    @property
    def book(self) -> QuotaBook:
        return self.active.book

    def _space(self, role: str, acct: Account | None = None) -> SpaceRef:
        acct = acct or self.active
        ref = acct.space(role, self.s)
        if ref is None:
            who = f" of account {acct.name}" if acct.name else ""
            raise ProviderError(
                f"OPENCINE_HF_SPACE_{role.upper()} has no Space{who}. Set the duplicates up with "
                f"`opencine gen hf-setup` (one role: opencine gen hf-duplicate {role} --to <you>/<name>; "
                f"upstream {SPACES[role].upstream})")
        return ref

    def _client(self, role: str, acct: Account | None = None):
        acct = acct or self.active
        key = (self._space(role, acct).id, acct.token)
        if key not in self._clients:
            self._clients[key] = _make_client(*key)
        return self._clients[key]

    def _pick(self, roles: list[str], cap: float) -> Account:
        """The account the next call runs on (``roles``: the Space and the ones it calls): the
        first, in order, whose daily quota still holds ``cap``. When none does: with
        ``hf_overquota`` the one with the most left (the Hub bills its credits past the
        quota), else ``QuotaExhausted``."""
        able = [a for a in self.accounts if a.serves(roles, self.s)]
        if not able:
            for r in roles:
                self._space(r)  # raises: not set on the account in use
            raise ProviderError(f"no account has every Space of {', '.join(roles)}")
        live = [a for a in able if not a.out(self.now())]
        if len(live) == 1 and self.s.hf_overquota:
            return live[0]  # the server decides
        for a in live:
            if a.book.status()["left"] < cap:  # the estimate may be stale: ask the Hub first
                self._read_hub(a, fresh_s=60)
            if a.book.status()["left"] >= cap:
                return a
        if live and self.s.hf_overquota:
            return max(live, key=lambda a: a.book.status()["left"])
        if len(able) == 1 and live:
            able[0].book.check(cap)  # raises with its numbers
        resets = [t for a in able if (t := a.back_at())]
        left = ", ".join(f"{a.label} {a.book.status()['left']:.0f}s" for a in able)
        raise QuotaExhausted(f"HF quota (local book): {cap:.0f}s reservation > what is left ({left} "
                             f"of {self.s.hf_quota_s:.0f}s each)", reset_at=min(resets, default=None))

    def _switch(self, acct: Account, roles: list[str]) -> None:
        """Calls go to ``acct`` from now on; its Spaces are woken first (they may sleep)."""
        if acct is not self.active:
            self._active = acct
            self.warm(roles)

    def _call(self, role: str, files: list[Path], prompt: str, seconds: float,
              seed: int | None, aspect: str = "9:16", settings: GenSettings | None = None,
              extra: dict[str, Any] | None = None) -> tuple[str, int | None, float, float, Any]:
        """Run one Space call → (downloaded file, seed, latency_s, quota charged, raw result).
        ``extra``: endpoint arguments over the ones ``build`` makes.

        The seed is drawn here when not given, so every output records the seed it used.
        The call runs on the account ``_pick`` chooses; a quota refusal moves it to the next
        account that can take it, and reaches the caller only when none is left. The quota
        charged counts every account's attempts.
        """
        if not self.token:
            raise AuthError("HF_TOKEN is not set; calls would use the guest IP quota")
        spec = SPACES[role]
        s = spec.tune(settings or self.s)
        if spec.api_name is None:
            raise ProviderError(f"{role}: not called by the engine")
        reserve = reservation_s(role, seconds, len(files), s, aspect)
        # side bookings (the conditioner) come first: all of it must fit, not just the denoise
        cap = reserve + spec.side_s(s)
        seed = seed if seed is not None else random.randint(0, MAX_SEED)
        roles = [*spec.needs, role]
        t0, used = _clock(), 0.0
        while True:
            try:
                acct = self._pick(roles, cap)
            except QuotaExhausted as e:
                raise e.spent(round(used, 1), "gpu_s") from None
            self._switch(acct, roles)
            try:
                path, seed_out, spent, result = self._attempts(
                    acct, role, s, files, prompt, seconds, seed, aspect, extra, reserve, cap)
            except ProviderError as e:
                used += e.quota_used
                if isinstance(e, QuotaExhausted):
                    acct.refused, acct.refused_until = True, e.reset_at
                    now = self.now()
                    if any(a.serves(roles, self.s) and not a.out(now) for a in self.accounts):
                        continue
                e.spent(round(used, 1), "gpu_s")
                raise
            return path, seed_out, _clock() - t0, used + spent, result

    def _attempts(self, acct: Account, role: str, s: GenSettings, files: list[Path], prompt: str,
                  seconds: float, seed: int, aspect: str, extra: dict[str, Any] | None,
                  reserve: float, cap: float) -> tuple[str, int, float, Any]:
        """The call on ``acct``'s duplicate, retried on transient errors → (file, seed, quota
        charged to ``acct``, raw result).

        Quota charged per attempt = time seen in ``PROCESSING`` × size factor (≤ ``cap``, the
        reservation plus the Space's side bookings); queue, upload and retry backoff are not
        charged. A successful attempt whose stage was never observed is charged its whole
        duration; an aborted task its whole reservation.
        """
        spec, space = SPACES[role], self._space(role, acct)
        factor = 2.0 if spec.size == "xlarge" else 1.0
        used = 0.0
        for attempt in range(self.s.hf_retries + 1):
            ta, gpu_s = _clock(), None
            try:
                job = self._client(role, acct).submit(
                    api_name=spec.api_for(s) if spec.api_for else spec.api_name,
                    **spec.build(files, prompt, seconds, seed, s, aspect) | (extra or {}))
                gpu_s = _wait_job(job, cap + self.s.hf_call_timeout_s)
                result = job.result()
                break
            except Exception as e:  # gradio AppError / httpx / anything from the Space
                if isinstance(e, CallHung):
                    gpu_s = e.gpu_s
                    err: ProviderError = TransientError(f"{space.id}: {e}")
                elif isinstance(e, httpx.HTTPError | ConnectionError | TimeoutError):
                    err = TransientError(f"{space.id}: network: {e}")
                else:
                    err = classify_hf_error(str(e), self.now())
                if isinstance(err, ServerQuotaExceeded):
                    acct.book.sync(err.left_s, err.reset_at)
                spent = reserve if isinstance(err, GpuAborted) else (
                    min(gpu_s * factor, cap) if gpu_s is not None else 0.0)
                if spent:
                    acct.book.charge(spent)
                    used += spent
                if err.retryable and attempt < self.s.hf_retries:
                    _sleep(min(10 * 2**attempt, 60))
                    continue
                raise err.spent(round(used, 1), "gpu_s") from e
        spent = min((gpu_s if gpu_s is not None else _clock() - ta) * factor, cap)
        acct.book.charge(spent)
        used += spent
        path = _find_file(result)
        if path is None:
            raise ProviderError(f"{space.id}{spec.api_name}: no file in result {result!r:.200}"
                                ).spent(round(used, 1), "gpu_s")
        found = _find_seed(result)
        return path, found if found is not None else seed, used, result

    def _result(self, role: str, src: str, out: Path, seed, latency, used,
                aspect: str | None = None, video: bool = False,
                extra: dict[str, Any] | None = None) -> GenResult:
        """Save the Space output to ``out`` (ffprobe-checked for clips). A failure here
        still carries the GPU time spent, for the ledger."""
        space = self._space(role)
        try:
            save_output(src, out)
            probe = {"probe": verify_video(out, aspect)} if video else {}
        except Exception as e:
            err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
            raise err.spent(round(used, 1), "gpu_s") from e
        return GenResult(path=out, provider=self.name, model=SPACES[role].model, seed=seed,
                         quota_used=round(used, 1), quota_unit="gpu_s",
                         latency_s=round(latency, 2),
                         extra={"space": space.id, "sha": space.sha, **self._account_extra(),
                                **probe, **(extra or {})})

    def _account_extra(self) -> dict[str, str]:
        """The account a result came from, when there are several to tell apart."""
        return {"account": self.active.name} if self.active.name else {}

    def edit_image(self, refs: list[Path], prompt: str, aspect: str = "9:16", *, out: Path,
                   seed: int | None = None, model: str | None = None) -> GenResult:
        """Qwen-Image-2.1 edit; a new camera angle is an instruction in the prompt. No
        ``refs``: text-to-image (the Space switches mode when it gets no image).
        ``model``: ``qwen-2.1`` (2K, 1536×2688 for 9:16, the default) | ``qwen-2.1-1k`` (1K).

        ``refs[0]`` is fitted to the still size (``still_size``, a fixed point of the model's
        ~1 MP input resize); other refs keep their whole frame, shrunk to that ~1 MP. The
        output is asked at ``q21_size``.
        """
        name, role, s = resolve_model("image", model, self.s)
        w, h = still_size(aspect)
        with tempfile.TemporaryDirectory() as tmp:
            files = [fit_canvas(refs[0], w, h, Path(tmp) / "ref0.png")] if refs else []
            files += [limit_pixels(r, Path(tmp) / f"ref{i}.png", QWEN_AREA)
                      for i, r in enumerate(refs[1:], 1)]
            src, seed_out, latency, used, _ = self._call(role, files, prompt, 0, seed, aspect,
                                                         settings=s)
        spec = SPACES[role]
        size = list(spec.out_size(aspect, s) if spec.out_size else (w, h))
        return self._result(role, src, out, seed_out, latency, used,
                            extra={"size": size, "prompt_sent": prompt, "model_choice": name})

    @staticmethod
    def _video_settings(canvas: str | None, s: GenSettings) -> GenSettings:
        if canvas is None or canvas == s.hf_video_quality:
            return s
        if canvas not in ("fast", "mid", "full"):
            raise ProviderError(f"hf video: canvas {canvas!r} is not fast, mid or full")
        return s.model_copy(update={"hf_video_quality": canvas})

    def video(self, first: Path | None, prompt: str, seconds: float = 5.0, aspect: str = "9:16",
              last: Path | None = None, *, out: Path, seed: int | None = None,
              canvas: str | None = None, model: str | None = None,
              refs: list[Path] | None = None) -> GenResult:
        """MiniMax-H3 from text alone (``first`` None), a first frame, or first + last frame;
        24 fps, with a soundtrack.
        ``model``: ``h3-turbo`` (Turbo LoRA, ``hf_video_turbo_steps``) | ``h3-turbo-hq`` (Turbo
        LoRA, 9 steps) | ``h3`` (28 steps); None = the default (``h3-turbo``). With ``refs``
        (character / place pictures): Ref2VA, ``h3-ref`` (8 steps) | ``h3-ref-hq`` (16 steps).

        Keyframes are fitted to the canvas exactly (the Space would stretch the first one).
        A Context-IR prompt gets its alignment line rebuilt and is sent without the
        conditioner's rewrite; otherwise the rewrite runs when ``hf_video_upsample`` is on, and
        what it wrote comes back in ``extra["refined_prompt"]`` (deterministic: reuse it).
        """
        low = max(MIN_SECONDS, self.s.hf_video_min_s)
        if not low - 1e-6 <= seconds <= MAX_SECONDS + 1e-6:
            raise ProviderError(f"hf clips are {low:g}-{MAX_SECONDS:g}s (OPENCINE_HF_VIDEO_MIN_S), "
                                f"asked {seconds}s")
        refs = list(refs or [])
        name, role, s = resolve_model("video", model or (REF_MODEL if refs else None), self.s)
        s = self._video_settings(canvas, s)
        if refs or role == "video_ref":
            return self._video_ref(name, role, s, first, last, refs, prompt, seconds, aspect, out,
                                   seed)
        label, h, w = h3_canvas(aspect, s.hf_video_quality)  # bad aspect: fail before upload
        if last is not None and first is None:
            raise ProviderError("hf video: a last frame needs a first frame")
        keyframes = [p for p in (first, last) if p is not None]
        prompt, is_ir = h3_prompt(prompt, seconds, len(keyframes))
        if is_ir:
            s = s.model_copy(update={"hf_video_upsample": False})
        with tempfile.TemporaryDirectory() as tmp:
            files = [fit_canvas(p, w, h, Path(tmp) / f"f{i}.png") for i, p in enumerate(keyframes)]
            src, seed_out, latency, used, result = self._call(
                role, files, prompt, seconds, seed, aspect, settings=s)
        refined, report = _h3_outputs(result)
        encoded = refined or prompt
        extra = {"canvas": label, "upsample": s.hf_video_upsample, "prompt_ir": is_ir,
                 "refined_prompt": refined, "report": report, "model_choice": name,
                 "steps": s.hf_video_steps,
                 # a second [Shot N] is a hard cut: a fault in an insert, unless the prompt
                 # asked for it (the service drops the flag then)
                 "flags": ["multishot"] if max(map(int, _SHOT.findall(encoded)), default=1) > 1
                 else []}
        return self._result(role, src, out, seed_out, latency, used, aspect, video=True,
                            extra=extra)

    def _video_ref(self, name: str, role: str, s: GenSettings, first: Path | None,
                   last: Path | None, refs: list[Path], prompt: str, seconds: float, aspect: str,
                   out: Path, seed: int | None) -> GenResult:
        """Ref2VA: ``refs`` (≤ ``h3ref.MAX_PICTURES`` with the first frame) at most the canvas area,
        the first frame fitted to the canvas (``<Picture 1>`` and the frame-0 guide), the prompt
        sent as it is (no alignment line, no rewrite)."""
        if role != "video_ref":
            raise ProviderError(f"hf video: refs need a Ref2VA model ({', '.join(MODELS['video'])} "
                                f"→ h3-ref | h3-ref-hq), not {name}")
        if not refs:
            raise ProviderError(f"{name}: give refs (the character / place pictures); without them "
                                "use h3-turbo")
        if last is not None:
            raise ProviderError("hf video: a refs clip takes a first frame at most, no last frame")
        pictures = len(refs) + (first is not None)
        if pictures > h3ref.MAX_PICTURES:  # the conditioner runs out of GPU memory: refused unpaid
            raise ProviderError(f"{name}: {pictures} pictures (the first frame counts); a refs clip "
                                f"takes ≤ {h3ref.MAX_PICTURES}")
        body = _IR_LINE.sub("", prompt.strip(), count=1).strip()
        if problems := h3ref.prompt_problems(body, pictures):
            raise ProviderError(f"{name}: {problems[0]}")
        label, h, w = h3_canvas(aspect, s.hf_video_quality)
        need = h3ref.need_s(w, h, h3_frames(seconds), s.hf_video_steps, pictures)
        if need > h3ref.GPU_MAX:
            raise ProviderError(f"{name}: {seconds:g} s at {w}x{h} with {pictures} pictures and "
                                f"{s.hf_video_steps} steps needs ~{need:.0f} s of GPU, over one "
                                f"call's {h3ref.GPU_MAX}; shorten the clip, use fewer pictures or h3-ref")
        with tempfile.TemporaryDirectory() as tmp:
            files = [fit_canvas(first, w, h, Path(tmp) / "first.png")] if first is not None else []
            files += [limit_pixels(r, Path(tmp) / f"ref{i}.png", REF_MAX_PIXELS)
                      for i, r in enumerate(refs)]
            over = {"images": [_file(x) for x in files[1:]], "first": _file(files[0])} if first else None
            src, seed_out, latency, used, result = self._call(
                role, files, body, seconds, seed, aspect, settings=s, extra=over)
        report = str(result[1]) if isinstance(result, list | tuple) and len(result) > 1 else ""
        extra = {"canvas": label, "model_choice": name, "steps": s.hf_video_steps,
                 "pictures": pictures, "first_guide": first is not None, "prompt_ir": True,
                 "report": report,
                 "flags": ["multishot"] if max(map(int, _SHOT.findall(body)), default=1) > 1 else []}
        return self._result(role, src, out, seed_out, latency, used, aspect, video=True,
                            extra=extra)

    def edit_video(self, source: Path, refs: list[Path], prompt: str, mode: str,
                   aspect: str = "9:16", *, out: Path, seed: int | None = None,
                   model: str | None = None, subject: str = "person",
                   control: str = "depth") -> GenResult:
        """Video-to-video on a footage window. ``swap``: SCAIL-2 (``refs[0]`` = the
        character in the pose and framing of the window's first frame; the person's masks and
        the cut-out are made here, ``subject`` names them) or MiniMax-H3 (``refs[0]`` = any
        full-body picture of the character, cut out onto grey here; the window is the guide;
        ``prompt`` = the character head to toe, put into the LoRA's template; colours pulled to
        the window's background after), chosen by ``model`` / ``OPENCINE_HF_SWAP_MODEL``;
        ``restyle``: LTX-2.3 Union-Control from ``refs[0]`` = the first frame restyled and a
        ``control`` clip (depth or canny) of the window made here.

        Frames go in 1:1 at the model's rate, padded to a count it takes; the answer is cut
        back to the window's frames and ``extra["covers_s"]`` says how much of it they span.
        """
        import numpy as np
        from PIL import Image

        from opencine.gen.control import control_video, pad_frames, read_rgb, swap_inputs, write_rgb
        from opencine.gen.qc import video_stream

        if aspect not in ("9:16", "16:9"):
            raise ProviderError(f"hf {mode}: aspect {aspect} not supported; use 9:16 or 16:9")
        name, role, s = resolve_model(mode, model, self.s)
        spec = SPACES[role]
        (w, h), fps = spec.out_size(aspect, s), spec.fps(s)
        seconds = float(video_stream(source)["duration"])
        if spec.max_frames and spec.frames(max(1, round(seconds * fps))) > spec.max_frames:
            raise ProviderError(f"{name}: {seconds:.2f} s is longer than one call takes "
                                f"({spec.max_frames / fps:.1f} s at {fps} fps); split the window")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            frames = read_rgb(source, w, h, fps)
            real = len(frames)
            if not real:
                raise ProviderError(f"{source.name}: no frames")
            n = min(spec.frames(real), spec.max_frames or spec.frames(real))
            frames = pad_frames(frames, n)
            real = min(real, n)
            still = fit_canvas(refs[0], w, h, work / "ref.png")
            extra: dict[str, Any] = {"model_choice": name, "mode": mode, "canvas": f"{w}x{h}",
                                     "fps_sent": fps, "frames_sent": n, "frames_real": real}
            if role == "swap":
                with Image.open(still) as im:
                    rgb = np.asarray(im.convert("RGB"))
                ref, ref_mask, masks, extra["masks"] = swap_inputs(
                    frames, rgb, fps, work, phrase=subject, grow=s.hf_scail_grow,
                    replace=s.hf_scail_replace)
                files = [write_rgb(frames, fps, work / "video.mp4"), ref, ref_mask, masks]
            elif role == "swap_h3":
                ref, extra["ref_cutout"] = h3swap_reference(refs[0], work, s.hf_h3swap_cutout)
                prompt = h3swap.prompt(prompt, subject)
                extra["prompt_sent"] = prompt
                files = [write_rgb(frames, fps, work / "guide.mp4"), ref]
            else:
                import cv2

                half = np.stack([cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                                 for f in frames])
                files = [control_video(half, control, fps, work / "control.mp4"), still]
                extra["control"] = control
            src, seed_out, latency, used, result = self._call(
                role, files, prompt, n / fps, seed, aspect, settings=s)
            if role == "swap_h3" and isinstance(result, list | tuple) and len(result) > 1:
                extra["report"] = str(result[1])
            res = self._result(role, src, out, seed_out, latency, used, aspect, video=True,
                               extra=extra)
            try:
                kept = trim_frames(out, real)
            except Exception as e:
                raise ProviderError(f"{out.name}: {e}").spent(res.quota_used, "gpu_s") from e
            if role == "swap_h3" and s.hf_h3swap_colour > 0:
                try:
                    res.extra["colour"] = h3swap_colour(out, frames[:kept], subject,
                                                        s.hf_h3swap_colour, work)
                except Exception as e:  # noqa: BLE001 — best effort: the clip stays as made
                    res.extra["colour"] = {"applied": False, "error": f"{type(e).__name__}: {e}"}
            if role == "swap_h3":  # the model may draw picture into the footage's black bars
                from opencine.gen.post import letterbox_rows, restore_letterbox

                top, bottom = letterbox_rows(frames[:kept])
                if top or bottom:
                    try:
                        res.extra["letterbox"] = restore_letterbox(out, top, bottom, h)
                    except Exception as e:  # noqa: BLE001 — best effort: the clip stays as made
                        res.extra["letterbox"] = {"error": f"{type(e).__name__}: {e}"}
        res.extra["covers_s"] = round(kept / fps, 4)
        if kept < real:
            res.extra["flags"] = ["length"]
        return res

    def music(self, caption: str, seconds: float, *, out: Path, bpm: int | None = None,
              key: str | None = None, seed: int | None = None) -> GenResult:
        """An instrumental track from ACE-Step 1.5 (``OPENCINE_HF_SPACE_MUSIC``) as WAV at ``out``;
        ``extra["meta"]`` holds what it used (BPM, key, length, caption the 5 Hz LM wrote)."""
        if not MUSIC_MIN_S <= seconds <= MUSIC_MAX_S:
            raise ProviderError(f"music is {MUSIC_MIN_S:g}-{MUSIC_MAX_S:g} s, asked {seconds} s")
        src, used_seed, latency, used, result = self._call(
            "music", [], caption, seconds, seed,
            extra={"bpm": int(bpm or 0), "keyscale": key or "", "thinking": True})
        meta: dict[str, Any] = {}
        if isinstance(result, list | tuple) and len(result) > 1 and isinstance(result[1], str):
            with contextlib.suppress(ValueError):
                meta = json.loads(result[1])
        return self._result("music", src, out, meta.get("seed", used_seed), latency, used,
                            extra={"meta": meta})

    def enhance(self, src: Path, *, out: Path, short: int | None = None) -> GenResult:
        """FlashVSR+ (turbo, one step, ``OPENCINE_HF_SPACE_ENHANCE``) on a whole clip: its frames at
        ``enhance_canvas`` for ``short`` (None = ``hf_enhance_short``), Lanczos down to
        ``enhance_size``, at its exact rate with its sound; ``length`` flags missing frames."""
        from opencine.gen.post import frame_count
        from opencine.gen.qc import fps_of, video_stream

        s = self.s if short is None else self.s.model_copy(update={"hf_enhance_short": short})
        v = video_stream(src)
        n, aspect = frame_count(v), f"{v['width']}:{v['height']}"
        (cw, ch), size = enhance_canvas(aspect, s.hf_enhance_short), enhance_size(aspect, s.hf_enhance_short)
        declared = _enhance_gpu_s(cw, ch, n, s.hf_enhance_compression)
        if declared >= ENHANCE_GPU_MAX:
            raise ProviderError(f"enhance: {n} frames at {cw}x{ch} need more than one call's "
                                f"{ENHANCE_GPU_MAX} s; use a shorter clip, 1080 or compression 4")
        with tempfile.TemporaryDirectory() as tmp:
            sent = enhance_input(src, Path(tmp) / "in.mp4")
            got, _, latency, used, _ = self._call("enhance", [sent], "", n / ENHANCE_RATE, 0,
                                                  aspect, settings=s)
            try:
                frames = enhance_finish(Path(got), src, fps_of(v), n, size, out)
            except Exception as e:  # the call was paid for: the error carries its GPU time
                err = e if isinstance(e, ProviderError) else ProviderError(f"{type(e).__name__}: {e}")
                raise err.spent(round(used, 1), "gpu_s") from e
        space = self._space("enhance")
        return GenResult(path=out, provider=self.name, model=SPACES["enhance"].model,
                         quota_used=round(used, 1), quota_unit="gpu_s", latency_s=round(latency, 2),
                         extra={"space": space.id, "sha": space.sha, **self._account_extra(),
                                "canvas": [cw, ch],
                                "size": list(size), "compression": s.hf_enhance_compression,
                                "color": s.hf_enhance_color, "declared_s": declared,
                                "frames": frames, "flags": ["length"] if frames < n else []})

    def face(self, crop: Path, refs: list[Path], prompt: str, strengths: list[float], *, out: Path,
             audio: Path | None = None, seed: int | None = None) -> GenResult:
        """The face pass on ``crop``: a square clip at ``hf_face_size``, 24 fps, ``17k + 5``
        frames following one face (``opencine.gen.face`` makes it), one denoise strength per frame, the
        identity from ``refs`` (≤ ``h3face.MAX_PICTURES``, at most the crop's area), ``audio`` the
        clip's soundtrack as 32 kHz 16-bit WAV (held clean so the mouth follows it; None = free).
        The redrawn crop, silent, at ``out``."""
        from opencine.gen.post import frame_count
        from opencine.gen.qc import video_stream

        s, size = self.s, self.s.hf_face_size
        if size % 32 or not 256 <= size <= 1024:
            raise ProviderError(f"face: OPENCINE_HF_FACE_SIZE {size} is not a multiple of 32 in 256-1024")
        n = frame_count(video_stream(crop))
        if n != h3face.frames(n) or len(strengths) != n:
            raise ProviderError(f"face: the crop must have 17k+5 frames, one strength each "
                                f"({n} frames, {len(strengths)} strengths)")
        if not refs or len(refs) > h3face.MAX_PICTURES:
            raise ProviderError(f"face: give 1-{h3face.MAX_PICTURES} identity pictures")
        need = h3face.need_s(size, n, s.hf_face_steps, len(refs))
        if need > h3ref.GPU_MAX:
            raise ProviderError(f"face: {n} frames at {size}² need ~{need:.0f} s of GPU, over one "
                                f"call's {h3ref.GPU_MAX}; use a shorter clip or OPENCINE_HF_FACE_SIZE=512")
        with tempfile.TemporaryDirectory() as tmp:
            pictures = [limit_pixels(r, Path(tmp) / f"ref{i}.png", size * size)
                        for i, r in enumerate(refs)]
            src, seed_out, latency, used, result = self._call(
                "face", [crop, *pictures], prompt, n / h3face.FPS, seed, "1:1",
                extra={"audio": _file(audio) if audio else None,
                       "strengths": ",".join(f"{v:.4f}" for v in strengths)})
        report = str(result[1]) if isinstance(result, list | tuple) and len(result) > 1 else ""
        return self._result("face", src, out, seed_out, latency, used, "1:1", video=True,
                            extra={"report": report, "size": size, "steps": s.hf_face_steps,
                                   "pictures": len(refs), "audio_lock": audio is not None,
                                   "declared_s": h3face.gpu_s(size, n, s.hf_face_steps, len(refs))})

    def interpolate(self, src: Path, out: Path) -> GenResult:
        """Not wired: the RIFE Space's call shape is unverified."""
        return _rife_disabled()

    def fingerprint(self, kind: Kind, model: str | None = None,
                    mode: str | None = None) -> dict[str, Any]:
        """The base Space's settings plus what the model choice (None = the default) runs
        with, so a new default or another choice never reuses a different model's output."""
        def space(role: str) -> str | None:
            ref = self.s.hf_space(role)
            return f"{ref.id}@{ref.sha}" if ref else None

        if kind == "edit":
            name, role, s = resolve_model(mode or "swap", model, self.s)
            return {"choice": {"name": name, "model": SPACES[role].model, "space": space(role),
                               **edit_knobs(role, s)}, "prep": EDIT_PREP_VERSION}
        if kind == "face":
            return {"model": SPACES["face"].model, "space": space("face"), "cond": space("cond"),
                    "size": self.s.hf_face_size, "steps": self.s.hf_face_steps,
                    "turbo": self.s.hf_video_ref_turbo, "prep": FACE_VERSION}
        if kind == "enhance":
            return {"model": SPACES["enhance"].model, "space": space("enhance"),
                    "short": self.s.hf_enhance_short, "compression": self.s.hf_enhance_compression,
                    "color": self.s.hf_enhance_color, "prep": ENHANCE_PREP_VERSION}
        if kind == "image":
            fp = {"model": SPACES["image"].model, "space": space("image"), "prep": PREP_VERSION}
        else:
            fp = {"model": SPACES["video"].model, "space": space("video"), "cond": space("cond"),
                  "prep": PREP_VERSION, "steps": self.s.hf_video_steps,
                  "canvas": self.s.hf_video_quality, "upsample": self.s.hf_video_upsample}
        name, role, s = resolve_model(kind, model, self.s)
        fp["choice"] = {"name": name, "model": SPACES[role].model, "space": space(role),
                        **({"cond": space("cond")} if "cond" in SPACES[role].needs else {}),
                        **({"steps": s.hf_video_steps} if kind == "video" else
                           {"quality": s.hf_image_quality}),
                        **({"turbo": s.hf_video_ref_turbo, "ref_prep": h3ref.VERSION}
                           if role == "video_ref" else {})}
        return fp

    def _read_hub(self, acct: Account, fresh_s: float | None = None) -> dict[str, Any] | None:
        """``acct``'s ZeroGPU quota as the Hub reports it, synced into its book; None when the
        Hub does not answer (network, or a fine-grained token without "Read billing usage"),
        ``OPENCINE_HF_QUOTA_API`` is off, or it was read less than ``fresh_s`` ago."""
        if not (acct.token and self.s.hf_quota_api):
            return None
        if fresh_s is not None and acct.hub_at is not None and _clock() - acct.hub_at < fresh_s:
            return None
        acct.hub_at = _clock()
        try:
            q = _make_api(acct.token).get_zero_gpu_quota()
        except Exception as e:  # noqa: BLE001 — the local book stands in
            acct.hub_error = f"{type(e).__name__}: {str(e)[:200]}"
            return None
        acct.hub_error = None
        reset = q.resets_at.astimezone().replace(tzinfo=None) if q.resets_at else None
        acct.book.sync(float(q.remaining), reset, hub=True)
        return {"base": float(q.base), "left": float(q.remaining),
                "reset_at": reset.isoformat(timespec="seconds") if reset else None,
                "overquota_used": float(q.overquota_used or 0)}

    def hub_quota(self) -> dict[str, Any] | None:
        """The quota of the account in use as the Hub reports it (``_read_hub``)."""
        return self._read_hub(self.active)

    def quota(self) -> dict[str, Any]:
        """ZeroGPU seconds left: one account's book (from the Hub when it answers), or with
        several, their sum (``reset_at`` = the earliest an account gets quota back) and each
        one under ``accounts``."""
        rows = []
        for a in self.accounts:
            hub = self._read_hub(a)
            st = a.book.status() | {"source": "hub" if hub else "local book"}
            if hub:
                st["overquota_used"] = hub["overquota_used"]
            elif a.hub_error:
                st["hub_error"] = a.hub_error
            rows.append(st | ({"account": a.name} if a.name else {}))
        if len(rows) == 1:
            return rows[0]
        sources = {r["source"] for r in rows}
        return {"provider": "hf", "unit": "gpu_s", "limit": sum(r["limit"] for r in rows),
                "used": round(sum(r["used"] for r in rows), 1),
                "left": round(sum(r["left"] for r in rows), 1), "window_start": None,
                "reset_at": min((r["reset_at"] for r in rows if r["reset_at"]), default=None),
                "source": sources.pop() if len(sources) == 1 else "mixed",
                "overquota_used": sum(r.get("overquota_used", 0) for r in rows), "accounts": rows}

    def fit(self, calls: list[tuple[str, str, float]]) -> dict[str, Any]:
        """Where a batch's calls ``(id, role, quota)`` land on the daily quotas left, in order, the
        way ``_pick`` places them: the first account with the role's Spaces that holds the call
        whole, its quota then counted as spent (one call never spans accounts, so the sum of
        what is left says nothing about a call). ``over``: ids no account holds by then; they run
        on credits with ``hf_overquota``, else the batch stops there. Reservations are counted,
        not the lower real spend: a batch that fits here fits."""
        for a in self.accounts:
            self._read_hub(a, fresh_s=60)
        now = self.now()
        left = {a: a.book.status()["left"] for a in self.accounts if not a.out(now)}
        placed: dict[str, str] = {}
        over: list[str] = []
        for cid, role, need in calls:
            roles = [*SPACES[role].needs, role]
            acct = next((a for a in left if a.serves(roles, self.s) and left[a] >= need), None)
            if acct is None:
                over.append(cid)
                continue
            left[acct] -= need
            placed[cid] = acct.label
        several = len(self.accounts) > 1
        return {"fits": not over, "over": over,
                **({"placed": placed} if several else {}),
                "left_after_s": ({a.label: round(v) for a, v in left.items()} if several
                                 else round(sum(left.values())))}

    def prepare(self, kind: Kind, slots: list[Any]) -> dict[str, Any]:
        """Take every account's quota from the Hub, choose the account the batch starts on and
        wake only the Spaces it needs there, once, before the first call."""
        if not self.token:
            raise AuthError("HF_TOKEN is not set; calls would use the guest IP quota")
        for a in self.accounts:
            self._read_hub(a)
        if kind in ("enhance", "face"):
            roles = [kind]
        else:
            roles = sorted({resolve_model(
                x.mode if kind == "edit" else kind,
                getattr(x, "model", None) or (REF_MODEL if getattr(x, "refs", None) and kind == "video"
                                              else None), self.s)[1] for x in slots})
        needs = [n for r in roles for n in SPACES[r].needs]
        self._active = self._pick([*dict.fromkeys(needs), *roles], 1.0)  # one with quota left
        return self.warm(roles)

    def warm(self, roles: list[str] | None = None, wait: bool = True) -> dict[str, Any]:
        """Wake Spaces (no GPU used) and check their sha against the pin: ``roles`` on the
        account in use; None = every configured role, on every account (keyed by account when
        there are several).

        Raises ``TransientError`` when a Space is still not running after
        ``hf_wake_timeout_s``, ``ProviderError`` when it is broken or unreachable.
        """
        if roles is None and len(self.accounts) > 1:
            out: dict[str, Any] = {}
            for a in self.accounts:
                try:
                    out[a.label] = self._warm(a, None, wait)
                except ProviderError as e:
                    out[a.label] = {"error": f"{type(e).__name__}: {e}"}
            return out
        return self._warm(self.active, roles, wait)

    def _warm(self, acct: Account, roles: list[str] | None, wait: bool) -> dict[str, Any]:
        roles = roles or [r for r in SPACES if acct.space(r, self.s) and r not in SPACE_HOSTS]
        # what a Space calls mid-request (the conditioner) must be up too, and woken first
        needed = [n for r in roles for n in SPACES[r].needs if n not in roles]
        roles = [*dict.fromkeys(needed), *roles]
        if not roles:
            raise ProviderError("no OPENCINE_HF_SPACE_* configured; run `opencine gen hf-setup`")
        api = _make_api(acct.token)
        out: dict[str, Any] = {}
        for role in roles:
            ref, t0 = self._space(role, acct), _clock()
            try:
                stage = _stage(api.get_space_runtime(ref.id))
                if stage == "SLEEPING":
                    try:  # visiting a sleeping Space restarts it (unverified for ZeroGPU)
                        self._clients[(ref.id, acct.token)] = _make_client(ref.id, acct.token)
                    except Exception:
                        api.restart_space(ref.id)
                elif stage in ("PAUSED", "STOPPED"):
                    api.restart_space(ref.id)
                while wait and stage != "RUNNING":
                    if stage.endswith("ERROR"):
                        raise ProviderError(f"{ref.id} is in {stage}; check the Space logs")
                    if _clock() - t0 > self.s.hf_wake_timeout_s:
                        raise TransientError(
                            f"{ref.id} still {stage} after {self.s.hf_wake_timeout_s:.0f}s; "
                            "rerun the batch once it is up")
                    _sleep(10)
                    stage = _stage(api.get_space_runtime(ref.id))
                sha = api.space_info(ref.id).sha
            except ProviderError:
                raise
            except Exception as e:  # huggingface_hub HTTP errors, network
                cls = (TransientError if isinstance(e, httpx.HTTPError | ConnectionError | TimeoutError)
                       else ProviderError)
                raise cls(f"{ref.id}: warm-up failed: {type(e).__name__}: {e}") from e
            out[role] = {"space": ref.id, "stage": stage, "sha": sha, "pinned": ref.sha,
                         "sha_ok": ref.sha is None or bool(sha and sha.startswith(ref.sha)),
                         "waited_s": round(_clock() - t0, 1)}
        return out


# what the Director runs, for ``quota_summary``: (use, seconds[, model choice])
QUOTA_CALLS: dict[str, tuple[Any, ...]] = {
    "image": ("image", 5.0), "clip_5s": ("video", 5.0), "clip_10s": ("video", 10.0),
    "clip_5s_h3": ("video", 5.0, "h3"), "clip_10s_h3": ("video", 10.0, "h3"),
    "clip_5s_ref": ("video", 5.0, "h3-ref"), "clip_10s_ref": ("video", 10.0, "h3-ref"),
    "music_30s": ("music", 30.0), "swap_5s": ("swap", 5.0), "restyle_5s": ("restyle", 5.0),
    "h3swap_5s": ("swap", 5.0, "h3-swap"), "h3swap_hq_5s": ("swap", 5.0, "h3-swap-hq"),
    "enhance_5s": ("enhance", 5.0), "enhance_10s": ("enhance", 10.0),
    "face_5s": ("face", 5.0), "face_10s": ("face", 10.0)}


def call_reserve_s(use: str, seconds: float, settings: GenSettings | None = None,
                   model: str | None = None) -> float:
    """Quota one call must find left: its reservation plus the side bookings (the
    conditioner) made inside it. ``model``: a choice of ``MODELS[use]``; None = the default.
    ``enhance``: an H3 clip (24 fps, 7:4) at ``hf_enhance_short``; a Ref2VA clip: 2 pictures
    (a character and a place); ``face``: the crop clip and one identity picture."""
    s = settings or get_gen_settings()
    if use in ("music", "enhance", "face"):
        role, s = use, SPACES[use].tune(s)
    else:
        _, role, s = resolve_model(use, model, s)
    images = 0 if use == "music" else 2 if role in ("video_ref", "face") else 1
    return reservation_s(role, seconds, images, s) + SPACES[role].side_s(s)


def slot_call(kind: str, slot: Any, settings: GenSettings | None = None) -> tuple[str, float]:
    """``(role, quota)`` of the call of one gen slot (``service.ImageSlot`` / ``VideoSlot``): the
    Space it runs on, and what it must find left on one account: its model's reservation for its
    inputs, length and frame, plus the conditioner."""
    s = settings or get_gen_settings()
    if kind == "image":
        _, role, s = resolve_model("image", slot.model, s)
        return role, reservation_s(role, 0, len(slot.refs), s, slot.aspect) + SPACES[role].side_s(s)
    refs = list(getattr(slot, "refs", None) or [])
    _, role, s = resolve_model("video", slot.model or (REF_MODEL if refs else None), s)
    s = HfSpaceProvider._video_settings(slot.canvas, s)
    if role == "video_ref":
        images = len(refs) + (1 if slot.first else 0)
    else:
        images = sum(1 for p in (slot.first, slot.last) if p)
    return role, reservation_s(role, slot.seconds, images, s, slot.aspect) + SPACES[role].side_s(s)


def slot_reserve_s(kind: str, slot: Any, settings: GenSettings | None = None) -> float:
    """Quota the call of one gen slot must find left on one account (``slot_call``)."""
    return slot_call(kind, slot, settings)[1]


def _call_role(use: str, model: str | None, s: GenSettings) -> str:
    return use if use in ("music", "enhance", "face") else resolve_model(use, model, s)[1]


def _back(row: dict[str, Any]) -> str:
    if row.get("reset_at"):
        return f"back to {row['limit']:g} s at {row['reset_at']} (24 h after the window's first GPU use)"
    return "full; the 24 h window starts at the next GPU call"


def quota_summary(provider: HfSpaceProvider | None = None) -> dict[str, Any]:
    """ZeroGPU seconds left on every account (from the Hub when it answers), when they come
    back, and per call kind the quota it needs and how many still fit in the daily quotas (a
    call runs on one account: each counts what it holds). Past them, calls run on credits when
    ``hf_overquota`` is on; otherwise a call is refused unless its whole reservation is left."""
    p = provider or HfSpaceProvider()
    q = p.quota()
    rows = q.get("accounts") or [q]
    accts = dict(zip([r.get("account", "") for r in rows], p.accounts, strict=True))
    left = float(q["left"])
    fits, unset = {}, []
    for name, (use, seconds, *model) in QUOTA_CALLS.items():
        need = call_reserve_s(use, seconds, p.s, *model)
        role = _call_role(use, model[0] if model else None, p.s)
        able = [r for r in rows if accts[r.get("account", "")].serves([*SPACES[role].needs, role], p.s)]
        fits[name] = {"needs_s": round(need), "fits": sum(int(r["left"] // need) for r in able)}
        if not able:
            unset.append(name)
    if len(rows) > 1:
        reset = "; ".join(f"{r['account']}: {_back(r)}" for r in rows)
        each = [{"account": r["account"], "left_s": round(r["left"]), "reset_at": r["reset_at"],
                 "source": r["source"],
                 "missing": [x for x in SPACES if p.s.hf_spaces(x)
                             and not accts[r["account"]].space(x, p.s)]} for r in rows]
    else:
        reset, each = _back(q), None
    return {"source": q["source"], "left_s": round(left), "limit_s": q["limit"],
            "used_s": q["used"], "left_min": round(left / 60, 1),
            # one call runs on one account: the largest one the daily quota still takes (with
            # several accounts left_s is their sum, never one call's room; batches: gen_check)
            "max_call_s": round(max(r["left"] for r in rows)), "reset": reset,
            "reset_at": q.get("reset_at"), "calls": fits,
            # several accounts: each one's quota, and the roles it has no duplicate for yet
            **({"accounts": each} if each else {}),
            # calls whose Space is not duplicated yet (OPENCINE_HF_SPACE_<ROLE> unset): they would fail
            **({"not_set": unset} if unset else {}),
            "overquota": ("credits: past the daily quota the Hub bills $1 / 10 GPU min and "
                          "refuses once the credits run out") if p.s.hf_overquota else "off",
            **({"overquota_used_s": q["overquota_used"]} if q.get("overquota_used") else {}),
            **({"hub_error": q["hub_error"]} if q.get("hub_error") else {})}


def apply_patches(text: str, patches: list[tuple[str, str]], where: str = "") -> str:
    """Apply ``(old, new)`` edits to ``text``. Each ``old`` must occur exactly once; an edit
    whose ``new`` is already there counts as applied, so re-running a duplicate is safe.
    Anything else means upstream moved: fail loudly, never half-patch."""
    for old, new in patches:
        if text.count(new) == 1:
            continue
        n_old = text.count(old)
        if n_old != 1:
            raise ProviderError(f"{where}: patch anchor found {n_old}× (expected once): "
                                f"{old.strip().splitlines()[0][:80]!r}; upstream changed, "
                                "update SPACES patches")
        text = text.replace(old, new)
    return text


def space_tree(api: Any, space_id: str, revision: str | None = None) -> dict[str, str]:
    """``{path: git blob id}`` of a Space's files: two Spaces with the same tree run the same code."""
    return {f.path: f.blob_id for f in api.list_repo_tree(
        space_id, repo_type="space", revision=revision, recursive=True) if getattr(f, "blob_id", None)}


def _at_revision(api: Any, space_id: str, upstream: str, revision: str) -> list[str]:
    """Put a fresh duplicate of ``upstream`` back to ``revision`` when upstream has moved since:
    files changed or added after it restored or deleted, in one commit → the paths touched."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    head = api.space_info(upstream).sha
    if head == revision:
        return []
    want, have = space_tree(api, upstream, revision), space_tree(api, upstream, head)
    ops: list[Any] = [CommitOperationDelete(path_in_repo=p) for p in sorted(have.keys() - want.keys())]
    ops += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=api.hf_hub_download(
        repo_id=upstream, filename=p, repo_type="space", revision=revision))
        for p, blob in sorted(want.items()) if have.get(p) != blob]
    if ops:
        api.create_commit(space_id, ops, repo_type="space",
                          commit_message=f"opencine: {upstream} at {revision[:8]} (patches written for it)")
    return [op.path_in_repo for op in ops]


def duplicate_space(role: str, to_id: str | None = None, *, token: str | None = None,
                    settings: GenSettings | None = None, source: str | None = None) -> dict[str, Any]:
    """Duplicate ``SPACES[role].upstream`` as a private ZeroGPU Space and record its sha.

    The duplicate is put back to ``SpaceSpec.revision`` when upstream has moved since, so
    every account gets the files the patches were written for. ``source``: duplicate that
    Space instead (one already set up, e.g. a public template; patches it has count as
    applied), as it is. Secrets the Space needs are set to ``token``; variables come from the
    settings (the video denoiser is pointed at ``OPENCINE_HF_SPACE_COND``, so duplicate ``cond``
    first); ``files`` replace the upstream's (pinned requirements) and ``patches`` edit the
    duplicate's own files (lossless outputs). Rerunning on an existing duplicate applies what
    is missing. The sha recorded is the one after them.
    """
    s = settings or get_gen_settings()
    if role in SPACE_HOSTS:
        raise ProviderError(f"{role} runs on the {SPACE_HOSTS[role]} Space: duplicate that one "
                            f"(opencine gen hf-duplicate {SPACE_HOSTS[role]})")
    spec = SPACES[role]
    token = token if token is not None else (get_settings().hf_account_tokens() or [None])[0]
    variables = spec.variables(s)
    if "cond" in spec.needs and "H3_CONDITIONER" not in variables:
        raise ProviderError("duplicate the conditioner first (opencine gen hf-duplicate cond --to "
                            "<you>/h3-conditioner) and set OPENCINE_HF_SPACE_COND")
    api = _make_api(token)
    pinned: list[str] = []
    if to_id and api.repo_exists(to_id, repo_type="space"):
        # an existing duplicate: refresh secrets/variables (duplicate_repo(exist_ok=True)
        # raises KeyError on the 409 in huggingface_hub 2.1)
        space_id = to_id
        for k in spec.secrets:
            api.add_space_secret(space_id, k, token)
        for k, v in variables.items():
            api.add_space_variable(space_id, k, v)
    else:
        url = api.duplicate_repo(
            source or spec.upstream, to_id, repo_type="space", private=True,
            space_hardware="zero-a10g",
            space_secrets=[{"key": k, "value": token} for k in spec.secrets] or None,
            space_variables=[{"key": k, "value": v} for k, v in variables.items()] or None)
        space_id = (to_id or getattr(url, "repo_id", None)
                    or str(url).rstrip("/").split("/spaces/")[-1])
        if source is None and spec.revision:
            pinned = _at_revision(api, space_id, spec.upstream, spec.revision)
    patched = {}
    for path, edits in (spec.patches or {}).items():  # all checked before anything is pushed
        local = api.hf_hub_download(repo_id=space_id, filename=path, repo_type="space")
        text = Path(local).read_text("utf-8")
        new = apply_patches(text, edits, f"{space_id}/{path}")
        if new != text:
            patched[path] = new
    for path, content in {**(spec.files or {}), **patched}.items():
        api.upload_file(path_or_fileobj=content.encode("utf-8"), path_in_repo=path,
                        repo_id=space_id, repo_type="space", commit_message=f"opencine: pin {path}")
    sha = api.space_info(space_id).sha
    log = s.gen_state_dir / "hf_spaces.json"
    data = json.loads(log.read_text("utf-8")) if log.exists() else {}
    data[space_id] = {"role": role, "space": space_id, "sha": sha, "upstream": spec.upstream,
                      "source": source or f"{spec.upstream}@{spec.revision or 'HEAD'}",
                      "ts": datetime.now().isoformat(timespec="seconds")}
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(data, indent=2), "utf-8")
    return {**data[space_id], **({"reverted": pinned} if pinned else {}),
            "env": f"OPENCINE_HF_SPACE_{role.upper()}={space_id}@{sha}"}
