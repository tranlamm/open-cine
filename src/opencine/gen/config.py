"""Generation settings (env prefix ``OPENCINE_``), kept apart from ``core.config``.

HF Spaces are the user's private duplicates, ``<owner>/<name>@<sha>`` (``@sha`` optional;
``opencine gen warm`` warns when the Space moved off it), one per HF account, comma-separated or a
JSON list. The first entry names the Space in cache keys, so append new accounts.
``opencine gen hf-setup`` makes the duplicates on every account and writes the lines.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic_settings import BaseSettings, SettingsConfigDict

from opencine.core.config import ENV_FILES, HOME, split_list

# roles whose endpoint lives on another role's duplicate: the face pass runs on the Ref2VA Space
SPACE_HOSTS = {"face": "video_ref"}


class SpaceRef(NamedTuple):
    id: str
    sha: str | None

    @property
    def owner(self) -> str:
        return self.id.partition("/")[0]

    def __str__(self) -> str:
        return f"{self.id}@{self.sha}" if self.sha else self.id


def parse_space(value: str) -> SpaceRef:
    sid, _, sha = value.strip().partition("@")
    return SpaceRef(sid, sha or None)


def parse_spaces(value: str | None) -> list[SpaceRef]:
    """``OPENCINE_HF_SPACE_<ROLE>``: one Space, or one per HF account."""
    return [parse_space(v) for v in split_list(value or "")]


class GenSettings(BaseSettings):
    # same .env as ``core.config`` whatever the cwd (MCP server, detached jobs)
    model_config = SettingsConfigDict(env_prefix="OPENCINE_", env_file=ENV_FILES, extra="ignore",
                                      env_ignore_empty=True)

    # account-level state (HF quota book, Space duplicate log)
    gen_state_dir: Path = HOME / "gen"

    # hf: OPENCINE_HF_SPACE_<ROLE> = "<owner>/<name>@<sha>[,<owner2>/<name>@<sha>…]" (one per account),
    # one setting per role of hf_space.SPACES
    hf_space_image: str | None = None  # Qwen-Image-2.1 (edit, new camera angles included)
    hf_space_video_turbo: str | None = None  # MiniMax-H3 + Turbo LoRA denoiser (the default)
    hf_space_video: str | None = None  # MiniMax-H3 denoiser, 28 steps (manual runs: --model h3)
    hf_space_cond: str | None = None  # MiniMax-H3 conditioner, called by the video Spaces
    hf_space_rife: str | None = None
    hf_space_music: str | None = None  # ACE-Step 1.5: instrumental background music
    # video-to-video: character swap (SCAIL-2) and restyle (LTX-2.3)
    hf_space_swap: str | None = None  # SCAIL-2, replacement mode
    hf_space_restyle: str | None = None  # LTX-2.3 distilled + Union-Control IC-LoRA
    # MiniMax-H3 Ref2VA + character-swap LoRA v1.1, the window as an aligned guide
    hf_space_swap_h3: str | None = None
    # MiniMax-H3 Ref2VA from character / place pictures: a second duplicate of the swap Space
    hf_space_video_ref: str | None = None
    hf_space_enhance: str | None = None  # FlashVSR+ turbo: sharper, larger copies of clips
    # enhance: output short side (H3 7:4 → 1920x1080 | 2520x1440, ~1.6x the GPU time); input frames
    # per latent (4; 2 or 1 keep fast small motion, ×1.19 / ×1.41 GPU); colour/light matched to input
    hf_enhance_short: Literal[1080, 1440] = 1080
    hf_enhance_compression: Literal[1, 2, 4] = 4
    hf_enhance_color: bool = True
    # what a swap runs when the slot names no model
    hf_swap_model: Literal["scail2", "scail2-hq", "h3-swap", "h3-swap-hq"] = "scail2"
    # H3 swap: first pass at low 416x736 | mid 576x1024 | high 768x1344 (9:16), Euler steps over the
    # folded Turbo LoRA; `upscale` = a second pass in the same call (3D latent upscaler to `high`,
    # then the sigma tail); colours pulled to the window's background (Reinhard Lab at this
    # strength, highlights kept; 0 = off); character cut out onto grey first. LoRA and Turbo files
    # are Space variables: run `opencine gen hf-setup --refresh --roles swap_h3` after a change
    hf_h3swap_size: Literal["low", "mid", "high"] = "mid"
    hf_h3swap_steps: int = 8
    hf_h3swap_upscale: bool = False
    hf_h3swap_up_sigmas: str = "0.9035,0.6316,0.3158"
    hf_h3swap_colour: float = 0.9
    hf_h3swap_cutout: bool = True
    hf_h3swap_lora: str = "loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors"
    # lightx2v's ref2v Turbo: 8-step v1.0 768p (1344×768, shift 12/3, 8 Euler steps, as LightX2V's
    # ref2av config) rather than 4-step v0.1 (544p, oily skin); v1.1 has a report of garbled speech
    hf_h3swap_turbo: str = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    # second pass only: the LMS sharpness LoRA at this strength (0 = off), its caption on that
    # pass's prompt (one more conditioner call, ~90 s of quota); the LMS file and the latent
    # upscaler (<repo>:<file>; the LoRA author's own: Alissonerdx/Minimax-H3-ComfyUI:
    # latent_upscaler/h3_upscaler_lms_v0.1.safetensors) are Space variables (--refresh after)
    hf_h3swap_lms: float = 0.7
    hf_h3swap_lms_caption: bool = True
    hf_h3swap_lms_file: str = "Alissonerdx/Minimax-H3-ComfyUI:loras/minimax_h3_lms_v1.0_r64.safetensors"
    hf_h3swap_upscaler: str = ("LBH-123-AI/Minimax_h3_latent_Upscaler:minimax_h3_latent_upscaler_3d_conv_v1/"
                               "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors")
    # SCAIL-2: 9:16 at 512x896 (low) | 704x1280 (high, ~3.4x the GPU time), frames 1:1 at this
    # rate (trained at 16-30 fps). Fast recipe: LightX2V distill LoRA, T2V for replacement (I2V +
    # UniPC at shift 1 leaves white specks), 8 first-order steps at shift 5, CFG 1 ("<repo>:<file>"
    # of the LoRA, "" = none: then CFG 5, 40 steps, unipc). The solver is a Space variable:
    # `opencine gen hf-setup --refresh --roles swap` after a change
    hf_scail_size: Literal["low", "high"] = "low"
    hf_scail_fps: int = 16
    hf_scail_steps: int = 8
    hf_scail_cfg: float = 1.0
    hf_scail_shift: float = 5.0
    hf_scail_solver: Literal["euler", "lcm", "unipc", "dpm++"] = "euler"
    hf_scail_lora: str = ("Kijai/WanVideo_comfy:"
                          "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors")
    hf_scail_replace: bool = True  # False: animation mode with the background kept by the masks
    hf_scail_grow: float = 0.03  # person masks widened by this share of the height (bulky armor)
    # LTX restyle: 9:16 at 640x1152 (low) | 768x1408 (high); how hard the control holds (0-1)
    hf_restyle_size: Literal["low", "high"] = "low"
    hf_restyle_strength: float = 0.9
    # Qwen-Image-2.1 out (9:16): 768x1344 | 1536x2688; set by the model choice (hf_space.MODELS)
    hf_image_quality: Literal["1k", "2k"] = "2k"
    # 9:16: 544x960 | 640x1152 | 768x1344; full = H3's native size (short edge 768)
    hf_video_quality: Literal["fast", "mid", "full"] = "full"
    hf_video_steps: int = 28  # h3, the Space default (10–40)
    # h3-turbo: the H3 scheduler runs N-1 model passes for N steps; the LoRA card: 6–8 passes
    hf_video_turbo_steps: int = 7
    # h3-ref (Ref2VA): Euler steps over the folded ref2v Turbo LoRA (a Space variable: run
    # `opencine gen hf-setup --refresh --roles video_ref` after changing the file)
    hf_video_ref_steps: int = 8
    hf_video_ref_turbo: str = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"  # as hf_h3swap_turbo
    # face pass (`gen_face`): the square crop canvas, Euler steps over the ref2v Turbo, the
    # base denoise (×1 for a face ≤ 30 px tall, falling to ×0.35 from 120 px; heights counted at
    # H3's 768 px short side), the crop side as a multiple of the face height, and the median face
    # height above which a clip is left as it is (unless forced)
    hf_face_size: int = 768  # 512 | 640 | 768 (a multiple of 32)
    hf_face_steps: int = 8
    hf_face_denoise: float = 0.40
    hf_face_crop: float = 2.5
    hf_face_max_px: float = 160.0
    hf_video_min_s: float = 4.0  # model card: 4–15 s; the Space accepts down to 2 s
    # rewrite a plain prompt into H3's trained format (Qwen3-VL); prompts already written in
    # that format (Context-IR) are never rewritten
    hf_video_upsample: bool = True
    hf_quota_s: float = 2400.0  # PRO: 40 min/day
    # past the daily quota the Hub bills the account's credits ($1 / 10 min) and refuses once
    # they run out; off = the local book refuses a call whose reservation is not left
    hf_overquota: bool = True
    hf_quota_api: bool = True  # read the quota left from the Hub (the token needs billing read)
    hf_retries: int = 3
    # one call's wait beyond its GPU reservation (upload, queue, download); past it the call
    # counts as hung: cancelled and retried (a Space once never answered for 35 min)
    hf_call_timeout_s: float = 600.0
    hf_wake_timeout_s: float = 900.0

    # fake provider: raise QuotaExhausted after this many calls (None = never)
    fake_quota_calls: int | None = None

    def hf_spaces(self, role: str) -> list[SpaceRef]:
        """Every account's duplicate for ``role``, as configured (a hosted role's: its host's)."""
        return parse_spaces(getattr(self, f"hf_space_{SPACE_HOSTS.get(role, role)}", None))

    def hf_space(self, role: str) -> SpaceRef | None:
        """The reference duplicate for ``role`` (the first entry: cache keys name it)."""
        refs = self.hf_spaces(role)
        return refs[0] if refs else None


@lru_cache
def get_gen_settings() -> GenSettings:
    return GenSettings()
