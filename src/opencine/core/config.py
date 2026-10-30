"""Engine settings: env vars with prefix ``OPENCINE_``, then ``~/.opencine/.env``, then a checkout's ``.env``."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Provider = Literal["hf", "fake"]  # fake: offline dry runs

# src/opencine/core/config.py → the checkout (fx/ and assets/ live there)
REPO_ROOT = Path(__file__).resolve().parents[3]
# user data: settings, projects, models, tools, taste (survives engine updates)
HOME = Path(os.environ.get("OPENCINE_HOME") or Path.home() / ".opencine").expanduser()
ENV_FILE = HOME / ".env"
ENV_FILES = (ENV_FILE, REPO_ROOT / ".env")  # later files win


def split_list(value: Any) -> list[str]:
    """A list setting: a JSON list or comma-separated values, blanks dropped."""
    if isinstance(value, str):
        text = value.strip()
        value = json.loads(text) if text.startswith("[") else text.split(",")
    return [str(v).strip() for v in value or [] if str(v).strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENCINE_", env_file=ENV_FILES, extra="ignore",
                                      env_ignore_empty=True)

    projects_dir: Path = HOME / "projects"
    default_rate: str = "30"

    # generation
    gen_image: Provider = "hf"
    gen_video: Provider = "hf"
    gen_edit: Provider = "hf"  # video-to-video: character swap, restyle
    gen_max_rerolls: int | None = None  # extra batches per slot; None = unlimited

    # budgets
    cost_cap_usd: float = 5.0
    image_budget_count: int = 25
    image_budget_tokens: int = 40_000
    image_review_count: int = 8  # self-review sheets (~1.6k tokens each)
    image_review_tokens: int = 12_000

    # external tools
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    node: str = "node"  # Node ≥ 22 for the motion / scribble layers (fx/)
    sfx_dir: Path | None = None  # shared SFX library; a project's sfx/ wins
    face_detect: bool = True  # subject tracking at ingest (YuNet)
    models_dir: Path = HOME / "models"
    # critic: another model family compares two versions before a gate
    critic: Literal["codex", "none"] = "none"
    codex_bin: str = "codex"
    critic_model: str = ""  # empty = Codex default
    critic_effort: str = ""  # low | medium | high | xhigh; empty = Codex default
    critic_timeout_s: int = 180
    # taste memory across projects
    taste: bool = True
    taste_dir: Path = HOME / "taste"
    # local ncnn upscale / frame interpolation for finals; unset = CPU fallbacks
    tools_dir: Path = HOME / "tools"
    esrgan_bin: str | None = None
    esrgan_model: str = "realesrgan-x4plus"
    rife_bin: str | None = None
    rife_model: str = "rife-v4.6"
    upscale_max_s: float = 600.0  # per final render; beyond it: lanczos + CAS
    mask_model: bool = True  # SAM 2.1 mask wipes (extra `mask`)
    vision_models: bool = True  # Florence-2 + EdgeTAM for scribble (extra `vision`)
    cv_device: Literal["auto", "cpu"] = "auto"  # auto = GPU when onnxruntime has one

    # speech gateways (STT / TTS); unset URL = off
    stt_url: str | None = None
    stt_token: str | None = None
    tts_url: str | None = None
    tts_token: str | None = None
    speech_language: str = "en"
    stt_on_ingest: bool = True

    # Telegram (optional): telegram_send posts stills, clips and finals to this chat
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_api_url: str = "https://api.telegram.org"  # a local Bot API server lifts the 50 MB cap

    # credentials
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENCINE_HF_TOKEN", "HF_TOKEN")
    )
    # several HF PRO accounts, used in order; a call moves on when one's daily ZeroGPU quota runs out
    hf_tokens: Annotated[list[str], NoDecode] = Field(
        default=[], validation_alias=AliasChoices("OPENCINE_HF_TOKENS", "HF_TOKENS")
    )

    @field_validator("hf_tokens", mode="before")
    @classmethod
    def _token_list(cls, value: Any) -> list[str]:
        return split_list(value)

    def hf_account_tokens(self) -> list[str]:
        """The HF tokens generation rotates through: ``OPENCINE_HF_TOKENS``, else ``HF_TOKEN``."""
        return list(dict.fromkeys(self.hf_tokens)) or ([self.hf_token] if self.hf_token else [])


@lru_cache
def get_settings() -> Settings:
    return Settings()
