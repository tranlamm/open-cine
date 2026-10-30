"""MiniMax-H3 face pass: redraw a small, waxy face of a finished clip at high resolution.

Method after ``Carasibana/ComfyUI-H3-FaceRefine`` (MIT), run on the ``video_ref`` Space (Ref2VA +
ref2v Turbo LoRA; role ``face``, endpoint ``/mc_facefix``). The engine sends a square crop that
tracks the face (``opencine.gen.face``) with a per-frame denoise strength (strong where the face
is small). The Space noises the crop to the strongest level (SDEdit, no guide), holds weaker
frames at their own level until the schedule reaches them, takes identity from ``<Picture 1>``,
and locks the clip's own soundtrack as condition audio rows so the mouth follows the sound. The
crop comes back silent; the engine pastes it back.

``API`` is appended to ``h3ref.API`` in the Space's ``mc_api.py``; ``PATCHES`` add its hidden
endpoint to ``app.py``; their code is in ``opencine/gen/spaces/h3_face/``. No imports from
``hf_space``.
"""

from __future__ import annotations

from opencine.gen import spaces
from opencine.gen.providers import h3ref

FPS = 24
API_NAME = "/mc_facefix"
SHIFT = 12.0  # H3's video schedule: denoise d → sigma 12d / (1 + 11d)
SIZES = (512, 640, 768)  # square crop canvases (768² = 2.25× the tokens of 512²)
MAX_FRAMES = 243
MAX_PICTURES = h3ref.MAX_PICTURES
AUDIO_RATE, AUDIO_HOP = 32000, 800  # audio VAE: 32 kHz per channel, 40 latents/s
ENCODE_S = 10  # GPU seconds for the audio + video VAE encodes (estimate)


def frames(n: int) -> int:
    """Frames a call runs for ``n`` clip frames: the next ``17k + 5`` (VAE grid), at least 22."""
    n = max(22, n)
    return n + (5 - n % 17) % 17


def sigma(d: float) -> float:
    """Start sigma of denoise ``d`` on H3's shifted schedule."""
    return SHIFT * d / (1 + (SHIFT - 1) * d)


def schedule(strengths: list[float], steps: int) -> tuple[list[float], list[float]]:
    """``(sigma tail without the final 0, start sigma per frame)``. The tail is ``steps`` points
    from the strongest frame's level down (like ComfyUI's ``BasicScheduler`` with that denoise),
    so step count and strength stay independent."""
    top = max(strengths)
    if not 0 < top < 1:
        raise ValueError(f"face strengths: the strongest must be in (0, 1), got {top}")
    return [sigma(top * (1 - k / steps)) for k in range(steps)], [sigma(d) for d in strengths]


def audio_samples(n: int) -> int:
    """Soundtrack samples a call of ``n`` frames encodes (audio latents × hop)."""
    return round(n / FPS * 40) * AUDIO_HOP


def need_s(size: int, n: int, steps: int, pictures: int) -> float:
    """GPU seconds one call needs, uncapped: the Ref2VA estimate without a guide plus encodes."""
    return h3ref.need_s(size, size, n, steps, pictures, guide=0) + ENCODE_S


def gpu_s(size: int, n: int, steps: int, pictures: int) -> int:
    """``@spaces.GPU`` duration the Space declares (mirrors the function in ``API``)."""
    return int(min(h3ref.GPU_MAX, need_s(size, n, steps, pictures)))


PROMPT = """subject_definitions:
<Subject 1> is the person in <Picture 1>: {details}.

summary:
[reference generation] A close-up of the head and face of <Subject 1>, shoulders at the bottom of \
the frame, the frame following the head as it moves.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, eyes, nose, mouth, skin tone, hair \
and head shape are retained.

detailed_description:
Live-action, photorealistic close-up with natural unretouched skin texture and visible fine pores, \
matte skin with soft highlight rolloff, clear eyes with fine lashes, crisp lips and teeth, \
individual hair strands, the light and colours of the video.
[Shot 1] 0.00 s: <Subject 1> fills the middle of the frame and keeps the exact head position, \
head motion, gaze, expressions, blinks and lip movement of the video, the frame following the head.

overall_soundscape:
The voice and sounds of the video.

non_diegetic_music:
None."""


def prompt(details: str = "") -> str:
    """The face pass's Ref2VA prompt; ``details`` describes the person in a few words, e.g.
    "a man in his thirties with short curly black hair and a trimmed beard"."""
    text = " ".join(details.split()).rstrip(".").strip() or "the person shown in the picture"
    return PROMPT.format(details=text[0].lower() + text[1:] if text[:2].istitle() else text)


API = "\n\n" + spaces.code("h3_face/api.py")

# Hidden endpoint, inserted after the UI's own run.click and outside the Ref2VA patch, so an
# already-patched duplicate gets only this addition.
_RUN_END = '        api_name="swap",\n    )\n'
_UI = "\n" + spaces.code("h3_face/ui.py", indent=4)
PATCHES = [(_RUN_END, _RUN_END + _UI)]
