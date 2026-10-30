"""One-call Ref2VA API for Open Cine (``opencine gen hf-duplicate video_ref`` writes this file; a
rerun replaces it): reference pictures (+ an optional first frame at the output size), a six-field
Ref2VA prompt -> mp4 with the generated soundtrack (CRF 10, BT.709) + a report line. app.py hands over
its pipeline and helpers (``bind``). Raises on failure."""
import contextlib
import os
import tempfile
import time

import gradio as gr
import numpy as np
import spaces
import torch

import h3_video

APP = {}  # pipe, error, encode_condition, client, ip_token, read_conditioning, conditioner, notes
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")
AUDIO_REPO = os.environ.get("OPENCINE_AUDIO_VAE", "MiniMaxAI/MiniMax-H3")
MAX_PICTURES = 8
_AUDIO = {}
_GUIDE = {"frames": None}

B, C, SAFETY, PIXEL_FRAMES = 1.1745e-4, 3.8396e-9, 1.15, 960 * 544 * 124
TEXT_ROWS, VISION_ROWS, GPU_MAX = 2000, 5750, 900


def ref_need_s(width, height, n, steps, pictures, guide=1):
    lat = (n - 5) // 17 * 5 + 2
    per = (height // 32) * (width // 32)
    audio = round(n / 24 * 40) * 2
    rows = TEXT_ROWS + pictures * (VISION_ROWS + per) + (guide + lat) * per + audio
    denoise = steps * (B * rows + C * rows * rows) * SAFETY
    return 45 + 5 * pictures + denoise + 20 + 18 * n * width * height / PIXEL_FRAMES + 10


def ref_gpu_s(width, height, n, steps, pictures, guide=1):
    return int(min(GPU_MAX, ref_need_s(width, height, n, steps, pictures, guide)))


def gpu_s(references, guide, embeds, tags, height, width, frames, steps, seed, *args, **kwargs):
    return ref_gpu_s(int(width), int(height), int(frames), int(steps), len(references),
                     0 if guide is None else 1)


def _load_audio():
    """The audio VAE (float32: in bfloat16 it decodes ~20 dB too quiet), on the CPU until a call."""
    if "vae" not in _AUDIO:
        import diffusers

        cls = getattr(diffusers, "AutoencoderKLMiniMaxH3Audio", None)
        if cls is None:
            from diffusers.models import AutoencoderKLMiniMaxH3Audio as cls
        vae = cls.from_pretrained(AUDIO_REPO, subfolder="audio_vae", torch_dtype=torch.float32).eval()
        if hasattr(vae, "set_attention_backend"):
            vae.set_attention_backend("native")  # no cuDNN kernel for the float32 audio VAE
        _AUDIO["vae"] = vae
    return _AUDIO["vae"]


def bind(**kwargs):
    APP.update(kwargs)
    try:
        _load_audio()
    except Exception as error:  # the clip is still made, silent
        APP.setdefault("notes", []).append(f"audio VAE not loaded: {type(error).__name__}: {error}")


# ---- layout: the charswap step with a guide of 0 or 1 latent frames -------------------------------
def _sequence(text_token_tags, reference_shapes, num_latent_frames, latent_height, latent_width,
              num_audio_latents, patch_size, guide_frames=0):
    """``[text | references | guide | target audio | target video]``: references on H3's native
    image rope (one rotary slot each), then a guide of ``guide_frames`` latent frames overlapping the
    target's first ones (0 = the stock Ref2VA layout)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _fill_audio_positions,
        _frame_position_grid,
        _temporal_position_grid,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_TAG,
        MINIMAX_H3_VIDEO_TAG,
    )

    _, ph, pw = patch_size
    n_text = text_token_tags.shape[0]
    per_frame = (latent_height // ph) * (latent_width // pw)
    ref_rows = [f * (h // ph) * (w // pw) for f, h, w in reference_shapes]
    n_guide = int(guide_frames) * per_frame
    n_audio = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    length = n_text + int(sum(ref_rows)) + n_guide + n_audio + num_latent_frames * per_frame
    pos = torch.zeros(length, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    blocks, cursor, clock = [], n_text, float(n_text)
    for (frames, h, w), count in zip(reference_shapes, ref_rows):
        rows = slice(cursor, cursor + count)
        cursor = rows.stop
        grid, _ = _frame_position_grid(h, w, ph, pw)
        pos[rows, 0] = clock
        pos[rows, 1:] = grid.repeat(frames, 1)
        blocks.append(torch.arange(rows.start, rows.stop))
        clock += 1.0
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, ph, pw)
    media_t = _temporal_position_grid(num_latent_frames, clock).repeat_interleave(frame_grid.shape[0])
    media_hw = frame_grid.repeat(num_latent_frames, 1)
    if n_guide:
        rows = slice(cursor, cursor + n_guide)
        cursor = rows.stop
        pos[rows, 0] = media_t[:n_guide]
        pos[rows, 1:] = media_hw[:n_guide]
        blocks.append(torch.arange(rows.start, rows.stop))
    audio_start, video_start = cursor, cursor + n_audio
    _fill_audio_positions(pos, slice(audio_start, video_start), num_audio_latents, clock, width_grid)
    pos[video_start:, 0] = media_t
    pos[video_start:, 1:] = media_hw
    video_idx = torch.cat(blocks + [torch.arange(video_start, length)])
    audio_idx = torch.arange(audio_start, video_start)
    text_idx = torch.arange(n_text)
    tags = torch.empty(length, dtype=torch.long)
    tags[text_idx] = text_token_tags.to(torch.long)
    tags[audio_idx] = MINIMAX_H3_AUDIO_TAG
    tags[video_idx] = MINIMAX_H3_VIDEO_TAG
    return pos, tags, video_idx, audio_idx, text_idx, int(sum(ref_rows)) + n_guide, 0


@contextlib.contextmanager
def _guide_frames(n):
    """The charswap layout step builds ``_sequence`` with an ``n``-frame guide for this call."""
    from h3_charswap_blocks import MiniMaxH3CharswapPrepareLayoutStep as step

    original = step.build_charswap_sequence
    step.build_charswap_sequence = lambda self, *a: _sequence(*a, guide_frames=n)
    try:
        yield
    finally:
        step.build_charswap_sequence = original


@contextlib.contextmanager
def _latents_only():
    """Run the pipeline without its video decode: ``latents`` is left unpacked in the state."""
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

    call = MiniMaxH3VideoDecodeStep.__call__
    MiniMaxH3VideoDecodeStep.__call__ = lambda self, components, state: (components, state)
    try:
        yield
    finally:
        MiniMaxH3VideoDecodeStep.__call__ = call


def _encode(pipe, pixels):
    """A picture through the VAE encoder (app.encode_condition), 128 px tiles on failure."""
    encode = APP["encode_condition"]
    args = (pipe.pixel_mean, pipe.pixel_std, pipe.keyframe_encode_seed)
    pixels = pixels.to("cuda")
    try:
        return encode(pipe.vae, pixels, *args)
    except RuntimeError as error:
        error.__traceback__ = None
        torch.cuda.empty_cache()
        pipe.vae.enable_tiling(tile_sample_min_height=128, tile_sample_min_width=128)
        return encode(pipe.vae, pixels, *args)


@torch.no_grad()
def _decode(pipe, latents):
    """MiniMaxH3VideoDecodeStep, then (T, H, W, 3) uint8; tiled on failure."""
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    latents = latents.to("cuda") * std + mean
    pm = torch.tensor(pipe.pixel_mean, device="cuda").view(1, -1, 1, 1, 1)
    ps = torch.tensor(pipe.pixel_std, device="cuda").view(1, -1, 1, 1, 1)

    def run():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            video = vae.decode(latents, return_dict=False)[0]
        return (video.float() * ps + pm).clamp(0, 1)

    try:
        video = run()
    except RuntimeError as error:
        error.__traceback__ = None
        torch.cuda.empty_cache()
        vae.enable_tiling()
        video = run()
    return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()


@torch.no_grad()
def _decode_audio(latents):
    """MiniMaxH3AudioDecodeStep on the (2, C, N) audio latents -> ((2, samples) float, rate) or (None, None)."""
    vae = _AUDIO.get("vae")
    if vae is None:
        return None, None
    vae.to("cuda")
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1)
    audio = vae.decode(latents.to("cuda", torch.float32) * std + mean, return_dict=False)[0]
    vae.to("cpu")
    return audio.float().permute(1, 0, 2)[0].cpu(), int(getattr(vae.config, "sampling_rate", 32000))


@spaces.GPU(duration=gpu_s, size=GPU_SIZE)
def _run(references, guide, embeds, tags, height, width, frames, steps, seed):
    pipe, times = APP["pipe"], {}
    pipe.vae.to("cuda")
    t0 = time.time()
    latents = [_encode(pipe, r) for r in references] + ([_encode(pipe, guide)] if guide is not None else [])
    torch.cuda.empty_cache()
    times["encode"] = time.time() - t0
    t0 = time.time()
    with _latents_only(), _guide_frames(0 if guide is None else 1):
        out = pipe(prompt_embeds=embeds.to("cuda"), text_token_tags=tags, condition_latents=latents,
                   num_reference_images=len(references), height=int(height), width=int(width),
                   num_frames=int(frames), num_inference_steps=int(steps) + 1,
                   generator=torch.Generator("cpu").manual_seed(int(seed)),
                   output=["latents", "audio_latents"])
    times["denoise"] = time.time() - t0
    t0 = time.time()
    video = _decode(pipe, out["latents"])
    audio, rate = _decode_audio(out["audio_latents"])
    times["decode"] = time.time() - t0
    return video, audio, rate, times


def _write(frames, path, audio=None, rate=None):
    """(T, H, W, 3) frames as H.264 at 24 fps, CRF 10, BT.709 YUV and tags, + the AAC soundtrack."""
    import av

    frames = np.asarray(frames)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = {"crf": "10", "preset": "medium",
                          "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"}
        sound = None
        if audio is not None:
            from diffusers.utils import export_utils

            sound = export_utils._prepare_audio_stream(container, rate)
        for array in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", interpolation="BICUBIC")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None:
            export_utils._write_audio(container, sound, audio, rate, av)
    return path


def _path(x):
    """A plain str path (gr.File gives gradio's NamedString, which PyAV refuses)."""
    return str(x) if isinstance(x, str) else str(getattr(x, "name", None) or x.get("path"))


def _frame(path, height, width):
    """The first frame at exactly the output size, (1, 3, 1, H, W) uint8 (the engine sends it fitted)."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if image.size != (width, height):
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None, :, None].contiguous()


def _embed(prompt, pictures, canvas, frames):
    """prompt_embeds + text_token_tags with one <Picture N> vision block per picture, in order."""
    from gradio_client import handle_file

    def call():
        client = APP["client"](APP["conditioner"], APP["ip_token"]())
        return client.predict(prompt=prompt, media=[handle_file(p) for p in pictures],
                              kinds=",".join(["image"] * len(pictures)), canvas=canvas,
                              num_frames=int(frames), rewrite_prompt=False, api_name="/encode_ref2va")

    try:
        path, _plan = call()
    except Exception as first:
        print(f"[mc] conditioner: retrying after {first}", flush=True)
        try:
            path, _plan = call()
        except Exception as second:
            raise gr.Error(f"The conditioner could not encode the prompt and pictures "
                           f"({type(second).__name__}: {second}).") from second
    return APP["read_conditioning"](path)


def mc_ref2va(images, first, prompt, frames, width, height, steps=8, seed=42, canvas="",
              progress=gr.Progress()):
    if APP.get("error"):
        raise gr.Error(str(APP["error"]).replace("**", ""))
    pictures = [_path(x) for x in (images or []) if x]
    first = _path(first) if first else None
    order = ([first] if first else []) + pictures
    if not pictures or len(order) > MAX_PICTURES:
        raise gr.Error(f"give 1-{MAX_PICTURES} pictures (refs, + the first frame)")
    if not str(prompt or "").strip().startswith("subject_definitions:"):
        raise gr.Error("the prompt must be the six-field Ref2VA text (subject_definitions: ...)")
    frames, width, height = int(frames), int(width), int(height)
    if frames < 5 or frames % 17 != 5:
        raise gr.Error(f"frames must be 17n+5, got {frames}")
    if width % 32 or height % 32:
        raise gr.Error("sizes must be multiples of 32")
    if ref_need_s(width, height, frames, int(steps), len(order), 1 if first else 0) > GPU_MAX:
        raise gr.Error(f"{frames} frames at {width}x{height}, {int(steps)} steps need more than {GPU_MAX} s")
    t0 = time.time()
    embeds, tags = _embed(str(prompt).strip(), order, canvas, frames)
    area = width * height / 1e6
    references = [h3_video.load_image(p, area) for p in order]
    guide = _frame(first, height, width) if first else None
    prep = time.time() - t0
    video, audio, rate, times = _run(references, guide, embeds, tags, height, width, frames, int(steps),
                                     int(seed))
    path = _write(video, os.path.join(tempfile.mkdtemp(prefix="mc_"), "ref2va.mp4"), audio, rate)
    report = (f"{width}x{height} · {frames} frames · {len(order)} pictures"
              + (" + first-frame guide" if first else "") + f" · prep {prep:.1f} s · "
              + " · ".join(f"{k} {v:.1f} s" for k, v in times.items())
              + f" · steps {int(steps)} · seed {int(seed)}" + ("" if audio is not None else " · silent")
              + " · " + "; ".join(APP.get("notes") or []))
    return path, report
