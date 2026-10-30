"""One-call API for Open Cine (``opencine gen hf-duplicate swap_h3`` writes this file; a rerun
replaces it): guide clip (24 fps, 17n+5 frames, at the output size), character picture, prompt ->
silent mp4 (CRF 10, BT.709) + a report line. ``up_*``: a second pass in the same GPU call (latent
upscaler 3D, then a short sigma tail with the guide re-encoded at the larger size; ``up_lms``: the
LMS LoRA at that strength on this pass only, ``up_caption``: its caption on this pass's prompt).
app.py hands over its pipeline and helpers (``bind``). Raises on failure."""
import contextlib
import os
import tempfile
import time

import gradio as gr
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

import h3_video

APP = {}  # pipe, error, encode_prompt, encode_condition, reference_mp, notes
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")
UPSCALER = os.environ.get(
    "OPENCINE_UPSCALER", "LBH-123-AI/Minimax_h3_latent_Upscaler:minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors")
LMS_FILE = os.environ.get("H3_LMS_FILE", "none")  # "<repo>:<file>" of the LMS LoRA, "none" = off
LMS_CAPTION = ("Enhance this video with sharp, crisp details while preserving a natural "
               "photorealistic appearance.")
LMS_CHUNK = 2 ** 27  # output elements one LoRA add touches at a time (bf16: 256 MB)
_UP = {}
_LMS = {}


def bind(**kwargs):
    APP.update(kwargs)
    _preload()


B, C, SAFETY, PIXEL_FRAMES = 1.1745e-4, 3.8396e-9, 1.15, 960 * 544 * 124
TEXT_ROWS, REF_ROWS, GPU_MAX = 7750, 300, 900
LMS_COST, LMS_LOAD = 1.1, 5


def h3swap_gpu_s(width, height, n, steps, up_width=0, up_height=0, up_steps=0, lms=False):
    lat = (n - 5) // 17 * 5 + 2
    audio = round(n / 24 * 40) * 2

    def denoise(w, h, k):
        rows = TEXT_ROWS + REF_ROWS + 2 * lat * (h // 32) * (w // 32) + audio
        return k * (B * rows + C * rows * rows) * SAFETY

    t = 45 + 9 * n * width * height / PIXEL_FRAMES + denoise(width, height, steps)
    w, h = width, height
    if up_steps:
        w, h = up_width, up_height
        t += 15 + 9 * n * w * h / PIXEL_FRAMES + denoise(w, h, up_steps) * (LMS_COST if lms else 1)
        t += LMS_LOAD if lms else 0
    return int(min(GPU_MAX, t + 20 + 18 * n * w * h / PIXEL_FRAMES))


def _tail(text):
    return [float(v) for v in str(text or "").replace(" ", "").split(",") if v]


def gpu_s(reference, guide, guide_up, embeds, tags, height, width, frames, steps, seed, up_height,
          up_width, up_sigmas, embeds_up=None, tags_up=None, up_lms=0.0, *args, **kwargs):
    n_up = len(_tail(up_sigmas)) if int(up_width) else 0
    return h3swap_gpu_s(int(width), int(height), int(frames), int(steps), int(up_width),
                        int(up_height), n_up, bool(n_up and float(up_lms or 0) > 0))


# ---- MiniMax-H3 latent upscaler 3D (LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, MIT) ----------
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]


def _gn(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = _gn(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.0), nn.Conv3d(channels, channels, 3, padding=1))
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = torch.chunk(self.emb_layers(emb).type(h.dtype)[..., None, None, None], 2, dim=1)
        return self.skip(x) + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = _gn(channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, temporal_every=2,
                 temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks, self.out_blocks = nn.ModuleList(), nn.ModuleList()
        for blocks, count in ((self.in_blocks, in_blocks), (self.out_blocks, out_blocks)):
            for b in range(count):
                blocks.append(ResBlockEmb3D(channels, 64))
                if temporal_every > 0 and b % temporal_every == 0:
                    blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _gn(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _segment(self, x, scale, size):
        emb = self.embed(torch.tensor([[scale - 1]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for blocks in (self.in_blocks, self.out_blocks):
            for b in blocks:
                x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
            if blocks is self.in_blocks:
                x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward(self, x, scale, size):
        """The node's temporal chunking: 32-frame chunks, 5-frame replicate-padded overlap, blended."""
        B_, C_, T, H, W = x.shape
        chunk, overlap = 32, 5
        if T <= chunk:
            return self._segment(x, scale, size)
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out = torch.zeros(B_, C_, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)
        for start in range(0, T, chunk):
            end = min(T, start + chunk)
            o0, o1 = max(0, start - overlap), min(T, end + overlap)
            lo, hi = max(0, o0 - overlap), min(T + 2 * overlap, o1 + overlap)
            seg = self._segment(padded[:, :, lo:hi].contiguous(), scale, (hi - lo, size[-2], size[-1]))
            s0 = o0 + overlap - lo
            w = torch.ones(o1 - o0, device=x.device, dtype=x.dtype)
            if start > o0:
                k = start - o0
                w[:k] = torch.arange(1, k + 1, device=x.device, dtype=x.dtype) / (k + 1)
            if o1 > end:
                k = o1 - end
                w[-k:] = torch.arange(k, 0, -1, device=x.device, dtype=x.dtype) / (k + 1)
            out[:, :, o0:o1] += seg[:, :, s0:s0 + (o1 - o0)] * w.view(1, 1, -1, 1, 1)
            weight[:, :, o0:o1] += w.view(1, 1, -1, 1, 1)
        return out / weight.clamp(min=1e-8)


def _load_upscaler():
    if "net" not in _UP:
        import re

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo, _, name = UPSCALER.partition(":")
        sd = load_file(hf_hub_download(repo, name))
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")} or sd
        ins = {int(m.group(1)) for k in sd if (m := re.match(r"in_blocks\.(\d+)\.in_layers\.", k))}
        outs = {int(m.group(1)) for k in sd if (m := re.match(r"out_blocks\.(\d+)\.in_layers\.", k))}
        kernel = next((v.shape[2] for k, v in sd.items() if k.endswith("dwconv.weight")), 0)
        net = LatentResizer3D(sd["conv_in.weight"].shape[1], len(ins), len(outs),
                              sd["conv_in.weight"].shape[0], 2 if kernel else 0, kernel or 5)
        net.load_state_dict(sd, strict=True)
        _UP["net"] = net.eval().requires_grad_(False).to(torch.float16)
        print(f"[mc] upscaler {UPSCALER}: {sum(p.numel() for p in net.parameters()):,} params", flush=True)
    return _UP["net"]


@torch.no_grad()
def _upscale(latents, height, width):
    net = _UP["net"].to("cuda")
    x = latents.to("cuda", torch.float16)
    mean = torch.tensor(LATENTS_MEAN, device="cuda", dtype=torch.float16).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device="cuda", dtype=torch.float16).view(1, -1, 1, 1, 1)
    h, w = height // 16, width // 16
    scale = (w / x.shape[-1] + h / x.shape[-2]) / 2
    out = net((x - mean) / std, scale, (x.shape[2], h, w)) * std + mean
    _UP["net"].to("cpu")
    return out.float()


# ---- the LMS LoRA, unfused, on the second pass only -------------------------------------------
def _load_lms():
    """The LMS LoRA as ``[(module name, A, B·scale)]`` on the transformer's ``nn.Linear`` modules
    (CPU, bf16), its ComfyUI names mapped the way ``h3_lora.fold_comfy_lora`` maps them; every
    target must land on a linear of the right shape, or nothing is used."""
    if "pairs" not in _LMS:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from h3_lora import _targets

        repo, _, name = LMS_FILE.partition(":")
        sd = load_file(hf_hub_download(repo, name))
        transformer = APP["pipe"].transformer_ref
        inner = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        modules = dict(transformer.named_modules())
        pairs, missed = [], []
        for base in sorted({k[: -len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight")}):
            a, b = sd[base + ".lora_A.weight"], sd[base + ".lora_B.weight"]
            alpha = sd.get(base + ".alpha")
            scale = float(alpha) / a.shape[0] if alpha is not None else 1.0
            for key, rows in _targets(base.removeprefix("diffusion_model."), b, inner):
                name_ = key.removesuffix(".weight")
                mod = modules.get(name_)
                if not isinstance(mod, nn.Linear) or tuple(mod.weight.shape) != (rows.shape[0], a.shape[1]):
                    missed.append(key)
                    continue
                pairs.append((name_, a.to(torch.bfloat16), (rows.float() * scale).to(torch.bfloat16)))
        if missed:
            raise RuntimeError(f"LMS LoRA {LMS_FILE}: {len(missed)} targets match no linear, e.g. {missed[:3]}")
        _LMS["pairs"] = pairs
        print(f"[mc] LMS {LMS_FILE}: {len(pairs)} linears", flush=True)
    return _LMS["pairs"]


def _preload():
    """The second pass's latent upscaler and LMS LoRA loaded on the CPU at startup (no GPU): a
    missing or mis-mapped file shows in the logs and every report now, not on a paid call."""
    if APP.get("error"):
        return
    steps = [("upscaler", _load_upscaler)]
    if LMS_FILE.strip().lower() not in ("", "none"):
        steps.append(("LMS", _load_lms))
    for name, load in steps:
        try:
            load()
        except Exception as error:  # the call path loads it again (and says so)
            note = f"{name} not loaded at startup: {type(error).__name__}: {error}"
            APP.setdefault("notes", []).append(note)
            print(f"[mc] {note}", flush=True)


@contextlib.contextmanager
def _lora(pairs, strength):
    """Each pair as an unfused LoRA on its linear while the block runs: ``y += strength · (x Aᵀ) Bᵀ``,
    added in place in row chunks (an AdaLN projection's output is the size of the sequence × 96k)."""
    modules = dict(APP["pipe"].transformer_ref.named_modules())
    handles = []

    def hook(a, b):
        def add(module, inputs, out):
            x = inputs[0].reshape(-1, a.shape[1])
            y = out if out.is_contiguous() else out.contiguous()
            flat = y.view(-1, b.shape[0])
            rows = max(1, LMS_CHUNK // b.shape[0])
            for i in range(0, x.shape[0], rows):
                part = F.linear(F.linear(x[i:i + rows].to(a.dtype), a), b)
                flat[i:i + rows].add_(part.to(flat.dtype), alpha=strength)
            return y
        return add

    try:
        for name, a, b in pairs:
            handles.append(modules[name].register_forward_hook(hook(a.to("cuda"), b.to("cuda"))))
        yield len(handles)
    finally:
        for h in handles:
            h.remove()


# ---- pipeline helpers ----------------------------------------------------------------------------
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


@contextlib.contextmanager
def _explicit_sigmas(pipe, tail):
    """Both schedulers run ``tail + [0]`` verbatim (no shift) whatever step count they are asked."""
    full = [*tail, 0.0]
    for sched in (pipe.scheduler, pipe.audio_scheduler):
        original = sched.set_timesteps
        sched.set_timesteps = lambda n=None, device=None, _o=original, **kw: _o(device=device, sigmas=full)
    try:
        yield
    finally:
        for sched in (pipe.scheduler, pipe.audio_scheduler):
            sched.__dict__.pop("set_timesteps", None)


def _encode(pipe, pixels):
    """A clip or a picture through the VAE encoder (app.encode_condition), 128 px tiles on failure."""
    encode = APP["encode_condition"]
    args = (pipe.pixel_mean, pipe.pixel_std, pipe.keyframe_encode_seed)
    pixels = pixels.to("cuda")
    try:
        return encode(pipe.vae, pixels, *args)
    except RuntimeError as error:
        error.__traceback__ = None
        torch.cuda.empty_cache()
        print(f"[mc] encode failed ({type(error).__name__}); retrying on 128px tiles", flush=True)
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
        print(f"[mc] decode failed ({type(error).__name__}); retrying tiled", flush=True)
        vae.enable_tiling()
        video = run()
    return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()


@spaces.GPU(duration=gpu_s, size=GPU_SIZE)
def _run(reference, guide, guide_up, embeds, tags, height, width, frames, steps, seed, up_height,
         up_width, up_sigmas, embeds_up=None, tags_up=None, up_lms=0.0):
    pipe, times = APP["pipe"], {}
    pipe.vae.to("cuda")
    t0 = time.time()
    ref = _encode(pipe, reference)
    lat_guide = _encode(pipe, guide)
    del guide
    torch.cuda.empty_cache()
    times["encode"] = time.time() - t0
    common = dict(prompt_embeds=embeds.to("cuda"), text_token_tags=tags, num_reference_images=1,
                  num_frames=int(frames))
    tail = _tail(up_sigmas) if int(up_width) else []
    t0 = time.time()
    if not tail:
        state = pipe(**common, condition_latents=[ref, lat_guide], height=int(height), width=int(width),
                     num_inference_steps=int(steps) + 1,
                     generator=torch.Generator("cpu").manual_seed(int(seed)), output_type="np")
        times["pass1"] = time.time() - t0
        video = state.get("videos")[0]  # float [0, 1]: sent back as uint8 (4x less to pickle)
        return (video.clip(0.0, 1.0) * 255.0).round().astype("uint8"), times
    with _latents_only():
        out = pipe(**common, condition_latents=[ref, lat_guide], height=int(height), width=int(width),
                   num_inference_steps=int(steps) + 1,
                   generator=torch.Generator("cpu").manual_seed(int(seed)),
                   output=["latents", "audio_latents"])
    times["pass1"] = time.time() - t0
    t0 = time.time()
    up = _upscale(out["latents"], int(up_height), int(up_width))
    lat_up = _encode(pipe, guide_up)
    del guide_up
    torch.cuda.empty_cache()
    times["upscale"] = time.time() - t0
    t0 = time.time()
    g = torch.Generator("cpu").manual_seed(int(seed) + 1)
    s0 = tail[0]
    noisy = (1 - s0) * up + s0 * torch.randn(up.shape, generator=g).to(up.device)
    audio = out["audio_latents"].float()
    noisy_audio = (1 - s0) * audio + s0 * torch.randn(audio.shape, generator=g).to(audio.device)
    second = dict(common)
    if embeds_up is not None:  # the prompt with the LMS caption
        second.update(prompt_embeds=embeds_up.to("cuda"), text_token_tags=tags_up)
    lms = float(up_lms or 0)
    with _latents_only(), _explicit_sigmas(pipe, tail), (
            _lora(_LMS["pairs"], lms) if lms > 0 else contextlib.nullcontext()):
        out = pipe(**second, condition_latents=[ref, lat_up], height=int(up_height), width=int(up_width),
                   num_inference_steps=len(tail) + 1, latents=noisy, audio_latents=noisy_audio,
                   generator=torch.Generator("cpu").manual_seed(int(seed) + 2), output=["latents"])
    times["pass2"] = time.time() - t0
    t0 = time.time()
    video = _decode(pipe, out["latents"])
    times["decode"] = time.time() - t0
    return video, times


def _write(frames, path):
    """(T, H, W, 3) frames as H.264 at 24 fps, CRF 10, BT.709 YUV and tags (the engine's H3 writer)."""
    import av
    import numpy as np

    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = {"crf": "10", "preset": "medium",
                          "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", interpolation="BICUBIC")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def mc_charswap(video, image, prompt, frames, width, height, steps=8, seed=46, up_width=0, up_height=0,
                up_sigmas="", up_lms=0.0, up_caption=0, progress=gr.Progress()):
    if APP.get("error"):
        raise gr.Error(str(APP["error"]).replace("**", ""))
    if not video or not image or not str(prompt or "").strip():
        raise gr.Error("video, image and prompt are required")
    video, image = str(video), str(image)  # gr.File gives gradio's NamedString; PyAV takes only str
    frames, width, height = int(frames), int(width), int(height)
    up_width, up_height = int(up_width or 0), int(up_height or 0)
    if frames < 5 or frames % 17 != 5:
        raise gr.Error(f"frames must be 17n+5, got {frames}")
    if width % 32 or height % 32 or up_width % 32 or up_height % 32:
        raise gr.Error("sizes must be multiples of 32")
    tail = _tail(up_sigmas) if up_width else []
    if up_width and (not tail or any(b >= a for a, b in zip(tail, tail[1:])) or not 0 < tail[0] < 1):
        raise gr.Error(f"up_sigmas must decrease in (0, 1), got {up_sigmas!r}")
    total, fps, _, _ = h3_video.probe(video)
    if total < frames:
        raise gr.Error(f"the guide has {total} frames, {frames} asked")
    t0 = time.time()
    embeds, tags = APP["encode_prompt"](str(prompt).strip(), image)
    reference = h3_video.load_image(image, APP.get("reference_mp", 0.3))
    plan = {"num_frames": frames, "source_frames": total, "source_fps": fps}
    guide = h3_video.load_guide(video, {**plan, "height": height, "width": width})
    guide_up, embeds_up, tags_up, lms, notes = None, None, None, 0.0, list(APP.get("notes") or [])
    if tail:
        _load_upscaler()
        guide_up = h3_video.load_guide(video, {**plan, "height": up_height, "width": up_width})
        lms = float(up_lms or 0) if LMS_FILE.strip().lower() not in ("", "none") else 0.0
        if lms > 0:
            try:
                _load_lms()
            except Exception as error:  # the second pass still runs, without it
                notes.append(f"LMS off: {type(error).__name__}: {error}")
                lms = 0.0
        if lms > 0 and int(up_caption or 0):
            embeds_up, tags_up = APP["encode_prompt"](f"{str(prompt).strip()} {LMS_CAPTION}", image)
    prep = time.time() - t0
    out, times = _run(reference, guide, guide_up, embeds, tags, height, width, frames, int(steps),
                      int(seed), up_height, up_width, ",".join(str(v) for v in tail), embeds_up,
                      tags_up, lms)
    path = _write(out, os.path.join(tempfile.mkdtemp(prefix="mc_"), "charswap.mp4"))
    size = f"{up_width}x{up_height}" if tail else f"{width}x{height}"
    report = (f"{size} · {frames} frames · prep {prep:.1f} s · "
              + " · ".join(f"{k} {v:.1f} s" for k, v in times.items())
              + f" · steps {int(steps)}" + (f" + {len(tail)}" if tail else "")
              + (f" · LMS {lms:g}" + (" + caption" if embeds_up is not None else "") if lms else "")
              + f" · seed {int(seed)} · " + "; ".join(notes))
    return path, report
