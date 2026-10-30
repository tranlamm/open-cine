# ---- face pass (Open Cine; ``opencine gen hf-duplicate video_ref`` writes it) -----------
FACE_SHIFT = 12.0
FACE_CANVAS = "960x544 · 16:9 fast"  # an image-only embedding: the conditioner checks, never uses it
FACE_COND_FRAMES = 124
FACE_ENCODE_S = 10


def face_need_s(size, n, steps, pictures):
    return ref_need_s(size, size, n, steps, pictures, 0) + FACE_ENCODE_S


def _face_gpu_s(references, crop, audio, embeds, tags, size, frames, steps, seed, strengths, *a, **k):
    return int(min(GPU_MAX, face_need_s(int(size), int(frames), int(steps), len(references))))


def _sigma(d):
    return FACE_SHIFT * d / (1 + (FACE_SHIFT - 1) * d)


def face_schedule(strengths, steps):
    top = max(strengths)
    return [_sigma(top * (1 - k / steps)) for k in range(steps)], [_sigma(d) for d in strengths]


def _per_latent(values, n_latent):
    """Per-frame values on the latent frames (linear, as the pack does)."""
    if n_latent == 1 or len(values) == 1:
        return [max(values)] * n_latent
    return np.interp(np.linspace(0, len(values) - 1, n_latent), np.arange(len(values)), values).tolist()


@contextlib.contextmanager
def _face_sigmas(pipe, tail):
    """The video scheduler runs ``tail + [0]`` verbatim (no shift) whatever step count it is asked;
    the audio one keeps its own full schedule over as many steps (a free soundtrack is made from
    noise; a locked one is never stepped)."""
    full, sched = [*tail, 0.0], pipe.scheduler
    original = sched.set_timesteps
    sched.set_timesteps = lambda n=None, device=None, _o=original, **kw: _o(device=device, sigmas=full)
    try:
        yield
    finally:
        sched.__dict__.pop("set_timesteps", None)


@contextlib.contextmanager
def _face_loop(x0_rows, eps_rows, starts, per_frame, lock_audio):
    """The denoise loop of this call: after every step a latent frame whose start sigma is below the
    level the rows now sit at is put back on that level from the clean crop (a frame at 0 stays the
    crop); with ``lock_audio`` every audio row is a condition row (timestep 1 = clean, never stepped)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep as plan
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep as update

    build, step = plan.__dict__["build_row_timesteps"], update.__dict__["__call__"]

    def rows(video_indices, audio_indices, ncv, nca, *rest):
        return build.__func__(video_indices, audio_indices, ncv,
                              audio_indices.numel() if lock_audio else nca, *rest)

    def held(self, components, block_state, i, t):
        nca = block_state.num_condition_audio_rows
        if lock_audio:
            block_state.num_condition_audio_rows = block_state.audio_latents.shape[0]
        try:
            components, block_state = step(self, components, block_state, i, t)
        finally:
            block_state.num_condition_audio_rows = nca
        sched = components.scheduler
        level = float(sched.sigmas[sched.step_index])
        gen = block_state.latents[block_state.num_condition_video_rows:]
        for f, s0 in enumerate(starts):
            if s0 <= 0 or s0 < level - 1e-6:
                r = slice(f * per_frame, (f + 1) * per_frame)
                gen[r] = (1 - level) * x0_rows[r] + level * eps_rows[r]
        return components, block_state

    plan.build_row_timesteps = staticmethod(rows)
    update.__call__ = held
    try:
        yield
    finally:
        plan.build_row_timesteps = build
        update.__call__ = step


def _wav(path, samples):
    """A 16-bit PCM WAV at 32 kHz (the engine writes it) as (2, 1, samples) float in [-1, 1]."""
    import wave

    with wave.open(path) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate != 32000 or width != 2:
        raise gr.Error(f"the soundtrack must be 16-bit 32 kHz WAV, got {rate} Hz, {8 * width} bit")
    x = data.reshape(-1, channels).T.astype(np.float32) / 32768.0
    x = np.repeat(x, 2, axis=0) if channels == 1 else x[:2]
    x = np.pad(x, ((0, 0), (0, max(0, samples - x.shape[1]))))[:, :samples]
    return torch.from_numpy(np.ascontiguousarray(x))[:, None]


@torch.no_grad()
def _encode_audio(wave_, n_latents):
    """The soundtrack through the audio VAE (its posterior mode), normalized like the pipeline's
    audio rows: (2, 32, n_latents); None when the audio VAE did not load."""
    vae = _AUDIO.get("vae")
    if vae is None:
        return None
    vae.to("cuda")
    z = vae.encode(wave_.to("cuda", torch.float32), return_dict=False)[0].mode().float()
    vae.to("cpu")
    if z.shape[-1] < n_latents:
        z = torch.cat([z, z[..., -1:].expand(-1, -1, n_latents - z.shape[-1])], dim=-1)
    mean = torch.tensor(vae.config.latents_mean, device=z.device).view(1, -1, 1)
    std = torch.tensor(vae.config.latents_std, device=z.device).view(1, -1, 1)
    return ((z[..., :n_latents] - mean) / std).cpu()


@spaces.GPU(duration=_face_gpu_s, size=GPU_SIZE)
def _face(references, crop, audio, embeds, tags, size, frames, steps, seed, strengths):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    pipe, times = APP["pipe"], {}
    pipe.vae.to("cuda")
    t0 = time.time()
    refs = [_encode(pipe, r) for r in references]
    x0 = _encode(pipe, crop)
    del crop
    torch.cuda.empty_cache()
    sound = _encode_audio(audio, round(int(frames) / 24 * 40)) if audio is not None else None
    times["encode"] = time.time() - t0
    tail, starts = face_schedule(strengths, int(steps))
    n_latent = x0.shape[2]
    eps = torch.randn(x0.shape, generator=torch.Generator("cpu").manual_seed(int(seed)))
    noisy = (1 - tail[0]) * x0 + tail[0] * eps
    x0_rows = patchify_video_latents(x0.to("cuda", torch.float32), pipe.patch_size)
    eps_rows = patchify_video_latents(eps.to("cuda", torch.float32), pipe.patch_size)
    t0 = time.time()
    with _latents_only(), _guide_frames(0), _face_sigmas(pipe, tail), _face_loop(
            x0_rows, eps_rows, _per_latent(starts, n_latent), x0_rows.shape[0] // n_latent,
            sound is not None):
        out = pipe(prompt_embeds=embeds.to("cuda"), text_token_tags=tags, condition_latents=refs,
                   num_reference_images=len(references), height=int(size), width=int(size),
                   num_frames=int(frames), num_inference_steps=len(tail) + 1, latents=noisy,
                   audio_latents=sound, generator=torch.Generator("cpu").manual_seed(int(seed) + 1),
                   output=["latents"])
    times["denoise"] = time.time() - t0
    t0 = time.time()
    video = _decode(pipe, out["latents"])
    times["decode"] = time.time() - t0
    return video, times, sound is not None


def mc_facefix(crop, images, audio, prompt, frames, size, steps=8, seed=42, strengths="",
               progress=gr.Progress()):
    if APP.get("error"):
        raise gr.Error(str(APP["error"]).replace("**", ""))
    pictures = [_path(x) for x in (images or []) if x]
    if not crop or not pictures or len(pictures) > MAX_PICTURES:
        raise gr.Error(f"give the crop clip and 1-{MAX_PICTURES} pictures")
    if not str(prompt or "").strip().startswith("subject_definitions:"):
        raise gr.Error("the prompt must be the six-field Ref2VA text (subject_definitions: ...)")
    crop, frames, size, steps = _path(crop), int(frames), int(size), int(steps)
    if frames < 5 or frames % 17 != 5:
        raise gr.Error(f"frames must be 17n+5, got {frames}")
    if size % 32 or not 256 <= size <= 1024:
        raise gr.Error("the crop size must be a multiple of 32 in 256-1024")
    d = [float(v) for v in str(strengths or "").replace(" ", "").split(",") if v]
    if len(d) != frames or min(d, default=-1) < 0 or not 0 < max(d, default=0) < 1:
        raise gr.Error(f"give one strength in [0, 1) per frame ({frames}), the strongest above 0")
    if face_need_s(size, frames, steps, len(pictures)) > GPU_MAX:
        raise gr.Error(f"{frames} frames at {size}x{size}, {steps} steps need more than {GPU_MAX} s")
    total, fps, _, _ = h3_video.probe(crop)
    if total < frames:
        raise gr.Error(f"the crop has {total} frames, {frames} asked")
    t0 = time.time()
    embeds, tags = _embed(str(prompt).strip(), pictures, FACE_CANVAS, FACE_COND_FRAMES)
    references = [h3_video.load_image(p, size * size / 1e6) for p in pictures]
    plan = {"num_frames": frames, "source_frames": total, "source_fps": fps, "height": size, "width": size}
    pixels = h3_video.load_guide(crop, plan)
    wave_ = _wav(_path(audio), round(frames / 24 * 40) * 800) if audio else None
    prep = time.time() - t0
    video, times, locked = _face(references, pixels, wave_, embeds, tags, size, frames, steps, int(seed), d)
    path = _write(video, os.path.join(tempfile.mkdtemp(prefix="mc_"), "facefix.mp4"))
    report = (f"{size}x{size} · {frames} frames · {len(pictures)} pictures · denoise {max(d):.2f}-"
              f"{min(d):.2f} · audio {'locked' if locked else 'free'} · prep {prep:.1f} s · "
              + " · ".join(f"{k} {v:.1f} s" for k, v in times.items())
              + f" · steps {steps} · seed {int(seed)} · " + "; ".join(APP.get("notes") or []))
    return path, report


_bind_ref2va = bind


def bind(**kwargs):
    """app.py's hand-over (the Ref2VA one), then the face pass's startup check."""
    _bind_ref2va(**kwargs)
    try:
        note = _face_check()
    except Exception as error:  # the Ref2VA endpoint still serves; the face pass will fail loudly
        note = f"face pass not ready: {type(error).__name__}: {error}"
        APP.setdefault("notes", []).append(note)
    print(f"[mc] {note}", flush=True)


def _face_check():
    """What the face pass leans on, checked on the CPU at startup (no GPU): the two diffusers steps
    it wraps, the latent packing, the audio VAE encoder on a quarter second of silence."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep

    if APP.get("error") or APP.get("pipe") is None:
        return "face pass not ready: the pipeline did not load"
    if not isinstance(MiniMaxH3SetTimestepsStep.__dict__.get("build_row_timesteps"), staticmethod):
        raise RuntimeError("MiniMaxH3SetTimestepsStep.build_row_timesteps is no staticmethod")
    if "__call__" not in MiniMaxH3LoopSchedulerStep.__dict__:
        raise RuntimeError("MiniMaxH3LoopSchedulerStep defines no __call__")
    patch = tuple(APP["pipe"].patch_size)
    rows = patchify_video_latents(torch.zeros(1, 24, 2, 4, 4), patch)
    vae = _AUDIO.get("vae")
    if vae is None:
        return f"face pass ready (patch {patch}, {tuple(rows.shape)} rows); no audio VAE: audio free"
    with torch.no_grad():
        z = vae.encode(torch.zeros(2, 1, 8000), return_dict=False)[0].mode()
    if tuple(z.shape) != (2, len(vae.config.latents_mean), 10):
        raise RuntimeError(f"audio VAE encodes 0.25 s to {tuple(z.shape)}, expected (2, 32, 10)")
    return f"face pass ready (patch {patch}, audio lock {tuple(z.shape)})"
