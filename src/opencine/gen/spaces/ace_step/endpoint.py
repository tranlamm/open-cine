# mc: one-call instrumental endpoint (Open Cine): caption, seconds, bpm, key, seed -> wav
_MC = {}


def _mc_gpu_s(caption, seconds, bpm, keyscale, seed, thinking):
    return int(min({{GPU_MAX_S}}, {{GPU_BASE_S}} + {{GPU_PER_S}} * float(seconds or 30)))


def mc_music(caption, seconds, bpm, keyscale, seed, thinking):
    import json
    import tempfile

    import gradio as gr
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    seed = int(seed)
    params = GenerationParams(caption=caption, lyrics="[Instrumental]", instrumental=True,
                              bpm=int(bpm) or None, keyscale=keyscale or "",
                              duration=float(seconds), seed=seed, thinking=bool(thinking))
    config = GenerationConfig(batch_size=1, use_random_seed=seed < 0,
                              seeds=[seed] if seed >= 0 else None, audio_format="wav")
    res = generate_music(_MC["dit"], _MC["llm"], params, config, save_dir=tempfile.mkdtemp())
    if not res.success or not res.audios or not res.audios[0].get("path"):
        raise gr.Error(res.error or res.status_message or "generation failed")
    meta = {k: v for k, v in (res.audios[0].get("params") or {}).items()
            if isinstance(v, (str, int, float, bool)) and k != "audio_codes"}
    return res.audios[0]["path"], json.dumps(meta)


if HAS_SPACES:
    mc_music = spaces.GPU(duration=_mc_gpu_s)(mc_music)
