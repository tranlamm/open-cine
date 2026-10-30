# mc: one-call API (Open Cine): styled first frame + control clip (depth or edges, half
# size) -> restyled mp4; no prompt enhancement, no source audio. Raises on failure.
def _mc_gpu_s(image, control, prompt, frames, height, width, strength, seed, *args, **kwargs):
    return int(min(240, max(60, 40 + int(height) * int(width) * int(frames) / 1.2e6)))


@spaces.GPU(duration=_mc_gpu_s, size="xlarge")  # ~50 GB of weights: the large half GPU is 48
@torch.inference_mode()
def mc_restyle(image, control, prompt, frames, height, width, strength, seed,
               progress=gr.Progress(track_tqdm=True)):
    frames, tiling = int(frames), TilingConfig.default()
    video, audio = pipeline(
        prompt=prompt, seed=int(seed), height=int(height), width=int(width), num_frames=frames,
        frame_rate=DEFAULT_FRAME_RATE,
        images=[ImageConditioningInput(path=str(image), frame_idx=0, strength=1.0)],
        audio_path=None, video_conditioning=[(str(control), 1.0)], tiling_config=tiling,
        enhance_prompt=False, conditioning_strength=float(strength))
    out = tempfile.mktemp(suffix=".mp4")
    encode_video(video=video, fps=DEFAULT_FRAME_RATE, audio=audio, output_path=out,
                 video_chunks_number=get_video_chunks_number(frames, tiling))
    return out
