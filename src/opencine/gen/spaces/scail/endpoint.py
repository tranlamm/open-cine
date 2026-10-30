# mc: one-call API (Open Cine): reference on black, its mask, driving clip, mask clip ->
# mp4. Raises on failure (the UI endpoints return the traceback as text instead).
def _mc_gpu_s(image, mask_image, video, mask_video, prompt, replace, steps, guide_scale, shift,
              seed, width, height, frames, *args, **kwargs):
    nfe = int(steps) * (2 if float(guide_scale) > 1 else 1)
    work = float(frames) * int(width) * int(height) / (81 * 512 * 896)
    return int(min(GPU_DURATION_MAX, 45 + nfe * 17 * work ** 1.15 * 1.2))


@spaces.GPU(duration=_mc_gpu_s, size=GPU_SIZE)
def mc_scail(image, mask_image, video, mask_video, prompt, replace, steps, guide_scale, shift,
             seed, width, height, frames, progress=gr.Progress(track_tqdm=True)):
    return str(_run_scail_job(image, mask_image, video, mask_video, prompt, bool(replace),
                              int(height), int(width), int(steps), float(guide_scale),
                              float(shift), int(seed), DEFAULT_SEGMENT_LEN,
                              DEFAULT_SEGMENT_OVERLAP, progress=progress))
