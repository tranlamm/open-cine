"""One-call API for Open Cine (``opencine gen hf-duplicate enhance`` writes this file; a rerun
replaces it): clip, output size (multiples of 128), temporal compression, colour correction,
frames -> mp4 (the clip's audio, if any, muxed back). app.py hands over its pipeline."""
import os
import tempfile

import gradio as gr
import spaces

from inference.request import InferenceRequest

APP = {}  # pipeline, decode_every


def gpu_s(video, width, height, compression, color, frames, *args, **kwargs):
    """Fitted on real calls + 25 % + 5 s (the engine's ``_enhance_gpu_s``)."""
    cost = {{FIXED_S}} + {{LATENT_MP_S}} * max(int(frames), 17) / int(compression) * (
        int(width) * int(height) / 1e6)
    return int(min({{GPU_MAX_S}}, cost * 1.25 + 5))


@spaces.GPU(duration=gpu_s)
def mc_enhance(video, width, height, compression, color, frames, progress=gr.Progress()):
    if video is None:
        raise gr.Error("no video")
    pipeline = APP["pipeline"]
    pipeline.config.seed = 2026
    pipeline.config.temporal_compression = int(compression)
    pipeline.config.decoder.decode_every = APP["decode_every"]
    if color:
        from inference.postprocessing import WaveletColorCorrector

        if not isinstance(pipeline.color_corrector, WaveletColorCorrector):
            pipeline.color_corrector = WaveletColorCorrector(levels=5, chunk_size=16)
    else:
        pipeline.color_corrector = None
    out = os.path.join(tempfile.mkdtemp(prefix="mc_"), "enhanced.mp4")
    return pipeline.infer(InferenceRequest(input_path=video, output_path=out, height=int(height),
                                           width=int(width), max_frames=int(frames))).output_path
