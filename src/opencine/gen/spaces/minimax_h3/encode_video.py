def _mc_encode_video(video, fps, output_path, audio=None, audio_sample_rate=None):
    """`diffusers.utils.encode_video` at CRF 10 with BT.709 YUV and tags (patched in by mc)."""
    import av
    import numpy as np

    if isinstance(video, list):
        video = np.stack([np.asarray(frame.convert("RGB")) for frame in video])
    elif hasattr(video, "cpu"):
        video = video.float().cpu().numpy() if video.is_floating_point() else video.cpu().numpy()
    video = np.asarray(video)
    if video.ndim == 4 and video.shape[-1] != 3 and video.shape[1] == 3:
        video = video.transpose(0, 2, 3, 1)
    if video.dtype != np.uint8:
        video = (video * 255.0).round().clip(0, 255).astype(np.uint8)
    _, height, width, _ = video.shape
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "10", "preset": "medium",
                      "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"}
    audio_stream = None
    if audio is not None:
        from diffusers.utils import export_utils

        audio_stream = export_utils._prepare_audio_stream(container, audio_sample_rate)
    for array in video:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", interpolation="BICUBIC")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    if audio is not None:
        from diffusers.utils import export_utils

        export_utils._write_audio(container, audio_stream, audio, audio_sample_rate, av)
    container.close()
