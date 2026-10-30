quality=None,  # mc: CRF 12, BT.709 YUV and tags (quality 5 was CRF 25, BT.601)
output_params=["-preset", preset, "-crf", "12",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-color_range", "tv",
               "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"],
