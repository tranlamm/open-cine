"""Scribble layer: hand-drawn strokes that follow things in a rendered EditPlan (outlines, fills,
rings, trails, riding doodles, a frame border).

``anchor`` resolves "S03@2.1" / "say:crispy" to timeline seconds, ``track`` follows things
(EdgeTAM masks, optical flow), ``place`` positions doodles, ``service`` draws with Remotion
(``fx/``) and overlays with FFmpeg.
"""
