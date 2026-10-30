"""Frame sampling and per-scene description.

Samples the proxy at ``SAMPLE_FPS`` (sharpness, brightness, clipping, motion, optical flow, colour),
drops bad frames, clusters the rest into keyframes, and gives each scene the L0 fields: camera
move, subject motion, head/tail direction, dominant colour and strongest sound onset.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean, median

import cv2
import imagehash
import numpy as np
from PIL import Image

from opencine.core.ffmpeg import run
from opencine.core.time import Rate, frames_for, to_frame, to_seconds
from opencine.index.scenes import merge_short, split_long

SAMPLE_FPS = 2
EDGE = 480  # long edge of analysed / saved frames
JPEG_Q = 85
MOTION_GAP_S = 0.25  # motion = mean |Δgrey| against a frame this far away in the same shot
MOTION_FULL = 20.0  # mean |Δgrey| (0–255, blurred 160 px) that maps to motion 1.0
SHARP_FULL = 1000.0  # Laplacian variance that maps to sharpness 1.0 (log scale)
BLACK_MAX, CLIPPED_MAX, BLUR_REL = 0.6, 0.3, 0.4  # bad-frame thresholds
CLUSTER_DIST = 0.3  # pHash + HSV histogram distance, average linkage
MAX_KEYFRAMES = 3
# keyframe = cluster member trading sharpness against typicality (the bare medoid is often
# motion-blurred): sharpness (0–1, log) given up per unit of mean distance to the cluster
REP_DIST_WEIGHT = 2.0
# speeds are frame widths per second; zoom is relative scale change per second
CAM_FULL = SUBJECT_FULL = 0.5  # speed that maps to camera / subject motion 1.0
MOVE_MIN = 0.08  # speed that counts as moving (a pan crossing the frame in ~12 s)
ZOOM_MIN = 0.06  # zoom that counts (6 %/s)
SHAKE_MIN = 0.15  # camera motion level without a steady direction that reads as handheld shake
STEADY = 0.6  # |mean motion| / mean |motion| above which the camera moves one way
AGREE = 0.75  # … and the share of samples moving the same way
FLAT_STD = 3.0  # grey std (0–255) under which a frame has no texture to track
EDGE_S = 1.0  # head / tail window of a scene
HUE_BINS, CHROMA_S, CHROMA_V, CHROMA_MIN = 36, 60, 40, 0.25  # colour: 10° bins; saturated pixels
AUDIO_SR, AUDIO_HOP, AUDIO_PEAK_REL = 22050, 512, 4.0  # onset peak ≥ 4× the clip's median
AUDIO_LEAD_S = 0.15  # onsets this close to the clip start are the recording starting
PARAMS = {"fps": SAMPLE_FPS, "edge": EDGE, "q": JPEG_Q, "gap": MOTION_GAP_S,
          "motion": MOTION_FULL, "sharp": SHARP_FULL, "bad": [BLACK_MAX, CLIPPED_MAX, BLUR_REL],
          "cluster": CLUSTER_DIST, "k": MAX_KEYFRAMES, "rep": REP_DIST_WEIGHT,
          "flow": [CAM_FULL, MOVE_MIN, ZOOM_MIN, SHAKE_MIN, STEADY, AGREE, FLAT_STD, EDGE_S, 4],
          "colour": [HUE_BINS, CHROMA_S, CHROMA_V, CHROMA_MIN],
          "audio": [AUDIO_SR, AUDIO_HOP, AUDIO_PEAK_REL, AUDIO_LEAD_S],
          "samples": [2, 16]}  # cache key part; samples: index/samples/<clip>.json (solver, match)
GRID = 16  # samples keep a 16×16 grey thumbnail for shape / light matching
SAMPLES_FILE = "samples.json"
Vec = tuple[float, float]


@dataclass
class Sample:
    frame: int
    t: float  # seconds within the clip
    sharp: float  # Laplacian variance
    bright: float  # mean luma 0–1
    clipped: float  # fraction of pixels ≥ 250
    black: float  # fraction of pixels ≤ 8
    phash: imagehash.ImageHash
    hist: np.ndarray
    jpeg: bytes
    motion: float = 0.0  # raw mean |Δgrey|
    cam: Vec = (0.0, 0.0)  # picture motion the camera makes (median flow), frame widths/s
    zoom: float = 0.0  # + = the picture grows (camera pushes in), per second
    subject: float = 0.0  # speed of the fastest tenth against the camera motion
    subject_vec: Vec = (0.0, 0.0)  # their mean direction × speed
    hues: np.ndarray | None = None  # share of the frame that is saturated, per 10° hue bin
    hue_v: np.ndarray | None = None  # summed brightness (0–1) of those pixels, per bin
    grey_v: float = 0.5  # mean brightness of the rest
    lab: tuple[float, float, float] = (50.0, 0.0, 0.0)  # mean CIELAB colour
    grid: list[int] | None = None  # GRID×GRID grey thumbnail, 0–255


def time_name(t: float) -> str:
    return f"t{round(t * 1000):07d}.jpg"


def frame_name(clip: str, t: float) -> str:
    """File name of a saved sample under ``index/frames/``."""
    return f"{clip}_{time_name(t)}"


def _resize(img: np.ndarray, edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    k = edge / max(h, w)
    return img if k >= 1 else cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def _motion_grey(img: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(_resize(img, 160), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (5, 5), 0).astype(np.int16)


def _measure(frame: int, rate: Rate, img: np.ndarray) -> Sample:
    small = _resize(img, EDGE)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    assert ok
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float64) / 255
    chroma = (sat >= CHROMA_S) & (val >= CHROMA_V / 255)
    bins = (h[chroma].astype(int) * HUE_BINS // 180).clip(0, HUE_BINS - 1)
    return Sample(
        frame=frame,
        t=round(float(to_seconds(frame, rate)), 3),
        sharp=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        bright=float(grey.mean() / 255),
        clipped=float((grey >= 250).mean()),
        black=float((grey <= 8).mean()),
        phash=imagehash.phash(Image.fromarray(grey)),
        hist=cv2.normalize(hist, None).flatten(),
        jpeg=jpg.tobytes(),
        hues=np.bincount(bins, minlength=HUE_BINS) / chroma.size,
        hue_v=np.bincount(bins, weights=val[chroma], minlength=HUE_BINS) / chroma.size,
        grey_v=float(val[~chroma].mean()) if (~chroma).any() else 0.5,
        lab=_lab_mean(small),
        grid=cv2.resize(grey, (GRID, GRID), interpolation=cv2.INTER_AREA).flatten().tolist(),
    )


def _lab_mean(img: np.ndarray) -> tuple[float, float, float]:
    """Mean CIELAB (L 0–100, a/b signed) of a BGR image (OpenCV's 8-bit Lab rescaled)."""
    L, a, b = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    return round(float(L) * 100 / 255, 1), round(float(a) - 128, 1), round(float(b) - 128, 1)


def _flow(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[Vec, float, float, Vec]:
    """Optical flow ``a → b`` (small greys ``dt`` s apart) → ``(camera, zoom, subject speed,
    subject vector)``; speeds in frame widths/s, zoom in scale change/s.

    Phase correlation removes the global shift, Farneback measures the rest. Camera =
    translation + scale least-squares fit on the inner frame (content enters and leaves at the
    border), refitted on the best-fitting 80 %; subject = the fastest tenth of what the fit leaves
    (inner, textured pixels)."""
    a8, b8 = a.astype(np.uint8), b.astype(np.uint8)
    h, w = a8.shape
    if a8.std() < FLAT_STD or b8.std() < FLAT_STD:  # nothing to track (black, a flat wall)
        return (0.0, 0.0), 0.0, 0.0, (0.0, 0.0)
    # global shift first, so Farneback only sees the small remainder
    (sx, sy), peak = cv2.phaseCorrelate(a8.astype(np.float64), b8.astype(np.float64))
    if peak < 0.05:  # no clear global shift
        sx = sy = 0.0
    b8 = cv2.warpAffine(b8, np.float64([[1, 0, -sx], [0, 1, -sy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    f = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    u, v = f[..., 0].astype(np.float64) + sx, f[..., 1].astype(np.float64) + sy
    ry, rx = np.mgrid[0:h, 0:w].astype(np.float64)
    rx -= (w - 1) / 2
    ry -= (h - 1) / 2
    m = int(min(h, w) * 0.08 + max(abs(np.median(u)), abs(np.median(v)))) + 1
    inner = (slice(m, max(m + 4, h - m)), slice(m, max(m + 4, w - m)))
    # flow is only trustworthy where there is texture: keep the better-textured half
    g = a.astype(np.float64)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))[inner].ravel()
    keep = grad >= np.quantile(grad, 0.5)
    u, v, rx, ry = (x[inner].ravel()[keep] for x in (u, v, rx, ry))
    use = np.ones(u.size, bool)
    for _ in range(2):
        A = np.zeros((2 * int(use.sum()), 3))
        A[0::2, 0], A[1::2, 1] = 1, 1
        A[0::2, 2], A[1::2, 2] = rx[use], ry[use]
        rhs = np.empty(len(A))
        rhs[0::2], rhs[1::2] = u[use], v[use]
        tx, ty, zoom = np.linalg.lstsq(A, rhs, rcond=None)[0]
        ru, rv = u - tx - zoom * rx, v - ty - zoom * ry
        mag = np.hypot(ru, rv)
        use = mag <= np.quantile(mag, 0.8)
    movers = mag >= np.quantile(mag, 0.9)
    k = 1 / (w * dt)  # px per interval → frame widths per second
    return ((float(tx) * k, float(ty) * k), float(zoom) / dt, float(mag[movers].mean()) * k,
            (float(ru[movers].mean()) * k, float(rv[movers].mean()) * k))


def sample(proxy: str | Path, rate: Rate, n_frames: int, cuts: list[int]) -> list[Sample]:
    """Decode the proxy once; measure every ``1/SAMPLE_FPS`` s and score motion within shots."""
    picks = sorted({f for k in range(math.ceil(n_frames / rate * SAMPLE_FPS) + 1)
                    if (f := to_frame(k / SAMPLE_FPS, rate, "nearest")) < n_frames})
    gap = max(1, frames_for(MOTION_GAP_S, rate))

    def shot(f: int) -> int:
        return bisect.bisect_right(cuts, f)

    partner = {}
    for f in picks:
        for p in (f + gap, f - gap):
            if 0 <= p < n_frames and shot(p) == shot(f):
                partner[f] = p
                break
    pick_set = set(picks)
    needed = pick_set | set(partner.values())

    cap = cv2.VideoCapture(str(proxy))
    samples: dict[int, Sample] = {}
    grey: dict[int, np.ndarray] = {}
    try:
        for i in range(max(needed) + 1):
            if not cap.grab():
                break
            if i not in needed:
                continue
            ok, img = cap.retrieve()
            if not ok:
                continue
            grey[i] = _motion_grey(img)
            if i in pick_set:
                samples[i] = _measure(i, rate, img)
    finally:
        cap.release()

    for f, s in samples.items():
        p = partner.get(f)
        if p in grey:
            s.motion = float(np.abs(grey[f] - grey[p]).mean())
            a, b = sorted((f, p))
            s.cam, s.zoom, s.subject, s.subject_vec = _flow(grey[a], grey[b], float(Fraction(b - a) / rate))
    return [samples[f] for f in sorted(samples)]


def _distance(a: Sample, b: Sample) -> float:
    hist = cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA)
    return 0.5 * (a.phash - b.phash) / 64 + 0.5 * float(hist)


def _clusters(samples: list[Sample]) -> tuple[list[list[int]], np.ndarray]:
    """Average-linkage agglomeration stopping at ``CLUSTER_DIST``; biggest cluster first."""
    n = len(samples)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = _distance(samples[i], samples[j])
    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        dist, i, j = min((d[np.ix_(gi, gj)].mean(), i, j)
                         for i, gi in enumerate(groups) for j, gj in enumerate(groups) if i < j)
        if dist > CLUSTER_DIST:
            break
        groups[i] += groups.pop(j)
    groups.sort(key=lambda g: (-len(g), min(g)))
    return groups, d


def _is_bad(s: Sample, med_sharp: float) -> bool:
    return s.black > BLACK_MAX or s.clipped > CLIPPED_MAX or s.sharp < BLUR_REL * med_sharp


def motion_norm(raw: float) -> float:
    return min(1.0, raw / MOTION_FULL)


def sharp_norm(var: float) -> float:
    return min(1.0, math.log10(1 + var) / math.log10(1 + SHARP_FULL))


def describe(samples: list[Sample]) -> tuple[dict, list[Sample]]:
    """Scene metrics + keyframes, and the samples worth saving (medoids, motion peak)."""
    med = median(s.sharp for s in samples)
    good = [s for s in samples if not _is_bad(s, med)] or samples
    groups, d = _clusters(good)
    medoids = [good[max(g, key=lambda i: sharp_norm(good[i].sharp) - REP_DIST_WEIGHT * d[i, g].mean())]
               for g in groups[:MAX_KEYFRAMES]]
    peak = max(samples, key=lambda s: s.motion)
    t0, t1 = samples[0].t, samples[-1].t
    edge = min(EDGE_S, max(1 / SAMPLE_FPS, (t1 - t0 + 1 / SAMPLE_FPS) / 2))
    info = {
        "motion": round(fmean(motion_norm(s.motion) for s in samples), 3),
        "brightness": round(fmean(s.bright for s in samples), 3),
        "sharpness": round(fmean(sharp_norm(s.sharp) for s in good), 3),
        "keyframes": [s.t for s in medoids],
        "motion_peak_s": peak.t,
        "keyframe_frames": [s.frame for s in medoids],
        "motion_peak_frame": peak.frame,
        "cam_motion": round(min(1.0, fmean(math.hypot(*s.cam) for s in samples) / CAM_FULL), 3),
        "cam_dir": camera_move(samples),
        "subject_motion": round(min(1.0, fmean(s.subject for s in samples) / SUBJECT_FULL), 3),
        "head_dir": screen_dir([s for s in samples if s.t < t0 + edge]),
        "tail_dir": screen_dir([s for s in samples if s.t > t1 - edge]),
        "colour": colour_name(samples),
    }
    return info, [*medoids, peak]


def _arrow(x: float, y: float) -> str:
    return ("→" if x > 0 else "←") if abs(x) >= abs(y) else ("↓" if y > 0 else "↑")


def _mean(vs: list[Vec]) -> Vec:
    return fmean(v[0] for v in vs), fmean(v[1] for v in vs)


def _zoom(samples: list[Sample]) -> str | None:
    """``zoom+`` / ``zoom-`` when the scale change outweighs the translation (a zoom moves the
    frame edge at half its rate, in frame widths)."""
    zoom = fmean(s.zoom for s in samples)
    if abs(zoom) >= ZOOM_MIN and abs(zoom) / 2 >= math.hypot(*_mean([s.cam for s in samples])):
        return "zoom+" if zoom > 0 else "zoom-"
    return None


def camera_move(samples: list[Sample]) -> str | None:
    """The camera's move over the scene: pan/tilt arrow (opposite to the picture's motion),
    zoom±, ``shake`` (moving with no steady direction: handheld), or None (still)."""
    if z := _zoom(samples):
        return z
    cam = _mean([s.cam for s in samples])
    if _steady(samples):
        return _arrow(-cam[0], -cam[1])
    if fmean(math.hypot(*s.cam) for s in samples) / CAM_FULL >= SHAKE_MIN:
        return "shake"
    return None


def _steady(samples: list[Sample]) -> bool:
    """The picture moves one way: fast enough on average, the mean keeps most of the individual
    speeds, and most samples agree in direction (shake averages out)."""
    cam = _mean([s.cam for s in samples])
    speed = math.hypot(*cam)
    agree = fmean(s.cam[0] * cam[0] + s.cam[1] * cam[1] > 0 for s in samples)
    return (speed >= MOVE_MIN and speed >= STEADY * fmean(math.hypot(*s.cam) for s in samples)
            and agree >= AGREE)


def screen_dir(samples: list[Sample]) -> str:
    """Which way things move on screen: the subject's direction when it outruns the camera's
    picture motion, else the picture's (zoom±, or a pan's arrow); ``·`` when nothing moves."""
    if not samples:
        return "·"
    cam = _mean([s.cam for s in samples])
    speed = math.hypot(*cam)
    subject, sv = fmean(s.subject for s in samples), _mean([s.subject_vec for s in samples])
    if subject >= max(MOVE_MIN, 2 * speed) and math.hypot(*sv) >= MOVE_MIN / 2:
        return _arrow(*sv)
    if z := _zoom(samples):
        return z
    return _arrow(*cam) if _steady(samples) else "·"


_HUES = [(15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "teal"),
         (250, "blue"), (290, "purple"), (345, "pink"), (361, "red")]  # (upper bound °, name)


def colour_name(samples: list[Sample]) -> str | None:
    """Dominant colour name: the strongest 30° hue window of the saturated pixels (warm and
    dark → ``brown``), or ``black``/``grey``/``white`` when too little is saturated."""
    hs = [s for s in samples if s.hues is not None]
    if not hs:
        return None
    hues = sum(s.hues for s in hs) / len(hs)
    if hues.sum() < CHROMA_MIN:
        v = fmean(s.grey_v for s in hs)
        return "black" if v < 0.2 else "grey" if v < 0.7 else "white"
    i = int(np.argmax(hues + np.roll(hues, 1) + np.roll(hues, -1)))
    win = [(i + k) % HUE_BINS for k in (-1, 0, 1)]
    w = hues[win]
    ang = np.deg2rad((np.array(win) + 0.5) * 360 / HUE_BINS)  # circular mean of the window
    deg = float(np.rad2deg(np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()))) % 360
    name = next(n for top, n in _HUES if deg < top)
    v = sum(s.hue_v[win].sum() for s in hs) / max(1e-9, sum(s.hues[win].sum() for s in hs))
    return "brown" if name in ("red", "orange", "yellow") and v < 0.45 else name


def onset_envelope(wav: Path) -> tuple[np.ndarray, float] | None:
    """Onset strength of a clip's audio and its step in seconds (None if silent)."""
    import tempfile

    import librosa
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        mono = Path(d) / "a.wav"
        run(["-y", "-loglevel", "error", "-i", str(wav), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
             "-c:a", "pcm_f32le", str(mono)])
        y, _ = sf.read(mono, dtype="float32")
    if y.size < AUDIO_HOP * 4 or not np.any(y):
        return None
    env = librosa.onset.onset_strength(y=y, sr=AUDIO_SR, hop_length=AUDIO_HOP)
    env[:math.ceil(AUDIO_LEAD_S * AUDIO_SR / AUDIO_HOP)] = 0  # the file starting is not an onset
    return env, AUDIO_HOP / AUDIO_SR


def audio_peak(env: np.ndarray, step: float, start_s: float, end_s: float) -> float | None:
    """Clip time of the strongest onset in ``[start_s, end_s)`` when it stands out (≥
    ``AUDIO_PEAK_REL``× the clip's median and ≥ 10 % of its loudest onset)."""
    i0, i1 = int(start_s / step), math.ceil(end_s / step)
    win = env[i0:i1]
    if not win.size:
        return None
    i = int(np.argmax(win))
    if win[i] <= 1e-6 or win[i] < AUDIO_PEAK_REL * float(np.median(env)) or win[i] < 0.1 * float(env.max()):
        return None
    return round((i0 + i) * step, 2)


def analyse_clip(proxy: str | Path, rate: Rate, n_frames: int, cuts: list[int],
                 out_dir: Path, audio: Path | None = None) -> list[dict]:
    """Scene dicts (``Scene`` fields minus ``id``/``clip``); keyframe JPEGs go to ``out_dir``.
    ``audio`` (the clip's WAV, same timeline as the proxy) adds each scene's sound onset peak."""
    samples = sample(proxy, rate, n_frames, cuts)
    onsets = onset_envelope(audio) if audio is not None and Path(audio).is_file() else None
    motion = [(s.frame, s.motion) for s in samples]
    spans = [p for span in merge_short(cuts, n_frames, rate) for p in split_long(span, motion, rate)]
    scenes, rows = [], []
    for a, b in spans:
        inside = [s for s in samples if a <= s.frame < b]
        if not inside:  # cannot happen for spans ≥ 1/SAMPLE_FPS; keep the nearest sample
            inside = [min(samples, key=lambda s: abs(s.frame - a))]
        med = median(s.sharp for s in inside)
        rows += [sample_row(s, med) for s in inside if a <= s.frame < b]
        info, keep = describe(inside)
        for s in keep:
            (out_dir / time_name(s.t)).write_bytes(s.jpeg)
        scenes.append({
            "start_frame": a,
            "end_frame": b,
            "start_s": round(float(to_seconds(a, rate)), 3),
            "end_s": round(float(to_seconds(b, rate)), 3),
            **info,
            "audio_peak_s": audio_peak(*onsets, float(to_seconds(a, rate)), float(to_seconds(b, rate)))
            if onsets else None,
        })
    (out_dir / SAMPLES_FILE).write_text(json.dumps(rows), "utf-8")
    return scenes


def screen_vec(s: Sample) -> Vec:
    """On-screen motion of one sample (frame widths/s): the subject's when it outruns the
    picture's own motion, else the picture's (as ``screen_dir`` reads a scene)."""
    if s.subject >= max(MOVE_MIN, 2 * math.hypot(*s.cam)):
        return s.subject_vec
    return s.cam


def sample_row(s: Sample, med_sharp: float) -> dict:
    """One ``index/samples/<clip>.json`` row: proxy frame, sharpness, brightness, motion (0–1),
    unusable (black, blown out, blurred), and for match cuts the mean Lab colour, a 16×16 grey
    thumbnail and the on-screen motion vector."""
    mv = screen_vec(s)
    return {"f": s.frame, "sharp": round(sharp_norm(s.sharp), 3), "bright": round(s.bright, 3),
            "motion": round(motion_norm(s.motion), 3), "bad": _is_bad(s, med_sharp),
            "lab": list(s.lab), "grid": s.grid or [], "mv": [round(mv[0], 3), round(mv[1], 3)]}
