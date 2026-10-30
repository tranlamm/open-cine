"""Timeline audio: source audio + music + SFX → one mix WAV at −14 LUFS / −1 dBTP.

Loudness is a plain gain, or gain into a 4×-oversampled lookahead limiter re-measured until it
lands (loudnorm's dynamic mode misses by up to 10 LU); the encoded file is checked again since
AAC can overshoot. Sample positions are exact (frame ``f`` at rate ``r`` = sample
``round(f / r · 48000)``). Edges: transitions crossfade; a hard cut is a 10 ms equal-power
crossfade (linear when one file continues without a gap); J/L-cuts (``audio_lead`` /
``audio_tail``) fade under the neighbouring picture; ramped items are muted; an item's ``sound``
plays other footage in real time (multicam). Music ducks by ``music.duck_db`` under voice-over
SFX (``audio/voice/``) and transcribed words of items playing at ≥ −12 dB.
"""

from __future__ import annotations

import math
import os
import re
from fractions import Fraction
from pathlib import Path

from opencine.core.cache import Cache, cache_key
from opencine.core.ffmpeg import probe_json, run
from opencine.core.project import Project
from opencine.core.time import Rate
from opencine.plan.media import Media, project_path
from opencine.plan.schema import ClipSource, EditItem, EditPlan, SceneSource, Source

SR = 48000
MICRO_FADE_S = Fraction(1, 100)
CUT_HANDLE_S = Fraction(1, 200)  # hard cut: each side overlaps the other by this much
TARGET_I, TARGET_TP = -14.0, -1.0
CEILING = TARGET_TP - 0.5  # limiter ceiling (and plain-gain limit), dBFS at 4× (margin for AAC)
TP_MARGIN = 0.2  # a post-encode trim aims this far under TARGET_TP
TOLERANCE_LU = 0.1
MAX_PASSES = 4
MAX_PUSH_DB = 12.0  # cap on one corrective gain step
DUCK_BLOCK = 480  # samples per volume step while ducking (10 ms)
DUCK_SPEECH_GAIN = -12.0  # source audio quieter than this is a bed, not speech to duck for
AUDIO_VERSION = 3
_FMT = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"

Span = tuple[Fraction, Fraction]


def _samples(frames: int | Fraction, rate: Rate) -> int:
    return round(Fraction(frames) / rate * SR)


def _sec(x: Fraction) -> str:
    return f"{float(x):.6f}"


def _atempo(speed: float) -> str:
    parts, s = [], speed
    while s > 2:
        parts.append("atempo=2")
        s /= 2
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1) > 1e-9:
        parts.append(f"atempo={s:.6f}")
    return "".join("," + p for p in parts)


def _file_id(p: Path) -> list:
    st = p.stat()
    return [str(p.resolve()), st.st_size, st.st_mtime_ns]


def _duration(path: Path) -> Fraction:
    return Fraction(str(probe_json(path)["format"].get("duration") or "0"))


def _is_voice(project: Project | None, path: Path) -> bool:
    if project is None:
        return False
    return (project.audio_dir / "voice").resolve() in path.resolve().parents


def _heard(it: EditItem, m: Media, snd: Media | None) -> tuple[Media, Source, int, int, float] | None:
    """What an item plays: ``(media, source, in, out, speed)`` — its own sound, or its
    ``sound`` (``snd``) in real time for the item's length; None when silent."""
    if it.gain is None:
        return None
    if it.sound is not None:
        if snd is None or snd.audio is None:
            return None
        return snd, it.sound.source, it.sound.in_, it.sound.in_ + it.length, 1.0
    if m.audio is None or it.ramp:  # a ramped item's own sound is muted
        return None
    return m, it.source, it.in_, it.out, it.speed


def speech_spans(plan: EditPlan, medias: list[Media], project: Project | None,
                 sounds: list[Media | None] | None = None) -> list[Span]:
    """Timeline seconds where someone speaks: voice-over SFX, and transcribed words inside the
    source range of items whose audio plays at ≥ ``DUCK_SPEECH_GAIN`` (mapped through their
    speed); an item without words in its range does not duck. ``sounds``: the media of each
    item's ``sound`` (``Resolver.sound``)."""
    rate = plan.fps
    sounds = sounds or [None] * len(plan.items)
    spans: list[Span] = []
    for s in plan.sfx:
        path = project_path(project, s.file)
        if _is_voice(project, path) and path.is_file():
            a = Fraction(s.at) / rate
            spans.append((a, a + _duration(path)))
    if project is None:
        return spans
    from opencine.index.model import load_index
    from opencine.speech.transcribe import load_transcripts

    trs = load_transcripts(project)
    if not trs:
        return spans
    index = load_index(project)
    for start, it, m, snd in zip(plan.starts(), plan.items, medias, sounds, strict=True):
        heard = _heard(it, m, snd)
        if heard is None or it.gain < DUCK_SPEECH_GAIN:
            continue
        m, src, f_in, f_out, speed = heard
        if isinstance(src, SceneSource):
            clip = index.scene(src.scene).clip
        elif isinstance(src, ClipSource):
            clip = src.clip
        else:
            continue
        t0, sp = Fraction(start) / rate, Fraction(str(speed))
        a_in = m.start_s + Fraction(f_in) / rate  # clip seconds at the picture's first frame
        a = a_in - Fraction(it.audio_lead) / rate * sp  # J/L-cut: the sound reaches further
        b = m.start_s + Fraction(f_out) / rate + Fraction(it.audio_tail) / rate * sp
        for seg in (trs.get(clip) or {}).get("segments") or []:
            for w in seg.get("words") or [seg]:
                if w.get("start") is None or w.get("end") is None:
                    continue
                ws, we = max(a, Fraction(str(w["start"]))), min(b, Fraction(str(w["end"])))
                if we > ws:
                    spans.append((t0 + (ws - a_in) / sp, t0 + (we - a_in) / sp))
    return spans


def _merge(spans: list[Span], gap: Fraction) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def duck_filter(spans: list[Span], duck_db: float, attack: Fraction, release: Fraction) -> str:
    """Music volume automation: ``duck_db`` inside each span, linear ramps (in dB) over
    ``attack`` before and ``release`` after, in 10 ms steps. "" when there is nothing to do."""
    if not spans or not duck_db:
        return ""
    traps = [f"clip(min((t-{_sec(a - attack)})/{_sec(attack)},({_sec(b + release)}-t)/{_sec(release)}),0,1)"
             for a, b in spans]
    g = traps[0]
    for tr in traps[1:]:
        g = f"max({g},{tr})"
    return f",asetnsamples=n={DUCK_BLOCK}:p=0,volume=volume='pow(10,{duck_db:.3f}*{g}/20)':eval=frame"


def mix_spec(plan: EditPlan, medias: list[Media], project: Project | None,
             sounds: list[Media | None] | None = None) -> tuple[list[str], str, dict]:
    """ffmpeg inputs, filter graph (output ``[mix]``) and a hashable spec for the timeline mix.
    ``sounds``: the media of each item's ``sound`` (``Resolver.sound``)."""
    rate = plan.fps
    total = _samples(plan.total_frames, rate)
    starts = plan.starts()
    inputs: list[str] = []
    chains: list[str] = []
    spec: list = [AUDIO_VERSION, str(rate)]
    handle = round(CUT_HANDLE_S * SR)

    def add_input(path: Path) -> int:
        inputs.extend(["-i", str(path)])
        return len(inputs) // 2 - 1

    # (file, speed, first sample, end sample) each item plays, None = silent
    plays: list[tuple[Path, float, int, int] | None] = []
    for it, m, snd in zip(plan.items, medias, sounds or [None] * len(plan.items), strict=True):
        heard = _heard(it, m, snd)
        if heard is None:
            plays.append(None)
            continue
        hm, _, f_in, f_out, speed = heard
        plays.append((hm.audio, speed, round((hm.start_s + Fraction(f_in) / rate) * SR),
                      round((hm.start_s + Fraction(f_out) / rate) * SR)))

    def seamless(a: int, b: int) -> bool:
        """Items ``a`` → ``b`` (a hard cut) play one file on without a gap: their crossfade is
        linear, so two copies of the same signal sum to it (equal power would bump +3 dB)."""
        pa, pb = plays[a], plays[b]
        return (plan.items[b].is_cut and pa is not None and pb is not None and pa[0] == pb[0]
                and pa[1] == pb[1] == 1 and pa[3] == pb[2])

    for i, (it, play) in enumerate(zip(plan.items, plays, strict=True)):
        if play is None:
            continue
        audio, speed, s0, s1 = play
        nxt = plan.items[i + 1] if i + 1 < len(plan.items) else None
        sp = Fraction(str(speed))
        delay = _samples(starts[i], rate)
        curve_in = "tri" if i and seamless(i - 1, i) else "qsin"
        curve_out = "tri" if nxt is not None and seamless(i, i + 1) else "qsin"
        # J/L-cut: the sound leads the picture in / trails it out (as far as timeline and media allow)
        lead = min(_samples(it.audio_lead, rate), delay, math.floor(s0 / sp)) if it.audio_lead else 0
        tail = _samples(it.audio_tail, rate) if it.audio_tail else 0
        # hard cuts: reach `handle` past the cut on both sides (source permitting) → crossfade
        pre = handle if i and it.is_cut and not lead and delay >= handle and s0 >= round(handle * sp) else 0
        post = handle if nxt is not None and nxt.is_cut and not tail else 0
        pre, post = pre or lead, post or tail
        s0, s1 = s0 - round(pre * sp), s1 + round(post * sp)
        n = _samples(it.length, rate) + pre + post
        dur = Fraction(n, SR)
        xfade = Fraction(2 * handle, SR)
        if lead:
            fade_in = f"afade=t=in:d={_sec(Fraction(lead, SR))}:curve=qsin"
        elif i and not it.is_cut:
            fade_in = f"afade=t=in:d={_sec(Fraction(it.transition_in.frames) / rate)}"
        elif pre:
            fade_in = f"afade=t=in:d={_sec(xfade)}:curve={curve_in}"
        else:
            fade_in = f"afade=t=in:d={_sec(MICRO_FADE_S)}"
        if tail:
            fade_out = f"afade=t=out:st={_sec(dur - Fraction(tail, SR))}:d={_sec(Fraction(tail, SR))}:curve=qsin"
        elif nxt is not None and not nxt.is_cut:
            fo = Fraction(nxt.transition_in.frames) / rate
            fade_out = f"afade=t=out:st={_sec(dur - fo)}:d={_sec(fo)}"
        elif post:
            fade_out = f"afade=t=out:st={_sec(dur - xfade)}:d={_sec(xfade)}:curve={curve_out}"
        else:
            fade_out = f"afade=t=out:st={_sec(dur - MICRO_FADE_S)}:d={_sec(MICRO_FADE_S)}"
        k = add_input(audio)
        chain = (f"[{k}:a]{_FMT},atrim=start_sample={s0}:end_sample={s1},asetpts=PTS-STARTPTS"
                 f"{_atempo(speed)},apad,atrim=end_sample={n},volume={it.gain}dB,"
                 f"{fade_in},{fade_out}")
        if delay - pre:
            chain += f",adelay=delays={delay - pre}S:all=1"
        chains.append(chain + f"[a{k}]")
        spec.append(["item", _file_id(audio), s0, s1, n, speed, it.gain, fade_in, fade_out,
                     delay - pre])

    if plan.music is not None:
        mu = plan.music
        path = project_path(project, mu.file)
        k = add_input(path)
        tot = Fraction(total, SR)
        fade = ""
        if mu.fade_in:
            fade += f",afade=t=in:d={_sec(Fraction(mu.fade_in) / rate)}"
        if mu.fade_out:
            fo = Fraction(mu.fade_out) / rate
            fade += f",afade=t=out:st={_sec(tot - fo)}:d={_sec(fo)}"
        attack = Fraction(max(1, mu.duck_attack)) / rate
        release = Fraction(max(1, mu.duck_release)) / rate
        spans = _merge(speech_spans(plan, medias, project, sounds), attack + release) if mu.duck_db else []
        duck = duck_filter(spans, mu.duck_db, attack, release)
        chains.append(f"[{k}:a]{_FMT},atrim=start_sample={_samples(mu.offset, rate)},"
                      f"asetpts=PTS-STARTPTS,apad,atrim=end_sample={total},volume={mu.gain}dB"
                      f"{fade}{duck}[a{k}]")
        spec.append(["music", _file_id(path), mu.model_dump(), [[str(a), str(b)] for a, b in spans]])

    for s in plan.sfx:
        path = project_path(project, s.file)
        k = add_input(path)
        delay = _samples(s.at, rate)
        chains.append(f"[{k}:a]{_FMT},volume={s.gain}dB"
                      + (f",adelay=delays={delay}S:all=1" if delay else "") + f"[a{k}]")
        spec.append(["sfx", _file_id(path), s.model_dump()])

    n_in = len(inputs) // 2
    if n_in == 0:
        inputs = ["-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo"]
        chains.append(f"[0:a]atrim=end_sample={total}[mix]")
    else:
        labels = "".join(f"[a{k}]" for k in range(n_in))
        chains.append(f"{labels}amix=inputs={n_in}:normalize=0:dropout_transition=0,"
                      f"apad,atrim=end_sample={total}[mix]")
    spec.append(total)
    return inputs, ";".join(chains), {"mix": spec}


def build_mix(plan: EditPlan, medias: list[Media], project: Project | None, cache: Cache,
              sounds: list[Media | None] | None = None) -> Path:
    """Render (or reuse) the loudness-normalised mix WAV."""
    inputs, graph, spec = mix_spec(plan, medias, project, sounds)
    key = cache_key("audio", spec, [TARGET_I, TARGET_TP, CEILING])
    wav = cache.path("audio", key, ".wav")
    if cache.hit("audio", key, ".wav") is None:
        mix = cache.path("audio", key, ".mix.wav")
        run(["-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[mix]",
             "-c:a", "pcm_f32le", "-ar", str(SR), str(mix)])
        tmp = wav.with_suffix(".tmp.wav")
        normalise(mix, tmp, _samples(plan.total_frames, plan.fps))
        os.replace(tmp, wav)
        mix.unlink()
    return wav


def measure(wav: Path) -> dict | None:
    """EBU R128 integrated loudness and true peak with ``ebur128`` (loudnorm's own pass-1
    numbers drift by a few tenths of an LU on short clips). None if (near) silent. Takes any
    file with audio (video is not decoded)."""
    proc = run(["-nostats", "-i", str(wav), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    summary = proc.stderr[proc.stderr.rfind("Summary:"):]
    num = r"(-?(?:[\d.]+|inf))"
    i = re.search(rf"I:\s+{num} LUFS", summary).group(1)
    tp = re.search(rf"Peak:\s+{num} dBFS", summary).group(1)
    if float(i) <= -70 or "inf" in tp:
        return None
    return {"i": float(i), "tp": float(tp)}


def gain_filter(gain_db: float, limit: bool, post_db: float = 0.0) -> str:
    """Plain gain, or gain into a lookahead limiter at 4× (≈ true peak, no added delay) followed
    by a linear ``post_db`` trim."""
    if not limit:
        return f"volume={gain_db:.3f}dB"
    return (f"aresample={4 * SR},volume={gain_db:.3f}dB,"
            f"alimiter=limit={10 ** (CEILING / 20):.5f}:attack=1:release=50:level=0:latency=1,"
            f"aresample={SR}" + (f",volume={post_db:.3f}dB" if post_db else ""))


def _apply(src: Path, dst: Path, af: str, samples: int) -> None:
    run(["-y", "-loglevel", "error", "-i", str(src), "-af", f"{af},apad,atrim=end_sample={samples}",
         "-c:a", "pcm_f32le", "-ar", str(SR), str(dst)])


def normalise(src: Path, dst: Path, samples: int) -> dict | None:
    """Pass 1 measures ``src``; pass 2 writes ``dst`` at −14 LUFS with the true peak ≤ −1 dBTP.
    Returns the final measurement (None = silent, copied as is)."""
    m = measure(src)
    if m is None:
        _apply(src, dst, "anull", samples)
        return None
    gain = TARGET_I - m["i"]
    if m["tp"] + gain <= CEILING:  # same margin as the limiter: AAC adds a little peak
        _apply(src, dst, gain_filter(gain, False), samples)
        return {"i": TARGET_I, "tp": m["tp"] + gain}
    # The limiter eats loudness, so 1 dB more gain buys < 1 LU (and peak-dominated material
    # barely moves): push with a secant step until loud enough, then trim down linearly —
    # lowering is exact and only lowers the peak.
    prev: tuple[float, float] | None = None  # (gain, loudness) of the last pass
    for _ in range(MAX_PASSES):
        _apply(src, dst, gain_filter(gain, True), samples)
        out = measure(dst)
        if out is None or abs(out["i"] - TARGET_I) <= TOLERANCE_LU:
            return out
        if out["i"] > TARGET_I:
            _apply(src, dst, gain_filter(gain, True, TARGET_I - out["i"]), samples)
            return measure(dst)
        slope = 1.0 if prev is None else (out["i"] - prev[1]) / (gain - prev[0])
        prev = (gain, out["i"])
        gain += min(MAX_PUSH_DB, (TARGET_I - out["i"]) / min(1.0, max(0.25, slope)))
    return out


def peak_trim(encoded: Path) -> float:
    """dB (≤ 0) to lower the mix by so the *encoded* audio's true peak is under ``TARGET_TP``."""
    m = measure(encoded)
    if m is None or m["tp"] <= TARGET_TP:
        return 0.0
    return TARGET_TP - TP_MARGIN - m["tp"]


def trimmed(wav: Path, db: float) -> Path:
    """``wav`` lowered by ``db`` (cached next to it)."""
    out = wav.with_name(f"{wav.stem}.trim{round(-db * 100)}.wav")
    if not out.exists():
        tmp = out.with_suffix(".tmp.wav")
        run(["-y", "-loglevel", "error", "-i", str(wav), "-af", f"volume={db:.3f}dB",
             "-c:a", "pcm_f32le", "-ar", str(SR), str(tmp)])
        os.replace(tmp, out)
    return out
