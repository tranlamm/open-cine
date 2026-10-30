"""Plan schemas: ``Brief`` → ``ShotPlan`` (intent) → ``EditPlan`` (frame-exact, rendered).

``EditPlan`` conventions (also in the field descriptions, which reach the Director as JSON Schema):

- Every time is an **integer frame at the plan ``rate``**. Source ``in``/``out`` are a
  half-open range ``[in, out)`` in source time; timeline positions (``sfx.at``, text, crop/zoom
  keyframes) count output frames.
- Seconds are accepted instead (``in_s``/``out_s``, ``sfx.at_s``, ``text.start_s``/``end_s``,
  ``sound.in_s``, same origin as the frame field) and converted to the nearest frame on load, so a
  stored plan only holds frames. Giving both forms of one time is an error.
- ``in``/``out`` are relative to the scene start (``{"scene": "S03"}``), the clip start
  (``{"clip": "C01"}``) or the file start (``{"gen": ...}``). Stills ignore ``in`` and just
  hold for ``out - in`` frames.
- An item lasts ``round((out - in) / speed)`` output frames. A non-cut ``transition_in`` of
  ``frames`` overlaps that many frames with the end of the previous item, so the timeline is
  ``Σ length − Σ transition frames``.
- Crop boxes and zoom centres are fractions of the source frame (0–1), so they do not depend
  on proxy vs original resolution. The renderer fills the deliverable from the crop box
  ("cover"): a box of the wrong aspect is centre-trimmed, never stretched.
- File paths are relative to the project root or absolute.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from opencine.core.time import Rate, to_frame

Frame = Annotated[int, Field(ge=0)]
Unit = Annotated[float, Field(ge=0, le=1)]


def _seconds(what: str, origin: str) -> Any:
    """A seconds alternative to a frame field: in the JSON Schema, converted by ``EditPlan`` on
    load and never stored."""
    return Field(None, ge=0, exclude=True, description=(
        f"Seconds instead of `{what}` (same origin: {origin}); converted to the nearest frame at "
        f"the plan rate. Give `{what}` or this, not both."))

DELIVERABLES: dict[str, tuple[int, int]] = {"1080x1920": (1080, 1920), "720x1280": (720, 1280),
                                             "1920x1080": (1920, 1080)}
TransitionKind = Literal["cut", "dissolve", "flash", "dip", "whip", "whip_left", "zoom", "zoom_out",
                         "slide_left", "slide_right", "slide_up", "slide_down", "mask"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---- EditPlan -------------------------------------------------------------
class SceneSource(_Model):
    scene: str = Field(description="Scene id from the scene table, e.g. 'S03'. in/out are relative to the scene start; going past the scene (into the clip) is allowed with a warning.")


class ClipSource(_Model):
    clip: str = Field(description="Source clip id, e.g. 'C01'. in/out are relative to the clip start.")


class GenSource(_Model):
    gen: str = Field(description="Generated video file, e.g. 'gen/clips/x.mp4'.")


class ImageSource(_Model):
    image: str = Field(description="Still image file (Ken Burns 1.0→1.1 unless `zoom` is given).")


Source = SceneSource | ClipSource | GenSource | ImageSource


class ItemSound(_Model):
    """Footage sound an item plays instead of its own (multicam: a generated angle over the
    take's real sound)."""

    scene: str | None = Field(None, description="Scene id, e.g. 'S01'.")
    clip: str | None = Field(None, description="Clip id, e.g. 'C01' (instead of scene).")
    in_: Frame = Field(alias="in", description="Source frame where the sound starts, from the scene/clip start like an item's `in`. Or give in_s.")
    in_s: float | None = _seconds("in", "scene/clip start, like S01@4.2")

    @model_validator(mode="after")
    def _one_source(self) -> ItemSound:
        if (self.scene is None) == (self.clip is None):
            raise ValueError("sound: give scene or clip, not both")
        return self

    @property
    def source(self) -> SceneSource | ClipSource:
        return SceneSource(scene=self.scene) if self.scene is not None else ClipSource(clip=self.clip)


class CropBox(_Model):
    x: Unit = Field(description="Left edge, fraction of source width.")
    y: Unit = Field(description="Top edge, fraction of source height.")
    w: float = Field(gt=0, le=1, description="Width, fraction of source width.")
    h: float = Field(gt=0, le=1, description="Height, fraction of source height.")


class CropKey(CropBox):
    at: Frame = Field(description="Output frame within the item (0 = item start).")


class ZoomKey(_Model):
    at: Frame = Field(description="Output frame within the item (0 = item start).")
    scale: float = Field(ge=1, le=8, description="Punch-in factor applied inside the crop box (1 = none).")
    cx: Unit = Field(0.5, description="Zoom anchor, fraction of the crop box width.")
    cy: Unit = Field(0.5, description="Zoom anchor, fraction of the crop box height.")


class SpeedKey(_Model):
    at: Frame = Field(description="Source frame counted from the item's `in` (not output frames).")
    speed: float = Field(gt=0, le=8, description="Speed at that source frame (0.35 = slow motion).")


class ColorMatch(_Model):
    match: str = Field(description="Scene or moment whose colour this item is pulled toward, e.g. 'S03' or 'S03@2.1'.")
    strength: float = Field(0.8, ge=0, le=1)


LookName = Literal["clean", "warm", "cool", "film", "vivid", "mono"]


class Look(_Model):
    name: LookName = Field(description="clean (light contrast) | warm | cool | film (soft, lifted blacks, teal/orange) | vivid (saturated) | mono.")
    strength: float = Field(1.0, ge=0, le=1)


class Transition(_Model):
    kind: TransitionKind = Field("cut", description="cut | dissolve | flash (dip to white) | dip (to black) | whip (camera whips right: picture slides left, blurred) | whip_left (picture slides right) | zoom (zoom-in blend) | zoom_out (the incoming shot settles from 1.25× while blending) | slide_left/right/up/down (picture pushed that way, eased, no blur) | mask (something passing in front of the camera at the end of the previous shot reveals this one through its shape; L0 'tail: object covers frame → mask'; becomes a whip when too little passes).")
    frames: Frame = Field(0, description="Overlap with the previous item in frames; ignored for cut.")
    point: tuple[Unit, Unit] | None = Field(None, description="mask: where the passing object is on the previous shot's frame (x, y as 0–1 of the output); null = the largest moving thing.")


TextStyle = Literal["plain", "label", "title", "time"]
TEXT_SIZES: dict[str, float] = {"plain": 0.045, "label": 0.04, "title": 0.075, "time": 0.035}


class TextOverlay(_Model):
    content: str = Field(description="Text to burn in; '\\n' breaks lines. {time} / {date} become the moment's shooting time (06:12) / date (02/10) when its clip has one (L0 'shot').")
    position: Literal["auto", "top", "center", "bottom"] = Field("auto", description="auto = the band of the safe area (top, center or bottom) that hides least of the subject (a face, else what moves, else what stands out) and of busy picture while the text shows, never the captions band while captions run; near ties go to the style's usual band (title center, label/time top, plain bottom). top/center/bottom force a band (a craft hint says when it hides the subject).")
    style: TextStyle = Field("plain", description="plain (outlined) | label (on a dark box: step labels, product benefits) | title (big, bold) | time (small, left-aligned: a vlog clock).")
    anim: Literal["none", "fade", "pop", "type", "rise", "wipe", "shake"] = Field("none", description="Entrance: fade (in and out) | pop (scales in: hook, labels) | type (typewriter: vlog clock, 'POV:'; never a hook, it takes 0.6 s to read) | rise (slides up while fading in: step labels, benefits) | wipe (revealed left to right: title, CTA, price) | shake (one shake: a payoff or a shocking price, once a video).")
    emphasis: list[str] = Field([], description="Words or phrases of `content` (case-insensitive) drawn in the accent colour at 112 %: the benefit, the price, the feeling.")
    box: bool = Field(False, description="A dark box behind the text (label always has one).")
    size: float | None = Field(None, gt=0, le=0.3, description="Font size as a fraction of output height; null = the style's (plain 0.045, label 0.04, title 0.075, time 0.035).")
    start: Frame = Field(0, description="First output frame within the item.")
    end: int | None = Field(None, description="End output frame within the item (exclusive); null = item end.")
    start_s: float | None = _seconds("start", "item start")
    end_s: float | None = _seconds("end", "item start")

    @property
    def font_size(self) -> float:
        return self.size if self.size is not None else TEXT_SIZES[self.style]


class EditItem(_Model):
    source: Source = Field(description='One of {"scene": "S03"}, {"clip": "C01"}, {"gen": "gen/clips/x.mp4"}, {"image": "gen/stills/a.png"}.')
    in_: Frame = Field(alias="in", description="First source frame (inclusive), relative to the source start. Or give in_s.")
    out: Frame = Field(description="Last source frame (exclusive). Or give out_s.")
    in_s: float | None = _seconds("in", "scene/clip/file start, like S03@2.5")
    out_s: float | None = _seconds("out", "scene/clip/file start; exclusive")
    speed: float = Field(1.0, gt=0, le=8, description="Constant speed; 2 = twice as fast, 0.5 = slow motion.")
    ramp: list[SpeedKey] | None = Field(None, description="Speed ramp instead of `speed`: keys at source frames from `in`, linear between keys, held outside; e.g. [{at:0,speed:1},{at:12,speed:0.35},{at:30,speed:0.35},{at:40,speed:1}] slows around source frames 12–30. The item lasts ∫ 1/speed over in→out. Its own sound is muted while ramped (use music/sfx; a `sound` still plays).")
    crop: CropBox | list[CropKey] | None = Field(None, description="Static crop box, or keyframes interpolated linearly (held before the first / after the last). null = centred 9:16 cover.")
    layout: Literal["fill", "fit_blur"] = Field("fill", description="fill = the crop covers the frame; fit_blur = a 9:16 picture in the middle of a landscape frame over a blurred copy of itself (16:9 versions, set by the 16:9 render).")
    zoom: list[ZoomKey] | None = Field(None, description="Zoom keyframes, linear between keys.")
    transition_in: Transition = Field(default_factory=Transition, description="Transition from the previous item.")
    text: TextOverlay | None = None
    color: Literal["auto", "none"] | ColorMatch = Field("auto", description="auto = nudge this item's colour toward the rest of the footage when it is a little off (white balance, exposure; not a deliberate change of light); none = leave it; {match: 'S03'} = pull it toward that scene.")
    gain: float | None = Field(0.0, description="Source audio gain in dB; null mutes it. A generated clip ({\"gen\": …}) is muted unless gain or sound is given.")
    sound: ItemSound | None = Field(None, description="Play this footage's sound instead of the item's own: {\"scene\": \"S01\", \"in_s\": 4.2} plays S01 from 4.2 s in real time for the item's whole length, whatever its speed or ramp (multicam: a generated angle of that moment over the take's real sound; consecutive items continue the take when each starts where the last ended). gain, audio_lead and audio_tail apply to it.")
    audio_lead: Frame = Field(0, description="J-cut: this item's sound starts this many frames before its picture, under the end of the previous item (fades in).")
    audio_tail: Frame = Field(0, description="L-cut: this item's sound runs on this many frames after its picture ends, under the next item (fades out).")
    note: str | None = Field(None, description="Free-form note (ignored by the renderer).")
    slot: int | None = Field(None, description="ShotPlan slot this item plays (plan_solve sets it). Keep it when you edit the item: a later plan_solve keeps hand-edited and locked slots as they are.")

    @model_validator(mode="before")
    @classmethod
    def _mute_generated(cls, data: Any) -> Any:
        """A generated clip's soundtrack is invented: mute it unless ``gain`` is given (an
        explicit 0 keeps it) or it plays a footage ``sound``. Dumps always carry ``gain``, so a
        stored plan reloads the same."""
        gen = isinstance(data, dict) and isinstance(data.get("source"), dict) and "gen" in data["source"]
        if gen and "gain" not in data and data.get("sound") is None:
            return {**data, "gain": None}
        return data

    @property
    def length(self) -> int:
        """Output length in frames: ``round((out - in) / speed)`` (a ramp: ``round(∫ 1/speed)``,
        ``opencine.plan.ramp``), at least 1."""
        if self.ramp:
            from opencine.plan.ramp import length

            return length(self.ramp, max(0, self.out - self.in_))
        return max(1, math.floor(Fraction(self.out - self.in_) / Fraction(str(self.speed)) + Fraction(1, 2)))

    @property
    def is_cut(self) -> bool:
        return self.transition_in.kind == "cut" or self.transition_in.frames == 0


class Music(_Model):
    file: str
    offset: Frame = Field(0, description="Frames into the music file where the timeline starts.")
    fade_in: Frame = Field(0, description="Fade-in frames at the timeline start.")
    fade_out: Frame = Field(0, description="Fade-out frames before the timeline end.")
    gain: float = Field(0.0, description="dB.")
    duck_db: float = Field(-10.0, ge=-40, le=0, description="Music gain change (dB) while someone speaks: voice-over sfx (files under audio/voice/) and transcribed speech in items whose audio plays (gain not null). 0 = no ducking.")
    duck_attack: Frame = Field(6, description="Frames the duck ramps down, ending where speech starts.")
    duck_release: Frame = Field(12, description="Frames the duck ramps back up after speech ends.")


class Captions(_Model):
    """Captions that follow the voice-over: each voice line's words (``audio/voice/<id>.words.json``,
    written by ``tts``) shown as they are said."""

    mode: Literal["chunk", "karaoke", "line"] = Field("chunk", description="chunk: 2–4 words at a time, big, the word being said in the highlight colour (narration that sells: product ads, talking vlogs) | karaoke: the whole sentence, words take the highlight colour as they are said (long narration) | line: plain subtitles, a sentence at a time (quiet voice, ASMR).")
    max_chars: int | None = Field(None, ge=6, le=60, description="Characters per chunk (chunk; default 15) or per line (karaoke/line; default 42, 2 lines at most).")
    case: Literal["upper", "as_is"] | None = Field(None, description="null = upper for chunk, as written otherwise.")
    highlight: str | None = Field("#FFD21F", pattern=r"^#[0-9A-Fa-f]{6}$", description="'#RRGGBB' of the word being said (chunk) or already said (karaoke); also the emphasis colour. null = no highlight.")
    emphasis: list[str] = Field([], description="Keywords (case-insensitive) always in the highlight colour at 112 %: benefits, prices, the payoff word. Keep to 5–15 % of the words.")
    position: Literal["bottom", "center", "top"] = Field("bottom", description="Same bands as item text; keep item text out of this band while captions run.")
    size: float | None = Field(None, gt=0, le=0.15, description="Font size as a fraction of output height; null = 0.055 (chunk) or 0.045.")
    box: bool = Field(False, description="A dark box behind the captions (busy or bright footage).")
    anim: Literal["pop", "fade", "none"] = Field("pop", description="Entrance of each chunk or sentence.")
    breaks: dict[str, str] = Field({}, description="Your own chunking of a voice line, '|' between chunks: {'v01': 'Squeeze the lemon | really fast | then pour'} (the words must be the line's).")


class Sfx(_Model):
    file: str
    at: Frame = Field(description="Timeline frame where the sound starts.")
    at_s: float | None = _seconds("at", "timeline start")
    gain: float = Field(0.0, description="dB.")


_ITEM_SECONDS = {"in_s": ("in", "in_"), "out_s": ("out",)}
_TEXT_SECONDS = {"start_s": ("start",), "end_s": ("end",)}
_SFX_SECONDS = {"at_s": ("at",)}
_SOUND_SECONDS = {"in_s": ("in", "in_")}


def _frames(obj: Any, rate: Rate, where: str, keys: dict[str, tuple[str, ...]]) -> Any:
    """Replace ``<x>_s`` seconds in ``obj`` by the frame field ``<x>`` (nearest frame)."""
    if not isinstance(obj, dict) or not any(k in obj for k in keys):
        return obj
    obj = dict(obj)
    for key, names in keys.items():
        if key not in obj:
            continue
        sec = obj.pop(key)
        if sec is None:
            continue
        if any(obj.get(n) is not None for n in names):
            raise ValueError(f"{where}: give {names[0]} (frames) or {key} (seconds), not both")
        if isinstance(sec, bool) or not isinstance(sec, int | float) or sec < 0:
            raise ValueError(f"{where}.{key} must be seconds >= 0, got {sec!r}")
        obj[names[0]] = to_frame(sec, rate, "nearest")
    return obj


class EditPlan(_Model):
    """Frame-exact edit, rendered by ``opencine.render.render``."""

    rate: str = Field("30", description="Plan frame rate, e.g. '30' or '30000/1001'; normally the project rate.")
    deliverable: Literal["1080x1920", "720x1280", "1920x1080"] = Field("1080x1920", description="Output size; 1920x1080 is made from an approved 9:16 plan by render(…, aspect='16:9'), or is a landscape film's own size (template film).")
    letterbox: float | None = Field(None, ge=1.0, le=4.0, description="Black bars (top and bottom, or the sides) that frame the picture at this aspect ratio, e.g. 2.39 (scope) or 1.85; drawn under the text. null = none.")
    items: list[EditItem]
    music: Music | None = None
    sfx: list[Sfx] = []
    captions: Captions | None = Field(None, description="Captions that follow the voice-over sfx (audio/voice/*); null = none.")
    look: Look | None = Field(None, description="One colour look over the whole video (plan_solve sets the template's); null = none.")

    @model_validator(mode="before")
    @classmethod
    def _seconds_to_frames(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        try:
            rate = Rate(data.get("rate", "30"))
        except (TypeError, ValueError, ZeroDivisionError):
            return data  # the rate field reports it
        data = dict(data)
        if isinstance(data.get("items"), list):
            items = []
            for i, it in enumerate(data["items"]):
                it = _frames(it, rate, f"items[{i}]", _ITEM_SECONDS)
                if isinstance(it, dict) and isinstance(it.get("text"), dict):
                    it["text"] = _frames(it["text"], rate, f"items[{i}].text", _TEXT_SECONDS)
                if isinstance(it, dict) and isinstance(it.get("sound"), dict):
                    it["sound"] = _frames(it["sound"], rate, f"items[{i}].sound", _SOUND_SECONDS)
                items.append(it)
            data["items"] = items
        if isinstance(data.get("sfx"), list):
            data["sfx"] = [_frames(s, rate, f"sfx[{k}]", _SFX_SECONDS) for k, s in enumerate(data["sfx"])]
        return data

    @field_validator("rate")
    @classmethod
    def _rate(cls, v: str) -> str:
        return str(Rate(v))

    @property
    def fps(self) -> Rate:
        return Rate(self.rate)

    @property
    def size(self) -> tuple[int, int]:
        return DELIVERABLES[self.deliverable]

    def starts(self) -> list[int]:
        """Timeline start frame of each item."""
        out, pos = [], 0
        for i, it in enumerate(self.items):
            if i and not it.is_cut:
                pos -= it.transition_in.frames
            out.append(pos)
            pos += it.length
        return out

    @property
    def total_frames(self) -> int:
        return max((s + it.length for s, it in zip(self.starts(), self.items, strict=True)), default=0)


# ---- ShotPlan / Brief -------------------------------------------------------
class FootageSource(_Model):
    scene: str
    at: float | None = Field(None, ge=0, description="Seconds into the scene of the moment to use (as in the ref S03@2.5); the storyboard shows this frame.")
    hint: str | None = Field(None, description="Which part of the scene, e.g. 'strongest squeeze'.")


class GenerateSource(_Model):
    ref: str | None = Field(None, description="Reference frame, e.g. 'S02@00:04.1'.")
    camera: str | None = Field(None, description="The new view: shot size, angle, lens, e.g. 'ECU macro, low angle'.")
    motion: str | None = Field(None, description="Subject action and camera move, e.g. 'knife comes down; slow push-in'.")
    prompt: str | None = None


class TransformSource(_Model):
    """A footage window remade by video-to-video (``gen_edit``): same motion, camera and timing."""

    scene: str = Field(description="Scene id, e.g. 'S03'.")
    in_s: float = Field(ge=0, description="Window start, seconds into the scene.")
    out_s: float = Field(gt=0, description="Window end (exclusive), seconds into the scene; ≤ 15 s after in_s.")
    mode: Literal["swap", "restyle"] = Field(description="swap = the person becomes another character (robot, mascot…); restyle = the whole picture takes a new look (oil painting, anime…).")
    look: str | None = Field(None, description="The new character or style in a few words, e.g. 'armored robot', 'Ghibli watercolour'.")


class SlotSource(_Model):
    footage: FootageSource | None = None
    generate: GenerateSource | None = None
    transform: TransformSource | None = Field(None, description="Footage remade with a new character or look (gen_edit); the item plays the footage's sound in step.")


Role = Literal["hook", "setup", "process", "build", "climax", "payoff", "loop", "outro"]
ShotSize = Literal["ECU", "CU", "MCU", "MS", "WS", "EWS"]
SlotTransition = Literal["cut", "dissolve", "flash", "dip", "whip", "whip_left", "whip_right", "zoom",
                         "zoom_out", "slide_left", "slide_right", "slide_up", "slide_down", "mask"]
SlotSpeed = Literal["normal", "fast", "slow", "ramp_slow_at_impact"]


class SlotMatch(_Model):
    kind: Literal["motion", "shape", "light", "color", "position", "sound"] = Field(
        description="The one element the cut into the next slot carries over (everything else changes).")
    note: str | None = Field(None, description="e.g. 'zip pulled left→right, ends at frame right edge'.")


class Slot(_Model):
    slot: int = Field(ge=1, description="Slot id, unique in the plan; generation uses it as the gen slot id (\"3\" → gen/clips/3~<seed>.mp4).")
    purpose: str
    dur_s: tuple[float, float] = Field(description="[min, max] duration in seconds.")
    source: SlotSource
    role: Role | None = Field(None, description="Place in the arc: hook (first 1–2 s) … payoff, loop (back to the opening).")
    shot: ShotSize | None = Field(None, description="Shot size; neighbours should differ (or change angle).")
    cut_on: Literal["action", "beat", "word", "free"] | None = Field(
        None, description="What the cut out of this slot lands on: mid-action, a music beat, a word boundary.")
    match: SlotMatch | None = Field(None, description="Match-ish cut into the next slot.")
    transition_out: SlotTransition | None = Field(None, description="cut | dissolve | flash (white) | dip (black) | whip_left/whip_right (the camera whips that way; whip = whip_right) | zoom | zoom_out | slide_left/right/up/down (picture pushed that way) | mask (an object passing in front of the camera at the end of this slot wipes to the next; L0 'tail: object covers frame → mask').")
    speed: SlotSpeed | None = Field(None, description="normal | fast (1.5–2×) | slow (0.5×) | ramp_slow_at_impact.")
    text: str | None = None
    sfx: str | None = Field(None, description="e.g. 'whoosh', 'pop', 'slice': a file sfx/<name>.wav|mp3|ogg, played at the cut out of this slot.")
    crop: CropBox | None = Field(None, description="Footage window to keep (fractions of the source frame), copied to the EditPlan item; null = centred 9:16 cover.")
    chapter: str | None = Field(None, description="Brief.story chapter this slot belongs to (vlog); slots go in chapter order.")
    reframe: Literal["track", "center"] | None = Field(None, description="Wide footage → 9:16: track = a crop path following the subject (face, else what moves); center = a fixed centred crop; null = track when the source is wider than 9:16 and a subject was found. `crop` wins over both.")


class ShotPlan(_Model):
    """Director intent; ``plan_solve`` turns it into a draft ``EditPlan``."""

    template: str | None = Field(None, description="process_asmr | day_in_life | product_ad | film: plan_solve's rhythm, sound levels, text styles, template sounds and output size (film: skill opencine-film); null = plain defaults.")
    music: str | None = Field(None, description="Track in music/ (music_add), e.g. 'x.mp3'.")
    deliverable: Literal["1080x1920", "720x1280", "1920x1080"] | None = Field(None, description="Output size of the draft EditPlan; null = the template's (film: 1920x1080, others: 1080x1920).")
    letterbox: float | None = Field(None, ge=1.0, le=4.0, description="EditPlan.letterbox for the draft, e.g. 2.39 for a scope film; null = none.")
    slots: list[Slot]


class Chapter(_Model):
    name: str = Field(description="Short name, e.g. 'morning', 'prep', 'run'.")
    clips: list[str] = Field([], description="Clip ids (C01…) this chapter draws from; empty = any.")
    note: str | None = None


class Brief(_Model):
    goal: str
    platform: Literal["tiktok", "reels", "shorts", "youtube", "any"] = "any"
    duration_s: tuple[float, float] = (15, 60)
    target_s: float | None = Field(None, gt=0, description="Target length in seconds (within duration_s).")
    hook: str | None = Field(None, description="What the first 1–2 s show and say, e.g. 'S02@3.2 juice splash + text \"This lemon is so juicy\"'.")
    hook_alternatives: list[str] = Field([], max_length=3, description="The other hook candidates considered (≤ 3).")
    arc: list[str] = Field([], description="Story beats in order, e.g. ['hook', 'setup: lemon + knife', 'squeeze (burst)', 'payoff: glass'].")
    story: list[Chapter] = Field([], description="Chapters in story order for multi-clip edits (vlog: morning → prep → run); L0 gives each clip's shooting time ('shot') and the shooting order. Slots name their chapter.")
    ending: Literal["payoff", "loop", "cta"] | None = Field(None, description="payoff held | loop back to the opening frame | call to action.")
    style: str | None = None
    audience: str | None = None
    language: str = "en"
    music: str | None = None
    must_include: list[str] = []
    avoid: list[str] = []
    notes: str | None = None


_KINDS: dict[str, type[BaseModel]] = {"brief": Brief, "shotplan": ShotPlan, "editplan": EditPlan}


def json_schema(kind: Literal["brief", "shotplan", "editplan"]) -> dict:
    """JSON Schema of a plan kind (by alias, so ``in`` not ``in_``)."""
    return _KINDS[kind].model_json_schema(by_alias=True)
