"""Text overlays and voice captions as an ASS script burned in with libass (``ass`` filter).

Item text: styles ``plain`` / ``label`` / ``title`` / ``time`` (+ ``box``), entrances ``fade`` /
``pop`` / ``type`` / ``rise`` / ``wipe`` / ``shake``, accented ``emphasis`` words, ``{time}`` /
``{date}`` placeholders; each card sits in a safe-area band (``auto`` is resolved first by
``render.textplace``). Voice captions (``opencine.plan.captions`` cues) are cut per segment:
``chunk`` (one event per word, same text so nothing moves), ``karaoke`` (``\\kf`` per word) or
``line``. Fonts ship in ``assets/fonts`` (OFL) and reach libass through ``fontsdir``, so every
machine draws the same letters: Montserrat for item text and sentence captions, Anton (narrow)
for chunk captions.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from opencine.plan.schema import TextOverlay

if TYPE_CHECKING:
    from opencine.plan.captions import Cue
    from opencine.plan.schema import Captions

FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
DISPLAY_FONT, CAPTION_FONT = "Montserrat", "Anton"
_ALIGN = {"bottom": 2, "center": 5, "top": 8}
_ALIGN_LEFT = {"bottom": 1, "center": 4, "top": 7}  # the ``time`` style hugs the left edge
# Safe zones common to Reels, TikTok and Shorts: top 15 %, bottom 35 %, sides 11 %; bottom-band
# captions keep to the middle 64 % (clear of the right-hand button column).
MARGIN_BOTTOM, MARGIN_TOP, MARGIN_SIDE = 0.35, 0.15, 0.11  # of height, height, width
CAPTION_SIDE = 0.18
# each style's bands, usual first: ``position: auto`` without tracking data takes the first, and
# ``render.textplace`` breaks near ties in this order
BAND_ORDER = {"title": ("center", "top", "bottom"), "label": ("top", "bottom", "center"),
              "time": ("top", "bottom", "center"), "plain": ("bottom", "top", "center")}
LANDSCAPE_MARGINS = (0.1, 0.08, 0.05)  # 16:9 (YouTube): the bottom tenth, title-safe sides


def margins(width: int, height: int) -> tuple[float, float, float]:
    """``(bottom, top, side)`` safe margins for the frame's orientation."""
    return LANDSCAPE_MARGINS if width > height else (MARGIN_BOTTOM, MARGIN_TOP, MARGIN_SIDE)


_ESCAPE = re.compile(r"\\(?=[Nnh])")
_PLACEHOLDER = re.compile(r"\{(time|date)\}")
STYLE_NAMES = {"plain": "Default", "label": "Label", "title": "Title", "time": "Time"}
FADE_MS = 120
TYPE_S, TYPE_SHARE, TYPE_STEPS = Fraction(3, 5), Fraction(2, 5), 30  # reveal in min(0.6 s, 40 %)
POP = r"\fscx70\fscy70\t(0,90,\fscx108\fscy108)\t(90,160,\fscx100\fscy100)"
RISE_MS, WIPE_MS = 180, 250
SHAKE = r"\t(0,40,\frz4)\t(40,80,\frz-4)\t(80,160,\frz0)"
HIDE = r"{\alpha&HFF&}"
WHITE = "&HFFFFFF&"
EMPHASIS_SCALE = 112
BOX_BACK = "&H50000000"  # a dark box, ~70 % opaque


def font_name() -> str:
    """The display font: the shipped one, else a common system font."""
    if (FONTS_DIR / "Montserrat-Bold.ttf").is_file():
        return DISPLAY_FONT
    return "Arial" if sys.platform == "win32" else "DejaVu Sans"


def caption_font() -> str:
    return CAPTION_FONT if (FONTS_DIR / "Anton-Regular.ttf").is_file() else font_name()


def mono_font() -> str:
    return "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"


def has_placeholder(content: str) -> bool:
    return bool(_PLACEHOLDER.search(content))


def fill(content: str, when: datetime | None) -> str:
    """``{time}`` → ``06:12`` and ``{date}`` → ``02/10`` from ``when``; without a time they are
    dropped (and the spaces around them tidied)."""
    if not has_placeholder(content):
        return content
    if when is None:
        return re.sub(r"[ \t]{2,}", " ", _PLACEHOLDER.sub("", content)).strip()
    return _PLACEHOLDER.sub(lambda m: when.strftime("%H:%M" if m.group(1) == "time" else "%d/%m"),
                            content)


def ass_colour(rgb: str) -> str:
    """``#RRGGBB`` → ASS ``&HBBGGRR&``."""
    h = rgb.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _ts(seconds: Fraction) -> str:
    # libass shows an event for frames with start <= t < end, and the ass filter passes the
    # frame time truncated to whole ms (3.7 s can arrive as 3699 ms). The last centisecond
    # strictly before the boundary keeps both ends exact for any rate below 100 fps.
    cs = max(0, math.ceil(seconds * 100) - 1)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _body(text: str) -> str:
    body = unicodedata.normalize("NFC", text).replace("{", "(").replace("}", ")")
    body = _ESCAPE.sub("\\\\⁠", body)  # a word joiner breaks literal \N \n \h
    return body.replace("\n", "\\N")


def _accent(text: str, colour: str) -> str:
    return (f"{{\\1c{colour}\\fscx{EMPHASIS_SCALE}\\fscy{EMPHASIS_SCALE}}}{text}"
            f"{{\\1c{WHITE}\\fscx100\\fscy100}}")


def _emphasized(content: str, phrases: list[str], colour: str) -> str:
    """``content`` as ASS text with ``phrases`` (case-insensitive, whole words) accented."""
    words = [p for p in phrases if p.strip()]
    if not words:
        return _body(content)
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(p.strip()) for p in sorted(words, key=len, reverse=True))
                     + r")(?!\w)", re.IGNORECASE)
    out, k = [], 0
    for m in pat.finditer(content):
        out += [_body(content[k:m.start()]), _accent(_body(m.group(0)), colour)]
        k = m.end()
    return "".join(out) + _body(content[k:])


def _style(name: str, face: str, size_px: int, primary: str, secondary: str, edge: str, back: str,
           bold: int, border: int, out: int, shade: int, align: int, side: int, vmargin: int) -> str:
    return (f"Style: {name},{face},{size_px},{primary},{secondary},{edge},{back},"
            f"{bold},0,0,0,100,100,0,0,{border},{out},{shade},{align},{side},{side},{vmargin},1")


def _styles(used: set[tuple[str, bool]], width: int, height: int, font: str) -> list[str]:
    outline, shadow = max(1, round(height / 320)), max(1, round(height / 640))
    bottom, _, side_share = margins(width, height)
    side, vm = round(width * side_share), round(height * bottom)
    pad = max(2, round(height / 110))  # box padding
    base = {  # name, face, size, edge colour, bold, border style, outline, shadow
        "plain": ("Default", font, 0.045, "&H00000000", 1, 1, outline, shadow),
        "label": ("Label", font, 0.04, "&H50101010", 1, 3, pad, 0),  # border 3: a box
        "title": ("Title", font, 0.075, "&H00000000", 1, 1, 2 * outline, 2 * shadow),
        "time": ("Time", mono_font(), 0.035, "&H00000000", 0, 1, outline, shadow),
    }
    lines = []
    for key in ("plain", "label", "title", "time"):
        for box in (False, True):
            if (key, box) not in used:
                continue
            name, face, size, edge, bold, border, out, shade = base[key]
            if box and key != "label":  # border 4: one box behind the whole event
                name, edge, border, out, shade = name + "Box", BOX_BACK, 4, pad, 0
            lines.append(_style(name, face, round(height * size), "&H00FFFFFF", "&H00FFFFFF", edge,
                                BOX_BACK if border == 4 else "&H80000000", bold, border, out, shade,
                                2, side, vm))
    return lines


def _style_name(t: TextOverlay) -> str:
    return STYLE_NAMES[t.style] + ("Box" if t.box and t.style != "label" else "")


def _anchor(align: int, position: str, width: int, height: int) -> tuple[int, int]:
    """Where libass puts an event's anchor for ``\\an<align>`` and the safe margins."""
    bottom, top, side = margins(width, height)
    x = round(width * side) if align in (1, 4, 7) else width // 2
    y = {"bottom": height - round(height * bottom), "top": round(height * top), "center": height // 2}[position]
    return x, y


def _entrance(anim: str, align: int, position: str, width: int, height: int) -> str:
    if anim == "fade":
        return f"\\fad({FADE_MS},{FADE_MS})"
    if anim == "pop":
        return POP
    x, y = _anchor(align, position, width, height)
    if anim == "rise":
        return f"\\move({x},{y + round(height * 0.03)},{x},{y},0,{RISE_MS})\\fad({FADE_MS},0)"
    if anim == "wipe":
        return f"\\clip(0,0,0,{height})\\t(0,{WIPE_MS},\\clip(0,0,{width},{height}))"
    if anim == "shake":
        return f"\\org({x},{y}){SHAKE}"
    return ""


# ---- voice captions ----------------------------------------------------------------
def caption_style(cap: Captions) -> tuple[str, float]:
    """``(font, size)`` of a captions mode."""
    if cap.mode == "chunk":
        return caption_font(), cap.size or 0.055
    return font_name(), cap.size or 0.045


def _caption_styles(cap: Captions, width: int, height: int) -> list[str]:
    face, size = caption_style(cap)
    outline, shadow = max(1, round(height / 240)), max(1, round(height / 640))
    bottom, top, side_share = margins(width, height)
    side = round(width * (CAPTION_SIDE if width < height and cap.position == "bottom" else side_share))
    vm = round(height * (bottom if cap.position == "bottom" else top))
    pad = max(2, round(height / 110))
    hl = "&H00" + ass_colour(cap.highlight)[2:-1] if cap.highlight else "&H00FFFFFF"
    # karaoke: \kf turns words from the secondary (white) to the primary (highlight) colour
    primary, secondary = (hl, "&H00FFFFFF") if cap.mode == "karaoke" else ("&H00FFFFFF", "&H00FFFFFF")
    edge, back, border, out, shade = (BOX_BACK, BOX_BACK, 4, pad, 0) if cap.box else \
        ("&H00000000", "&H80000000", 1, outline, shadow)
    return [_style("Caption", face, round(height * size), primary, secondary, edge, back, 1, border,
                   out, shade, _ALIGN[cap.position], side, vm)]


def _cs(seconds: Fraction) -> int:
    return max(1, round(seconds * 100))


def caption_events(cues: list[Cue], cap: Captions, t0: Fraction, t1: Fraction, width: int,
                   height: int) -> list[str]:
    """Dialogue lines for the part of ``cues`` inside ``[t0, t1)`` (timeline seconds), timed
    from ``t0``. A cue that began before ``t0`` comes back without its entrance."""
    from opencine.plan.captions import emphasis_mask

    upper = cap.case == "upper" or (cap.case is None and cap.mode == "chunk")
    hl = ass_colour(cap.highlight) if cap.highlight else None
    accent = hl or WHITE
    entrance = {"pop": POP, "fade": f"\\fad({FADE_MS},{FADE_MS})", "none": ""}[cap.anim]
    out: list[str] = []

    def word(text: str, active: bool, stressed: bool) -> str:
        t = _body(text.upper() if upper else text)
        if stressed:
            return _accent(t, accent)
        return f"{{\\1c{hl}}}{t}{{\\1c{WHITE}}}" if active and hl else t

    def emit(a: Fraction, b: Fraction, text: str, first: bool) -> None:
        a, b = max(a, t0), min(b, t1)
        if b <= a:
            return
        tags = entrance if first else ""
        out.append(f"Dialogue: 0,{_ts(a - t0)},{_ts(b - t0)},Caption,,0,0,0,,{{{tags}}}{text}")

    for cue in cues:
        if cue.end <= t0 or cue.start >= t1:
            continue
        fresh = cue.start >= t0  # its entrance is in this segment
        breaks = set(cue.lines)
        stress = emphasis_mask([w.text for w in cue.words], cap.emphasis)
        if cap.mode == "chunk":
            for i, w in enumerate(cue.words):
                a = cue.start if i == 0 else w.start
                b = cue.words[i + 1].start if i + 1 < len(cue.words) else cue.end
                text = " ".join(word(x.text, j == i, stress[j]) for j, x in enumerate(cue.words))
                emit(a, b, text, fresh and i == 0)
        elif cap.mode == "karaoke":
            parts, lead = [], max(Fraction(0), cue.words[0].start - max(cue.start, t0))
            if lead:
                parts.append(f"{{\\k{_cs(lead)}}}")
            for i, w in enumerate(cue.words):
                nxt = cue.words[i + 1].start if i + 1 < len(cue.words) else w.end
                sep = "\\N" if i + 1 in breaks else " "
                parts.append(f"{{\\kf{_cs(max(nxt - max(w.start, t0), Fraction(1, 100)))}}}"
                             + word(w.text, False, stress[i]) + (sep if i + 1 < len(cue.words) else ""))
            emit(cue.start, cue.end, "".join(parts), fresh)
        else:
            text = "".join(word(w.text, False, stress[i]) + ("\\N" if i + 1 in breaks else " ")
                           for i, w in enumerate(cue.words)).rstrip()
            emit(cue.start, cue.end, text, fresh)
    return out


# ---- the script ------------------------------------------------------------------------
def ass_script(events: list[tuple[Fraction, Fraction, TextOverlay]], width: int, height: int,
               font: str, captions: tuple[Captions, list[str]] | None = None,
               accent: str = "#FFD21F") -> str:
    """``events``: ``(start_s, end_s, overlay)`` relative to the segment start, placeholders
    already filled (``fill``); ``captions``: the plan's captions settings and this segment's
    caption Dialogue lines (``caption_events``)."""
    used = {(t.style, t.box and t.style != "label") for _, _, t in events}
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *_styles(used, width, height, font),
        *(_caption_styles(captions[0], width, height) if captions and captions[1] else []),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    bottom, top, _ = margins(width, height)
    colour = ass_colour(accent)
    for start, end, t in events:
        pos = BAND_ORDER[t.style][0] if t.position == "auto" else t.position  # render resolves auto first
        margin = round(height * (bottom if pos == "bottom" else top))
        align = (_ALIGN_LEFT if t.style == "time" else _ALIGN)[pos]
        tags = f"\\an{align}\\fs{round(t.font_size * height)}"
        if t.style == "title":
            tags += "\\b800"
        if t.anim != "type":
            tags += _entrance(t.anim, align, pos, width, height)
        head = f"0,{{}},{{}},{_style_name(t)},,0,0,{margin},,{{{{{tags}}}}}"
        if t.anim != "type" or len(t.content) < 2:
            lines.append("Dialogue: " + head.format(_ts(start), _ts(end))
                         + _emphasized(t.content, t.emphasis, colour))
            continue
        reveal = min(TYPE_S, (end - start) * TYPE_SHARE)
        n = len(t.content)
        k_max = min(n, TYPE_STEPS)
        cuts = sorted({math.ceil(n * (k + 1) / k_max) for k in range(k_max)})
        for k, c in enumerate(cuts):
            a = start + reveal * k / len(cuts)
            b = end if k == len(cuts) - 1 else start + reveal * (k + 1) / len(cuts)
            rest = t.content[c:]
            text = _body(t.content[:c]) + (HIDE + _body(rest) if rest else "")
            lines.append("Dialogue: " + head.format(_ts(a), _ts(b)) + text)
    if captions:
        lines += captions[1]
    return "\n".join(lines) + "\n"


def ass_filter(path: Path) -> str:
    """``ass`` filter with the paths escaped for the filtergraph (Windows drive colons); the
    shipped fonts come in through ``fontsdir``."""
    def esc(p: Path) -> str:
        s = p.resolve().as_posix()
        if "'" in s:
            raise ValueError(f"libass path must not contain a quote: {s}")
        return s.replace(":", "\\:")

    fonts = f":fontsdir='{esc(FONTS_DIR)}'" if FONTS_DIR.is_dir() else ""
    return f"ass=filename='{esc(path)}'{fonts}"
