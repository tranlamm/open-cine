"""Prompt checks before a generation: the machine-checkable rules of the film skill's prompting guide.

H3 clips: shot numbering, cut times and lengths, one camera move per shot, negative/hype/emotion
words, thin shots, Vietnamese dialogue, the film bible (``Brief.notes``) copied verbatim. Qwen
stills: ``<imageN>`` vs refs, negative phrasing, sizes in the prompt, thin text-to-image prompts.
Hints never block a call; the Director fixes the prompt or keeps it on purpose.
"""

from __future__ import annotations

import re

SHOT_MIN_S, SHOT_MAX_S = 2.5, 8.0  # the 8 s ceiling is empirical (no first-party source)
SHOT_WORDS_MIN = 50  # clearly thin; IR density is ~80–120 words a shot (unmeasured on Turbo)
REF_WORDS_MIN = 250  # Ref2VA detailed_description: 350–500 words
T2I_WORDS_MIN = 150  # Qwen text-to-image: 250–500 words
MAX_CHARS = 7000  # Context-IR input cap (MiniMax API)
NOTE_COVER = (0.35, 0.9)  # share of a film bible line's words found: below = absent, above = copied

FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
          "subject_definitions", "summary", "retention_analysis", "detailed_description")
_FIELD = re.compile(rf"(?m)^\s*({'|'.join(FIELDS)}):")
_SHOT = re.compile(r"\[Shot (\d+)\]")
_AT = re.compile(r"^\s*At (\d{1,2}):(\d{2}(?:\.\d+)?)")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b")
_DIALOGUE = re.compile(r"<d>.*?</d>", re.S)
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")

# camera moves by kind; "track" (following the subject) combines with one other move
CAMERA_MOVES = {
    "push": r"\bpush(?:es|ing)? in\b|\bdoll(?:y|ies|ying) in\b",
    "pull": r"\bpull(?:s|ing)? (?:out|back)\b|\bdoll(?:y|ies|ying) (?:out|back)\b",
    "zoom": r"\bzoom(?:s|ing)? (?:in|out)\b",
    "pan": r"\bpan(?:s|ning)?\b",
    "tilt": r"\btilt(?:s|ing)? (?:up|down)\b",
    "truck": r"\btruck(?:s|ing)? (?:left|right)\b",
    "pedestal": r"\bpedestal(?:s|ing)? (?:up|down)\b|\bcran(?:es|ing)\b",
    "arc": r"\barc(?:s|ing)?\b|\borbit(?:s|ing)?\b|\bcircl(?:es|ing)\b",
    "static": r"\bstatic\b|\blocked[- ]off\b",
    "track": r"\btrack(?:s|ing)\b|\bfollow(?:s|ing)\b",
}
_MOVES = {k: re.compile(v, re.I) for k, v in CAMERA_MOVES.items()}
_NEGATIVE = re.compile(r"\b(no|not|never|without|nothing|nobody|don't|doesn't|isn't|aren't|avoids?"
                       r"|avoiding)\b(?:\s+\w+)?", re.I)
_HYPE = re.compile(r"\b(epic|stunning|breathtaking|gorgeous|amazing|awesome|masterpiece|best quality"
                   r"|high quality|ultra[- ]?realistic|hyper[- ]?realistic|8k|4k|award[- ]winning"
                   r"|make it cinematic|very cinematic|beautiful(?:ly)?)\b", re.I)
_EMOTION = re.compile(r"\b(sad(?:ly|ness)?|angr(?:y|ily)|scared|afraid|frightened|terrified"
                      r"|happ(?:y|ily)|lonely|depressed|furious|heartbroken|emotional(?:ly)?"
                      r"|cold(?:ly)? stare|looks? coldly)\b", re.I)
_VIET = re.compile(r"<d>\s*\[(?:vietnamese|vi|ti" "\u1ebfng vi\u1ec7t" r")\]", re.I)
_SIZE = re.compile(r"\b(aspect ratio|\d{3,4} ?x ?\d{3,4}|16:9|9:16|4:3|1:1)\b", re.I)
_IMAGE = re.compile(r"<image ?(\d+)>", re.I)
_STOP = {"with", "from", "that", "this", "into", "over", "under", "their", "them", "they", "then",
         "while", "where", "which", "after", "before", "about", "around", "behind", "beside", "frame",
         "shot", "camera", "scene", "the", "and", "her", "his", "its", "she", "him"}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sections(prompt: str) -> dict[str, str]:
    """``{field: body}`` of a Context-IR / Ref2VA prompt (text before the first field: ``""``)."""
    parts = _FIELD.split(prompt)
    out = {"": parts[0].strip()}
    for name, body in zip(parts[1::2], parts[2::2], strict=False):
        out[name] = body.strip()
    return out


def shots(description: str) -> tuple[str, list[tuple[int, str]]]:
    """``(text before [Shot 1], [(n, text), …])`` of a description field."""
    parts = _SHOT.split(description)
    return parts[0].strip(), [(int(n), t.strip()) for n, t in zip(parts[1::2], parts[2::2], strict=False)]


def h3_seconds(seconds: float) -> float:
    """The clip length H3 makes for ``seconds``: the next 17n + 5 frames at 24 fps."""
    frames = max(1, round(seconds * 24))
    while frames % 17 != 5:
        frames += 1
    return frames / 24


def camera_moves(text: str) -> set[str]:
    """Kinds of camera move named in the sentences of ``text`` whose subject is the camera."""
    found: set[str] = set()
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if re.search(r"\b(camera|lens|shot)\b", sentence, re.I):
            found |= {k for k, rx in _MOVES.items() if rx.search(sentence)}
    return found


def _quote(matches: list[str], n: int = 3) -> str:
    seen = list(dict.fromkeys(m.strip().lower() for m in matches))
    return ", ".join(f"'{m}'" for m in seen[:n]) + ("…" if len(seen) > n else "")


def _notes(notes: str | None) -> list[tuple[str, str]]:
    """``[(label, text)]`` of the film bible: the ``Look:`` line and the ``C1 Name:`` lines."""
    out = []
    for line in (notes or "").splitlines():
        if m := re.match(r"^\s*(Look|C\d+\s+[^:]{1,40}):\s*(.+?)\s*$", line):
            out.append((m[1].strip(), m[2]))
    return out


def _drift(prompt: str, notes: str | None, *, characters: bool) -> list[str]:
    """Film bible lines that are in the prompt only in part (rewritten instead of copied);
    ``characters`` False checks the ``Look`` line only (identity comes from pictures)."""
    have, flat = set(_words(prompt)), " ".join(prompt.lower().split())
    out = []
    for label, text in _notes(notes):
        if not characters and label != "Look":
            continue
        if " ".join(text.lower().split()).rstrip(".") in flat:
            continue
        words = [w for w in dict.fromkeys(_words(text)) if len(w) > 3 and w not in _STOP]
        if not words:
            continue
        missing = [w for w in words if w not in have]
        cover = 1 - len(missing) / len(words)
        if NOTE_COVER[0] <= cover < NOTE_COVER[1]:
            out.append(f"{label}: the prompt rewrites the film bible line (missing {_quote(missing)}): "
                       "copy it verbatim so every shot gets the same description")
    return out


def video_hints(prompt: str, seconds: float, *, refs: int = 0, first: bool = False,
                notes: str | None = None) -> list[str]:
    """Hints for one H3 clip prompt of ``seconds`` (``refs`` pictures, a ``first`` frame)."""
    text = prompt.strip()
    out: list[str] = []
    if len(text) > MAX_CHARS:
        out.append(f"{len(text)} characters: H3 takes ≤ {MAX_CHARS}; cut detail that repeats")
    parts = sections(text)
    ref_mode = "subject_definitions" in parts
    desc = parts.get("detailed_description" if ref_mode else "integrated_multimodal_description")
    if desc is None:
        out.append("plain prompt: the conditioner rewrites it (+120 s quota) and may add cuts; write "
                   "Context-IR (integrated_multimodal_description: [Shot 1] …)")
        return out
    missing = [f for f in (("summary", "retention_analysis") if ref_mode else ())
               + ("overall_soundscape", "non_diegetic_music") if f not in parts]
    if missing:
        out.append(f"no {', '.join(missing)}: write every field (non_diegetic_music: N/A when the "
                   "film has its own music)")
    style, shot_list = shots(desc)
    if ref_mode:
        if not style:
            out.append("detailed_description: open with one or two style sentences before [Shot 1]")
        summary = parts.get("summary", "")
        if summary and not summary.startswith("["):
            out.append("summary: start with the task type, e.g. [reference generation]")
        elif first and "keyframe completion" not in summary:
            out.append("summary: a first frame is given, so the task is "
                       "[reference generation + keyframe completion]")
        if len(_words(desc)) < REF_WORDS_MIN:
            out.append(f"detailed_description has {len(_words(desc))} words: Ref2VA wants 350–500")
    if not shot_list:
        out.append("no [Shot 1]: start the description with [Shot 1]")
        return out
    if [n for n, _ in shot_list] != list(range(1, len(shot_list) + 1)):
        out.append(f"shots are numbered {[n for n, _ in shot_list]}: number them 1, 2, 3 … in order")
    total = h3_seconds(seconds)
    starts = [0.0]
    for n, body in shot_list[1:]:
        m = _AT.match(body)
        if not m:
            out.append(f"[Shot {n}] has no cut time: write '[Shot {n}] At 00:04.000, the camera "
                       "cuts to …'")
            return out
        starts.append(int(m[1]) * 60 + float(m[2]))
    if any(b <= a for a, b in zip(starts, starts[1:], strict=False)) or starts[-1] >= total:
        out.append(f"cut times {starts[1:]} must rise and stay inside the clip ({total:.2f} s)")
        return out
    thin = []
    for (n, body), start, end in zip(shot_list, starts, [*starts[1:], total], strict=True):
        length = end - start
        if length < SHOT_MIN_S:
            out.append(f"[Shot {n}] lasts {length:.1f} s: give a shot ≥ {SHOT_MIN_S:g} s or merge it")
        elif length > SHOT_MAX_S:
            out.append(f"[Shot {n}] is one {length:.1f} s take: past ~{SHOT_MAX_S:g} s the second half "
                       "often degrades; cut it into two [Shot N] or two clips")
        inner = _AT.sub("", body, count=1) if n > 1 else body
        moves = camera_moves(inner)
        core = moves - {"track"}
        if len(core) >= 2 or len(moves) >= 3:
            out.append(f"[Shot {n}] moves the camera {len(moves)} ways ({', '.join(sorted(moves))}): "
                       "one camera move a shot")
        if clock := _CLOCK.findall(inner):
            out.append(f"[Shot {n}] has a time inside the take ({clock[0]}): times only open a "
                       "[Shot N]; inside, write early / midway / in the final second")
        if len(_words(inner)) < SHOT_WORDS_MIN:
            thin.append(n)
    if thin and not ref_mode:
        out.append(f"thin shot(s) {thin}: ~80–120 words a shot — more visible and audible detail "
                   "of the same event")
    said = _DIALOGUE.sub(" ", " ".join([desc, parts.get("overall_soundscape", "")]))
    if neg := [m.group(0) for m in _NEGATIVE.finditer(said)]:
        out.append(f"negative phrasing ({_quote(neg)}): H3 has no negative prompt; describe what is "
                   "there instead")
    if hype := [m.group(0) for m in _HYPE.finditer(text)]:
        out.append(f"empty words ({_quote(hype)}): name the lens, light and colours instead")
    if feel := [m.group(0) for m in _EMOTION.finditer(_DIALOGUE.sub(" ", desc))]:
        out.append(f"emotion words ({_quote(feel)}): show it — posture, face, breath, light")
    if _VIET.search(text):
        out.append("Vietnamese dialogue: H3 cannot speak it; make the line with tts and keep the "
                   "lips closed on screen")
    out += _drift(text, notes, characters=not (refs or first))
    return out


def image_hints(prompt: str, refs: int, notes: str | None = None) -> list[str]:
    """Hints for one Qwen-Image prompt with ``refs`` reference images."""
    text = prompt.strip()
    out: list[str] = []
    cited = {int(n) for n in _IMAGE.findall(text)}
    if refs == 0 and cited:
        out.append("<imageN> without refs: text-to-image has no images to name; describe the picture")
    elif refs == 1 and cited - {1}:
        out.append("one ref: call it 'the image' (<image2>… do not exist)")
    elif refs >= 2:
        if extra := sorted(cited - set(range(1, refs + 1))):
            out.append(f"<image{extra[0]}> is named but only {refs} refs are given")
        if missing := [n for n in range(1, refs + 1) if n not in cited]:
            out.append(f"refs[{missing[0] - 1}] is never named: say what <image{missing[0]}> gives "
                       "(identity, place, light…)")
    if re.search(r"<Picture \d+>", text):
        out.append("<Picture N> is H3's naming; Qwen names refs <image1>, <image2>")
    if neg := [m.group(0) for m in _NEGATIVE.finditer(text)]:
        out.append(f"negative phrasing ({_quote(neg)}): describe what should be there")
    if size := [m.group(0) for m in _SIZE.finditer(text)] + [m.group(0) for m in _HYPE.finditer(text)]:
        out.append(f"sizes or quality tags in the prompt ({_quote(size)}): the engine sets the size; "
                   "describe the picture")
    if refs == 0 and len(_words(text)) < T2I_WORDS_MIN:
        out.append(f"text-to-image prompt of {len(_words(text))} words: describe the finished picture "
                   "in 250–500 words (places across the frame, one light sentence)")
    if refs == 0:
        out += _drift(text, notes, characters=True)
    return out
