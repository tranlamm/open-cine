"""What a new take changes from the slot's last one (one change per take).

The request and prompt are compared with the slot's latest sidecar and sorted onto axes: ``frame``,
``model``, ``length``, and per prompt part ``camera``/``light``/``wardrobe``/``sound``/``action``. A new
seed alone is no change. Two or more axes in one take get a hint (nobody can tell which one helped).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from opencine.gen.lint import sections, shots

FRAME_KEYS = ("first", "last", "refs", "source", "crop")
MODEL_KEYS = ("model", "canvas", "mode", "subject", "control")
LENGTH_KEYS = ("seconds", "aspect")
CAMERA = {
    "camera", "push", "pushes", "pushing", "pull", "pulls", "pulling", "zoom", "zooms", "pan", "pans",
    "panning", "tilt", "tilts", "tilting", "truck", "trucks", "pedestal", "pedestals", "crane", "arc",
    "arcs", "orbit", "orbits", "track", "tracks", "tracking", "follow", "follows", "following",
    "dolly", "static", "handheld", "shake", "shakes", "shaking", "amplitude", "speed", "close-up",
    "closeup", "wide", "medium", "aerial", "overhead", "low-angle", "high-angle", "angle", "pov",
    "lens", "shot", "framing", "frames", "framed", "focus", "35mm", "50mm", "85mm", "24mm",
}
LIGHT = {
    "light", "lights", "lit", "lighting", "backlight", "backlit", "sunlight", "sun", "sunset",
    "sunrise", "dawn", "dusk", "golden", "glow", "glows", "glowing", "shadow", "shadows", "neon",
    "lamp", "lamps", "candle", "candlelight", "moonlight", "moon", "rim", "flare", "flares", "haze",
    # not "dark" / "bright": they describe skin, hair and clothes as often as light
    "fog", "mist", "dim", "darkness", "palette", "hue", "contrast", "silhouette",
    "silhouetted", "overcast", "daylight", "twilight", "highlights", "glare", "grade", "grain",
}
WARDROBE = {
    "coat", "jacket", "shirt", "dress", "skirt", "trousers", "pants", "jeans", "scarf", "hat", "hood",
    "hoodie", "cap", "boots", "shoes", "sneakers", "gloves", "goggles", "glasses", "helmet",
    "uniform", "suit", "tie", "sweater", "cardigan", "vest", "apron", "cloak", "robe", "armor",
    "armour", "necklace", "earrings", "belt", "backpack", "wears", "wearing", "outfit", "costume",
    "sleeves", "collar", "hair", "hairstyle",
}
_VOCABS = (("camera", CAMERA), ("light", LIGHT), ("wardrobe", WARDROBE))
SOUND_PARTS = ("overall_soundscape", "non_diegetic_music")
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")
_MARK = re.compile(r"^\s*At \d{1,2}:\d{2}(?:\.\d+)?,?\s*")


def request_of(slot: Any) -> dict[str, Any]:
    """What a slot asks for besides its prompt (kept in the sidecar as ``extra.request``)."""
    return slot.model_dump(mode="json", exclude={"id", "prompt", "variants"}, exclude_none=True)


def parts(prompt: str) -> dict[str, str]:
    """A prompt cut into the parts a take changes one by one: ``Shot N``, the style sentence before
    [Shot 1], each other field; a prompt with no fields is one part ``prompt``."""
    secs = sections(prompt)
    if len(secs) == 1:
        return {"prompt": secs[""]}
    out = {k: v for k, v in secs.items() if k and k not in ("integrated_multimodal_description",
                                                            "detailed_description")}
    desc = secs.get("detailed_description", secs.get("integrated_multimodal_description", ""))
    style, shot_list = shots(desc)
    if style:
        out["style"] = style
    for n, body in shot_list:
        out[f"Shot {n}"] = _MARK.sub("", body, count=1)
    return out


def _axes_of(old: str, new: str) -> tuple[set[str], int, int]:
    """Axes of the words that changed between two texts, ``+`` and ``-`` word counts."""
    a, b = _WORD.findall(old.lower()), _WORD.findall(new.lower())
    axes, plus, minus = set(), 0, 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        changed = a[i1:i2] + b[j1:j2]
        plus, minus = plus + (j2 - j1), minus + (i2 - i1)
        near = a[max(0, i1 - 1):i1] + a[i2:i2 + 1] + b[j2:j2 + 1]  # "rust-red" → "blue" (scarf)
        hit = {name for name, vocab in _VOCABS if any(w in vocab for w in changed)}
        axes |= hit or {name for name, vocab in _VOCABS if any(w in vocab for w in near)} or {"action"}
    return axes, plus, minus


def compare(prev: dict[str, Any] | None, request: dict[str, Any], prompt: str,
            take: int) -> dict[str, Any]:
    """``{take, from?, axes, parts?, seed?}``: this request against the slot's last sidecar."""
    if not prev or not prev.get("path"):
        return {"take": take, "axes": []}
    old = (prev.get("extra") or {}).get("request")
    axes: list[str] = []
    if old is not None:
        for axis, keys in (("frame", FRAME_KEYS), ("model", MODEL_KEYS), ("length", LENGTH_KEYS)):
            if any(old.get(k) != request.get(k) for k in keys):
                axes.append(axis)
    elif (prev.get("extra") or {}).get("model_choice") not in (None, request.get("model")) \
            and request.get("model"):
        axes.append("model")
    was, now = parts(prev.get("prompt") or ""), parts(prompt)
    detail: dict[str, str] = {}
    prompt_axes: set[str] = set()
    for name in dict.fromkeys([*was, *now]):
        if was.get(name, "") == now.get(name, ""):
            continue
        found, plus, minus = _axes_of(was.get(name, ""), now.get(name, ""))
        if name in SOUND_PARTS:
            found = {"sound"}
        prompt_axes |= found
        detail[name] = f"+{plus}/-{minus} ({', '.join(sorted(found))})"
    axes += [a for a in ("camera", "light", "wardrobe", "action", "sound") if a in prompt_axes]
    out: dict[str, Any] = {"take": take, "from": prev["path"], "axes": axes}
    if detail:
        out["parts"] = detail
    if old is not None and "seed" in request and old.get("seed") != request.get("seed"):
        out["seed"] = "new"
    return out


def hint(changes: dict[str, Any] | None) -> str | None:
    """The hint for a take that changes two axes or more at once."""
    axes = (changes or {}).get("axes") or []
    if len(axes) < 2:
        return None
    return (f"take {changes['take']} changes {len(axes)} things at once ({', '.join(axes)}) from "
            f"{changes.get('from')}: change one per take to see what made it better "
            "(say 'keep … · change …' at the gate)")
