"""Where a named thing is in a frame: Florence-2-base-ft open-vocabulary detection.

``see(rgb)`` encodes the frame once (768² squashed; ~1.7 s CPU, ~0.35 s on a small GPU);
``find(seen, phrase)`` asks for one English phrase (greedy, ~0.4 s CPU) and returns its boxes,
falling back to phrase grounding. One phrase per call: extra phrases in a prompt are ignored.
Florence gives no confidence. The decoder runs on the CPU: its KV-cache branch decodes garbage
on DirectML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from opencine.vision import models, runtime

SIZE = 768
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BOS, EOS, DECODER_START = 0, 2, 2
NO_REPEAT = 3  # generation_config: no_repeat_ngram_size
MAX_TOKENS = 96
DETECT = "Locate {} in the image."  # <OPEN_VOCABULARY_DETECTION>
GROUND = "Locate the phrases in the caption: {}"  # <CAPTION_TO_PHRASE_GROUNDING>
GROUP_RE = re.compile(r"([^<]*)((?:<(?:loc_\d+|poly|/poly)>)+)")
LOC_RE = re.compile(r"<loc_(\d+)>")


@dataclass(frozen=True)
class Found:
    label: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as frame fractions

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def parse(text: str) -> list[Found]:
    """Boxes in Florence's answer: ``label<loc_x0><loc_y0><loc_x1><loc_y1>…`` (several boxes may
    follow one label) or ``label<poly><loc_x><loc_y>…</poly>`` (its bounding box). Bins are
    thousandths of the frame."""
    out = []
    for label, locs in GROUP_RE.findall(text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")):
        nums = [(int(v) + 0.5) / 1000 for v in LOC_RE.findall(locs)]
        name = label.strip()
        if "<poly>" in locs:
            for poly in re.findall(r"<poly>(.*?)</poly>", locs) or [locs]:
                p = [(int(v) + 0.5) / 1000 for v in LOC_RE.findall(poly)]
                if len(p) >= 6:
                    xs, ys = p[0::2], p[1::2]
                    out.append(Found(name, (min(xs), min(ys), max(xs), max(ys))))
            continue
        for k in range(0, len(nums) - 3, 4):
            x0, y0, x1, y1 = nums[k:k + 4]
            if x1 > x0 and y1 > y0:
                out.append(Found(name, (x0, y0, x1, y1)))
    return out


class Florence:
    def __init__(self, sessions: dict[str, Any], tokenizer: Any):
        self.s, self.tok = sessions, tokenizer
        dec = self.s["decoder"]
        self.past = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values")]
        shape = next(i.shape for i in dec.get_inputs() if i.name in self.past)
        self.heads = shape[1] if isinstance(shape[1], int) else 12
        self.head_dim = shape[3] if isinstance(shape[3], int) else 64
        self.outs = [o.name for o in dec.get_outputs()]

    @classmethod
    def load(cls) -> Florence | None:
        d = models.fetch("florence2")
        if d is None:
            return None
        from tokenizers import Tokenizer

        f = dict(zip(("tokenizer", "vision", "encoder", "decoder", "embed"), models.MODELS["florence2"]["files"],
                     strict=True))
        return cls({"vision": runtime.session(d / f["vision"]), "encoder": runtime.session(d / f["encoder"]),
                    "decoder": runtime.session(d / f["decoder"], gpu=False),
                    "embed": runtime.session(d / f["embed"], gpu=False)},
                   Tokenizer.from_file(str(d / f["tokenizer"])))

    def see(self, rgb: np.ndarray) -> np.ndarray:
        """Image features of an RGB frame (any size)."""
        x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        v = self.s["vision"]
        return v.run(None, {v.get_inputs()[0].name: x})[0]

    def find(self, seen: np.ndarray, phrase: str) -> list[Found]:
        """Boxes of ``phrase`` (English, a few words) in the frame ``seen`` came from."""
        for prompt in (DETECT, GROUND):
            found = parse(self._generate(seen, prompt.format(phrase.strip())))
            if found:
                return found
        return []

    def _embed(self, ids: list[int]) -> np.ndarray:
        e = self.s["embed"]
        return e.run(None, {e.get_inputs()[0].name: np.array([ids], np.int64)})[0]

    def _generate(self, seen: np.ndarray, prompt: str) -> str:
        x = np.concatenate([seen, self._embed(self.tok.encode(prompt).ids)], axis=1)
        mask = np.ones(x.shape[:2], np.int64)
        enc = self.s["encoder"]
        hidden = enc.run(None, {i.name: (x if "embeds" in i.name else mask) for i in enc.get_inputs()})[0]
        dec = self.s["decoder"]
        names = {i.name for i in dec.get_inputs()}
        past = {n: np.zeros((1, self.heads, 0, self.head_dim), np.float32) for n in self.past}
        out = [DECODER_START]
        for step in range(MAX_TOKENS):
            feed = {"inputs_embeds": self._embed([out[-1]]), "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask, "use_cache_branch": np.array([step > 0]), **past}
            res = dict(zip(self.outs, dec.run(None, {k: v for k, v in feed.items() if k in names}), strict=True))
            logits = res["logits"][0, -1]
            for a in range(len(out) - NO_REPEAT + 1):  # no repeated 3-grams
                if out[a:a + NO_REPEAT - 1] == out[len(out) - NO_REPEAT + 1:]:
                    logits[out[a + NO_REPEAT - 1]] = -np.inf
            nxt = BOS if step == 0 else int(np.argmax(logits))
            out.append(nxt)
            for n in self.past:  # the encoder's keys/values come once, at step 0
                if ".decoder." in n or step == 0:
                    past[n] = res[n.replace("past_key_values", "present")]
            if nxt == EOS:
                break
        return self.tok.decode(out[1:], skip_special_tokens=False)
