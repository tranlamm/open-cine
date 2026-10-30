"""A thing's mask through a run of frames: EdgeTAM (SAM 2 with a light memory, Apache-2.0).

The ONNX export has the four networks; the memory bank between them is assembled here as in
transformers' ``EdgeTamVideoModel``: the prompted first frame (plus the learned no-memory
embedding) gives the conditioning memory (temporal slot -1); each later frame attends to it,
the ``MEMORIES`` latest frame memories (offset ``k`` → temporal encoding ``k - 1``) and up to
``POINTERS`` object pointers (conditioning frame's, then the latest, newest first; each 256-d
pointer = four 64-d tokens), then decodes unprompted, keeping the mask with the best predicted
IoU. Object score ≤ 0 = not visible (empty mask). Frames are squashed to 1024²; ~0.8 s a frame
on the CPU, ~0.2 s on a small GPU.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from opencine.vision import models, runtime

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MEMORIES = 6  # num_maskmem - 1 frame memories besides the conditioning one
POINTERS = 16  # max_object_pointers_in_encoder
NETS = ("vision_encoder", "mask_decoder", "memory_attention", "memory_encoder")


def bank(cond: tuple[np.ndarray, np.ndarray], recent: list[tuple[np.ndarray, np.ndarray]],
         pointers: list[np.ndarray], tpos: np.ndarray) -> dict[str, np.ndarray]:
    """Memory attention inputs. ``cond``: the conditioning memory ``(tokens, pos)``;
    ``recent``: frame memories oldest first, the last one from the previous frame; ``pointers``:
    the conditioning frame's then the frames' object pointers, newest first; ``tpos``: the
    learned temporal encodings ``(num_maskmem, 1, 1, 64)``."""
    recent = recent[-MEMORIES:]
    mem = [cond[0]] + [m for m, _ in recent]
    pos = [cond[1] + tpos[-1, 0]] + [p + tpos[len(recent) - 1 - j, 0] for j, (_, p) in enumerate(recent)]
    ptr = np.stack(pointers[:POINTERS]).reshape(-1, 1, 64).astype(np.float32)
    return {"spatial_memory": np.concatenate(mem)[:, None].astype(np.float32),
            "spatial_memory_position_embeddings": np.concatenate(pos)[:, None].astype(np.float32),
            "pointer_memory": ptr, "pointer_memory_position_embeddings": np.zeros_like(ptr)}


class Tam:
    def __init__(self, sessions: dict[str, Any], extras: dict[str, np.ndarray]):
        self.s = sessions
        self.no_mem = extras["no_memory_embedding"].reshape(1, -1, 1, 1).astype(np.float32)
        self.tpos = extras["memory_temporal_positional_encoding"].astype(np.float32)

    @classmethod
    def load(cls) -> Tam | None:
        d = models.fetch("edgetam")
        if d is None:
            return None
        with np.load(d / "extras.npz") as z:
            extras = {k: z[k] for k in z.files}
        return cls({n: runtime.session(d / f"{n}.onnx") for n in NETS}, extras)

    def _run(self, net: str, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        s = self.s[net]
        return dict(zip([o.name for o in s.get_outputs()], s.run(None, feed), strict=True))

    def _see(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        x = (cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - MEAN) / STD
        return self._run("vision_encoder", {"pixel_values": x.transpose(2, 0, 1)[None]})

    def _decode(self, v: dict[str, np.ndarray], feats: np.ndarray, points: np.ndarray,
                labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        o = self._run("mask_decoder", {"feats0": v["feats0"], "feats1": v["feats1"], "feats2": feats,
                                       "input_points": points[None, None].astype(np.float32),
                                       "input_labels": labels[None, None].astype(np.int32)})
        iou = o["iou_scores"].reshape(-1)
        masks = o["high_res_masks"].reshape(-1, *o["high_res_masks"].shape[-2:])
        k = int(np.argmax(iou)) if len(masks) == len(iou) else 0
        ptr = o["object_pointer"].reshape(-1, 256)
        return masks[k], ptr[min(k, len(ptr) - 1)], float(o["object_score_logits"].reshape(-1)[0])

    def _memory(self, v: dict[str, np.ndarray], logits: np.ndarray, prompted: bool) -> tuple[np.ndarray, np.ndarray]:
        if logits.shape != (SIZE, SIZE):
            logits = cv2.resize(logits, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        o = self._run("memory_encoder", {"vision_features": v["feats2"],
                                         "pred_masks_high_res": logits[None, None].astype(np.float32),
                                         "is_mask_from_pts": np.array([prompted])})
        return o["memory_tokens"][0], o["memory_pos_enc"][0]

    def propagate(self, frames: list[np.ndarray] | np.ndarray, box: tuple[float, float, float, float] | None = None,
                  point: tuple[float, float] | None = None, soft: bool = False) -> list[tuple[np.ndarray, float]]:
        """``(mask, object score)`` per frame of the thing at ``box`` (x0, y0, x1, y1) or
        ``point`` (x, y) — frame fractions — on the first frame. ``soft``: the mask is how sure
        the network is, 0–255 (the sigmoid of its logits), instead of a thresholded bool."""
        if not len(frames):
            return []
        h, w = frames[0].shape[:2]
        if box is not None:
            pts, labels = np.array([[box[0], box[1]], [box[2], box[3]]]) * SIZE, np.array([2, 3])
        elif point is not None:
            pts, labels = np.array([point]) * SIZE, np.array([1])
        else:
            raise ValueError("propagate needs a box or a point")
        v = self._see(frames[0])
        logits, ptr, score = self._decode(v, v["feats2"] + self.no_mem, pts, labels)
        cond, cond_ptr = self._memory(v, logits, True), ptr
        recent: list[tuple[np.ndarray, np.ndarray]] = []
        ptrs: list[np.ndarray] = []  # newest first
        out = [(self._mask(logits, w, h, soft), score)]
        blank_pts, blank_labels = np.zeros((1, 2)), -np.ones(1)
        for f in frames[1:]:
            v = self._see(f)
            cur = v["feats2"].reshape(256, -1).T[:, None]
            cur_pos = v["pos2"].reshape(256, -1).T[:, None]
            cond_feats = self._run("memory_attention", {
                "current_vision_features": cur, "current_vision_position_embeddings": cur_pos,
                **bank(cond, recent, [cond_ptr, *ptrs], self.tpos)})["conditioned_features"]
            feats = cond_feats[0, 0].T.reshape(1, 256, *v["feats2"].shape[-2:])
            logits, ptr, score = self._decode(v, feats, blank_pts, blank_labels)
            recent = [*recent[-(MEMORIES - 1):], self._memory(v, logits, False)]
            ptrs = [ptr, *ptrs[:POINTERS - 2]]
            out.append((self._mask(logits, w, h, soft), score))
        return out

    @staticmethod
    def _mask(logits: np.ndarray, w: int, h: int, soft: bool = False) -> np.ndarray:
        up = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
        if not soft:
            return up > 0
        return np.rint(255 / (1 + np.exp(-np.clip(up, -30, 30)))).astype(np.uint8)
