"""Match cuts: which moment of one scene carries into which moment of another.

Two 2 fps samples score per kind (0–1): ``color`` (mean Lab, 1 − ΔE76 / 40), ``shape`` (thumbnail
detail correlation), ``light`` (4×4 brightness layout), ``motion`` (motion vectors, both moving),
``position`` (subject centre and size). Same-shot pairs and retakes are left out; ingest keeps the
top pairs in ``index/matches.json`` for L0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.index.model import SceneIndex

KINDS = ("color", "shape", "light", "motion", "position")
WEIGHT = {"color": 0.7, "shape": 1.0, "light": 0.8, "motion": 1.0, "position": 0.9}  # ranking only
KIND_LABELS = {"color": "colour", "shape": "shape", "light": "light", "motion": "motion",
               "position": "position"}
MOVING = 0.08  # frame widths/s
SAME_SHOT_S = 2.0  # same clip, closer than this: a continuous shot, not a match cut
SAME_PICTURE = 0.85  # shape, light and colour all at least this: the same picture (a retake)
SAME_DETAIL = 0.95  # … or detail this alike alone: the same footage (e.g. a clip sent twice,
# compressed and as a file, with different colours)
TOP_FILE = 20


@dataclass
class Desc:
    clip: str
    scene: str
    f: int  # clip frame
    t: float  # seconds into the scene
    lab: np.ndarray
    grid: np.ndarray  # raw 0–1
    shape: np.ndarray | None  # normalised (None: flat picture)
    mv: np.ndarray
    box: tuple[float, float, float, float] | None


def _blur(g: np.ndarray) -> np.ndarray:
    """Separable [1 4 6 4 1] blur with edge padding (the low frequencies ``shape`` ignores)."""
    k = np.array([1, 4, 6, 4, 1], float) / 16
    p = np.pad(g, 2, mode="edge")
    rows = sum(k[i] * p[:, i:i + g.shape[1]] for i in range(5))
    return sum(k[i] * rows[i:i + g.shape[0], :] for i in range(5))


def _box_at(rows: list[dict] | None, f: int, near: int) -> tuple[float, float, float, float] | None:
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(r["f"] - f))
    return tuple(r["box"]) if abs(r["f"] - f) <= near else None


def describe(project: Project, index: SceneIndex) -> list[Desc]:
    """Descriptors of every usable sample that has the match fields."""
    from opencine.core.time import Rate
    from opencine.index.model import load_samples, load_subjects

    rate = Rate(index.rate)
    out: list[Desc] = []
    for clip in index.clips:
        if clip.same_as:  # sent twice: its twin is described
            continue
        rows = load_samples(project, clip.id) or []
        subj = load_subjects(project, clip.id)
        scenes = [s for s in index.scenes if s.clip == clip.id]
        for r in rows:
            if r.get("bad") or not r.get("grid"):
                continue
            sc = next((s for s in scenes if s.start_frame <= r["f"] < s.end_frame), None)
            if sc is None:
                continue
            side = math.isqrt(len(r["grid"]))
            if side * side != len(r["grid"]) or side < 4:
                continue
            grid = np.asarray(r["grid"], float).reshape(side, side) / 255
            detail = grid - _blur(grid)
            sd = detail.std()
            k = side // 4
            out.append(Desc(clip.id, sc.id, r["f"], round((r["f"] - sc.start_frame) / rate.fps, 2),
                            np.asarray(r["lab"], float), grid.reshape(4, k, 4, k).mean(axis=(1, 3)).ravel(),
                            ((detail - detail.mean()) / sd).ravel() if sd > 0.01 else None,
                            np.asarray(r.get("mv") or [0, 0], float),
                            _box_at(subj, r["f"], round(rate.fps / 2))))
    return out


def score(a: Desc, b: Desc, kind: str) -> float:
    """How well a cut from ``a`` into ``b`` carries ``kind`` over (0–1)."""
    if kind == "color":
        return max(0.0, 1 - float(np.linalg.norm(a.lab - b.lab)) / 40)
    if kind == "shape":
        if a.shape is None or b.shape is None:
            return 0.0
        return max(0.0, float(np.mean(a.shape * b.shape)))
    if kind == "light":
        return max(0.0, 1 - 4 * float(np.mean(np.abs(a.grid - b.grid))))
    if kind == "motion":
        sa, sb = float(np.linalg.norm(a.mv)), float(np.linalg.norm(b.mv))
        if sa < MOVING or sb < MOVING:
            return 0.0
        return max(0.0, float(a.mv @ b.mv) / (sa * sb)) * min(sa, sb) / max(sa, sb)
    if kind == "position":
        if a.box is None or b.box is None:
            return 0.0
        ca = (a.box[0] + a.box[2] / 2, a.box[1] + a.box[3] / 2)
        cb = (b.box[0] + b.box[2] / 2, b.box[1] + b.box[3] / 2)
        size = math.sqrt(max(1e-6, a.box[2] * a.box[3]) / max(1e-6, b.box[2] * b.box[3]))
        return max(0.0, 1 - math.dist(ca, cb) / 0.25) * min(size, 1 / size)
    raise ValueError(f"unknown match kind {kind!r}; one of {', '.join(KINDS)}")


def best(a: Desc, b: Desc, kind: str | None = None) -> tuple[str, float]:
    """``(kind, score)``: the given kind, or the one that ranks highest."""
    if kind is not None:
        return kind, score(a, b, kind)
    return max(((k, score(a, b, k)) for k in KINDS), key=lambda ks: ks[1] * WEIGHT[ks[0]])


def _matrix(src: list[Desc], dst: list[Desc], kind: str) -> np.ndarray:
    """``score`` for every (src, dst) pair, vectorised."""
    if kind == "color":
        la, lb = np.stack([d.lab for d in src]), np.stack([d.lab for d in dst])
        return np.clip(1 - np.linalg.norm(la[:, None] - lb[None], axis=2) / 40, 0, None)
    if kind == "shape":
        zero = np.zeros(len(next((d.shape for d in src + dst if d.shape is not None), np.zeros(1))))
        sa = np.stack([d.shape if d.shape is not None else zero for d in src])
        sb = np.stack([d.shape if d.shape is not None else zero for d in dst])
        return np.clip(sa @ sb.T / sa.shape[1], 0, None)
    if kind == "light":
        ga, gb = np.stack([d.grid for d in src]), np.stack([d.grid for d in dst])
        return np.clip(1 - 4 * np.abs(ga[:, None] - gb[None]).mean(axis=2), 0, None)
    return np.array([[score(a, b, kind) for b in dst] for a in src])


def candidates(project: Project, *, from_scene: str | None = None, kind: str | None = None,
               top: int = 5, descs: list[Desc] | None = None) -> list[dict[str, Any]]:
    """Best cut per scene pair, best first: ``{from: "S03@4.2", to: "S11@0.6", kind, score}``."""
    from opencine.core.time import Rate
    from opencine.index.model import load_index

    index = load_index(project)
    descs = describe(project, index) if descs is None else descs
    if kind is not None and kind not in KINDS:
        raise ValueError(f"unknown match kind {kind!r}; one of {', '.join(KINDS)}")
    src = [d for d in descs if from_scene is None or d.scene == from_scene.strip().upper()]
    if not src or len(descs) < 2:
        return []
    kinds = [kind] if kind else list(KINDS)
    raw = {k: _matrix(src, descs, k) for k in set(kinds) | {"shape", "light", "color"}}
    same = (raw["shape"] >= SAME_DETAIL) | (
        (raw["shape"] >= SAME_PICTURE) & (raw["light"] >= SAME_PICTURE) & (raw["color"] >= SAME_PICTURE))
    stack = np.stack([raw[k] * (1 if kind else WEIGHT[k]) for k in kinds])
    which, val = stack.argmax(axis=0), stack.max(axis=0)
    rate = float(Rate(index.rate).fps)
    pairs: dict[tuple[str, str], tuple[float, int, int]] = {}
    for i, a in enumerate(src):
        for j, b in enumerate(descs):
            if a.scene == b.scene or same[i, j] or (a.clip == b.clip and abs(a.f - b.f) < SAME_SHOT_S * rate):
                continue
            v = float(val[i, j])
            key = (a.scene, b.scene) if from_scene else tuple(sorted((a.scene, b.scene)))
            if v > pairs.get(key, (-1.0, 0, 0))[0]:
                pairs[key] = (v, i, j)
    ranked = sorted(pairs.values(), key=lambda p: -p[0])[:top]
    out = []
    for _, i, j in ranked:
        k = kinds[int(which[i, j])]
        out.append({"from": f"{src[i].scene}@{src[i].t:g}", "to": f"{descs[j].scene}@{descs[j].t:g}",
                    "kind": k, "score": round(score(src[i], descs[j], k), 2)})
    return out


def matches_path(project: Project):
    return project.index_dir / "matches.json"


def save_top(project: Project) -> list[dict[str, Any]]:
    """Keep the strongest pairs for L0 (``index/matches.json``)."""
    found = candidates(project, top=TOP_FILE)
    matches_path(project).write_text(json.dumps(found, ensure_ascii=False), "utf-8")
    return found


def l0_lines(project: Project, n: int = 3) -> list[str]:
    """``- S03@4.2 → S11@0.6 · shape 0.82`` for the strongest ``n`` pairs (if ingest made them)."""
    p = matches_path(project)
    if not p.exists():
        return []
    rows = json.loads(p.read_text("utf-8"))[:n]
    return [f"- {r['from']} → {r['to']} · {KIND_LABELS[r['kind']]} {r['score']:.2f}" for r in rows]
