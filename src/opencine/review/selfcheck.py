"""The Director's own look at its work before a review gate: cheap, deterministic sheets.

- ``review_sheet`` (G4): frames of the rendered preview as the viewer sees them, plus a timeline table.
- ``storyboard_sheet`` (G1): one tile per ShotPlan slot, cropped to the plan's frame.
- ``gen_sheet`` (G2/G3): generated stills and clips, for content QC CPU checks miss (extra fingers).

Charged to the review image pool; each kind keeps its newest ``KEEP_SHEETS`` in ``review/self``.
"""

from __future__ import annotations

import json
import re
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

from opencine.core.ffmpeg import run
from opencine.core.project import Project
from opencine.core.retention import remove, stale_sheets
from opencine.core.time import fmt_ts
from opencine.index.sheets import ImageRef, compose, frame_at
from opencine.plan.schema import ClipSource, EditItem, EditPlan, GenSource, SceneSource, Slot
from opencine.plan.store import load_plan

TILE = 384  # tile long edge: 216×384 cells, well above the ~200 px where vision misreads
MAX_TILES = 28  # 7 × 4 such cells fit one ≤ 1568 px sheet (~3k tokens); more → another sheet
ASPECT = 9 / 16
FRAME_REF = re.compile(r"^[SC]\d+(@[\d:.,]+)?$", re.IGNORECASE)  # what frame_at takes


def _num(x: float) -> str:
    return f"{x:.1f}"


def sheets(project: Project, stem: str, tiles: list[tuple[Image.Image, str]],
           aspect: float = ASPECT) -> list[ImageRef]:
    """``review/self/<stem>[-k].jpg``, ``MAX_TILES`` tiles each, cells at ``aspect`` (w / h);
    older sheets of a kind go."""
    groups = [tiles[i:i + MAX_TILES] for i in range(0, len(tiles), MAX_TILES)]
    out = project.review_dir / "self"
    refs = [compose(g, out / (stem + (f"-{k}" if len(groups) > 1 else "") + ".jpg"),
                    aspect=aspect, max_cell=TILE)
            for k, g in enumerate(groups, 1)]
    remove(stale_sheets(project))
    return refs


# ---- review sheet (EditPlan → its preview) -----------------------------------------
def render_path(project: Project, version: int, preset: str | None = None) -> Path:
    """The newest finished 9:16 render of editplan ``version``: preview, else final (or only
    ``preset``)."""
    found: dict[str, tuple[str, Path]] = {}
    for f in (project.root / "jobs").glob("render-*.json"):
        try:
            job = json.loads(f.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        res = job.get("result") or {}
        path = Path(res.get("path") or "")
        if job.get("status") != "done" or res.get("version") != version or not path.is_file():
            continue
        if (job.get("params") or {}).get("aspect") == "16:9":  # the landscape version: not reviewed
            continue
        kind = (job.get("params") or {}).get("preset") or "preview"
        if kind not in found or job.get("finished", "") > found[kind][0]:
            found[kind] = (job.get("finished", ""), path)
    for p in ("preview", "final") if preset is None else (preset,):
        if p in found:
            return found[p][1]
    what = f"{preset} render" if preset else "render"
    raise FileNotFoundError(f"editplan v{version} has no finished {what}; "
                            f"call render(pid, version={version}) first")


def grab_frames(video: Path, frames: list[int], height: int = TILE) -> dict[int, Image.Image]:
    """Exact frames ``n`` of a CFR ``video`` (scaled to ``height``) in one decode pass."""
    want = sorted(set(frames))
    select = "+".join(f"eq(n\\,{f})" for f in want)
    with tempfile.TemporaryDirectory() as d:
        run(["-y", "-loglevel", "error", "-i", str(video), "-vf", f"select={select},scale=-2:{height}",
             "-fps_mode", "passthrough", "-q:v", "3", str(Path(d) / "f%04d.jpg")])
        images = []
        for p in sorted(Path(d).glob("f*.jpg")):
            with Image.open(p) as im:
                images.append(im.copy())
    if len(images) != len(want):
        raise RuntimeError(f"{video.name}: got {len(images)} of {len(want)} frames; "
                           "is the render older than the plan?")
    return dict(zip(want, images, strict=True))


def _source_text(it: EditItem, rate: Fraction) -> str:
    src = it.source
    if isinstance(src, SceneSource | ClipSource):
        sid = src.scene if isinstance(src, SceneSource) else src.clip
        span = f"{sid} {float(it.in_ / rate):.2f}–{float(it.out / rate):.2f} s"
    else:
        span = Path(src.gen if isinstance(src, GenSource) else src.image).name
    if it.ramp:
        from opencine.plan.ramp import label

        return f"{span} {label(it.ramp)}"
    return span + (f" ×{it.speed:g}" if it.speed != 1 else "")


BAND_VI = {"top": "top", "center": "middle", "bottom": "bottom"}


def timeline_table(plan: EditPlan, version: int, bands: dict[int, str] | None = None) -> str:
    """One line per item: ``#3 00:04.2 +1.4 s · S05 0.40–1.80 s · whip 6f · text "…" (top)``;
    ``bands``: item index → where its text went (``render.textplace``)."""
    bands = bands or {}
    rate = plan.fps
    lengths = [it.length / rate for it in plan.items]
    total = plan.total_frames / rate
    head = (f"# editplan v{version} · {len(plan.items)} item · {_num(float(total))} s · "
            f"shot {_num(float(min(lengths)))}–{_num(float(max(lengths)))} s, "
            f"avg {_num(float(sum(lengths) / len(lengths)))} s")
    lines = [head]
    for n, (start, it) in enumerate(zip(plan.starts(), plan.items, strict=True), 1):
        tr = "cut" if it.is_cut else f"{it.transition_in.kind} {it.transition_in.frames}f"
        parts = [f"#{n} {fmt_ts(start / rate)} +{_num(float(it.length / rate))} s",
                 _source_text(it, rate), tr]
        if it.text is not None:
            band = BAND_VI.get(bands.get(n - 1, it.text.position), "")
            parts.append(f'text "{it.text.content.replace(chr(10), " / ")[:40]}"' + (f" ({band})" if band else ""))
        if it.gain is None:
            parts.append("muted")
        elif it.sound is not None:
            s = it.sound
            parts.append(f"audio {s.scene or s.clip} {float(s.in_ / rate):.2f} s")
        if it.note:
            parts.append(f"note: {it.note[:40]}")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def review_sheet(project: Project, version: int | None = None,
                 cuts: list[int] | None = None) -> tuple[list[ImageRef], str]:
    """Sheets ``review/self/editplan-vNNN[-k].jpg`` and the timeline table.

    ``cuts``: item numbers ``n`` (1-based, ≥ 2) whose incoming cut to show as a pair: the last
    frame before it and the first after it (after the transition, if any)."""
    v, plan = load_plan(project, "editplan", version)
    video = render_path(project, v)
    starts, last = plan.starts(), plan.total_frames - 1
    rate = plan.fps
    wanted: list[tuple[int, str]] = [(0, "hook 0s")]
    for n, (start, it) in enumerate(zip(starts, plan.items, strict=True), 1):
        wanted.append((min(last, start + it.length // 2), f"#{n} {float(it.length / rate):.1f}s"))
    for n in cuts or []:
        if not 2 <= n <= len(plan.items):
            raise ValueError(f"cut {n}: give item numbers 2–{len(plan.items)} (the item after the cut)")
        it = plan.items[n - 1]
        after = starts[n - 1] + (0 if it.is_cut else it.transition_in.frames)
        wanted += [(max(0, starts[n - 1] - 1), f"#{n - 1} out"), (min(last, after), f"#{n} in")]
    images = grab_frames(video, [f for f, _ in wanted])
    w, h = plan.size
    refs = sheets(project, f"editplan-v{v:03d}", [(images[f], label) for f, label in wanted],
                  aspect=w / h)
    from opencine.render.textplace import layout

    bands = {p.item: p.band for p in layout(project, plan)}
    return refs, timeline_table(plan, v, bands) + f"\nrender: {video.name}"


# ---- storyboard (ShotPlan) -------------------------------------------------------
def _cover(im: Image.Image, aspect: float = ASPECT) -> Image.Image:
    """Centre crop to ``aspect``, as the renderer's default (no crop box) does."""
    w, h = im.size
    if w / h > aspect:
        nw = round(h * aspect)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = round(w / aspect)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def _gen_image(project: Project, slot: Slot) -> tuple[Image.Image | None, str]:
    """What was generated for a slot with no ref (approved > latest; a clip's first frame)."""
    from opencine.plan.solve import _gen_media

    found = _gen_media(project, str(slot.slot))
    if found is None:
        return None, "GEN (no ref)"
    kind, rel = found
    if kind == "video":
        return grab_frames(project.root / rel, [0])[0], rel
    with Image.open(project.root / rel) as im:
        return im.convert("RGB"), rel


def _slot_image(project: Project, slot: Slot,
                aspect: float = ASPECT) -> tuple[Image.Image | None, str]:
    """(picture cropped to ``aspect`` or None, where it comes from) for one slot."""
    src = slot.source
    if src.footage is not None:
        f = src.footage
        ref = f.scene if f.at is None else f"{f.scene}@{f.at}"
    elif src.generate is not None and src.generate.ref:
        ref = src.generate.ref
    elif src.transform is not None:  # the window's first frame (before it is remade)
        ref = f"{src.transform.scene}@{src.transform.in_s}"
    elif src.generate is not None:
        img, where = _gen_image(project, slot)
        return (_cover(img, aspect) if img is not None else None), where
    else:
        return None, "?"
    if FRAME_REF.match(ref.strip()):
        path = Path(frame_at(project, ref).path)
    else:  # a file: an approved still, an image the user sent
        path = Path(ref) if Path(ref).is_absolute() else project.root / ref
        if not path.is_file():
            raise FileNotFoundError(f"slot {slot.slot}: ref {ref!r} is neither a frame ref "
                                    "(S03@2.5, C01@12.4) nor a file")
    with Image.open(path) as im:
        return _cover(im.convert("RGB"), aspect), ref


def _gen_status(made: dict[str, dict[str, Any]], sid: str) -> str:
    """A generated slot's state for the storyboard: "pending", or "image ×1 · clip ×2 ok"."""
    parts = [f"{word} ×{e.get('attempts', 0)}" + (" ok" if e.get("approved") else "")
             for kind, word in (("image", "image"), ("video", "clip"))
             if (e := made.get(kind, {}).get(sid))]
    return " · ".join(parts) or "pending"


def storyboard_sheet(project: Project, version: int | None = None) -> tuple[list[ImageRef], str]:
    """Sheet ``review/self/shotplan-vNNN.jpg`` (one tile per slot, ``#n hook GEN CU``) and a
    text table of the slots (durations, source, intent)."""
    from opencine.gen import slots as gen_slots
    from opencine.plan.schema import DELIVERABLES
    from opencine.plan.templates import template

    v, sp = load_plan(project, "shotplan", version)
    w, h = DELIVERABLES[sp.deliverable or template(sp.template).deliverable]
    aspect = w / h
    made = gen_slots.load(project)
    tiles, lines = [], [f"# shotplan v{v} · {len(sp.slots)} slot · "
                        f"{_num(sum(s.dur_s[0] for s in sp.slots))}–{_num(sum(s.dur_s[1] for s in sp.slots))} s"]
    for s in sp.slots:
        img, where = _slot_image(project, s, aspect)
        gen = s.source.generate is not None
        tf = s.source.transform
        if img is None:
            img = Image.new("RGB", (216, 384) if aspect < 1 else (384, 216), (60, 60, 60))
        kind = "GEN" if gen else (tf.mode.upper() if tf else None)
        label = " ".join(x for x in (f"#{s.slot}", s.role, kind, s.shot) if x)
        tiles.append((img, label))
        extra = [x for x in (
            s.shot, s.cut_on and f"cut on {s.cut_on}",
            s.match and f"match {s.match.kind}" + (f" ({s.match.note[:30]})" if s.match.note else ""),
            s.transition_out and f"→ {s.transition_out}", s.speed and s.speed != "normal" and s.speed,
            s.text and f'text "{s.text[:30]}"', s.sfx and f"sfx {s.sfx}",
            (gen or tf) and _gen_status(made, str(s.slot))) if x]
        how = ("generated from " if gen else
               f"{'swap' if tf.mode == 'swap' else 'restyle'}"
               f"{' → ' + tf.look if tf.look else ''}, {_num(tf.out_s - tf.in_s)} s from " if tf else "")
        lines.append(" · ".join([f"#{s.slot} {s.purpose[:50]}", f"{_num(s.dur_s[0])}–{_num(s.dur_s[1])} s",
                                 how + where, *extra]))
    return sheets(project, f"shotplan-v{v:03d}", tiles, aspect), "\n".join(lines)


# ---- generated media (before G2/G3) -------------------------------------------------------
def _frame_count(video: Path) -> int:
    from opencine.gen.qc import fps_of, video_stream

    v = video_stream(video)
    if str(v.get("nb_frames") or "").isdigit():
        return int(v["nb_frames"])
    return max(1, int(float(v.get("duration") or 0) * fps_of(v)))


def gen_sheet(project: Project, paths: list[str]) -> tuple[list[ImageRef], str]:
    """Sheet(s) ``review/self/gen-<hash>.jpg`` of generated stills and clips, numbered in
    ``paths`` order like the gate's files: one tile per still, first · middle · last frame per
    clip; plus a table of each variant's QC flags. Cells take the first item's aspect."""
    import hashlib

    tiles, lines = [], []
    aspect: float | None = None
    for n, rel in enumerate(paths, 1):
        p = Path(rel) if Path(rel).is_absolute() else project.root / rel
        if not p.is_file():
            raise FileNotFoundError(f"#{n}: {rel} does not exist")
        side = p.with_suffix(".json")
        meta = json.loads(side.read_text("utf-8")) if side.exists() else {}
        flags = ", ".join(meta.get("flags") or []) or "—"
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            with Image.open(p) as im:
                aspect = aspect or im.width / im.height
                tiles.append((_cover(im.convert("RGB"), aspect), f"#{n}"))
            lines.append(f"#{n} {rel} · image · flags: {flags}")
            continue
        count = _frame_count(p)
        picks = sorted({0, count // 2, max(0, count - 1)})
        got = grab_frames(p, picks)
        aspect = aspect or got[picks[0]].width / got[picks[0]].height
        for f, label in zip(picks, ("head", "mid", "tail"), strict=False):
            tiles.append((_cover(got[f], aspect), f"#{n} {label}"))
        secs = f"{meta['frames'] / float(project.rate):.2f} s" if meta.get("frames") else f"{count} frames"
        lines.append(f"#{n} {rel} · clip {secs} · flags: {flags}")
    key = hashlib.sha256("|".join(paths).encode()).hexdigest()[:8]
    return sheets(project, f"gen-{key}", tiles, aspect or ASPECT), "\n".join(lines)
