"""L0: the scene table as compact text (``index/scenes.md``, ~35 tokens/scene).

Under each scene line: the Director's note (``note:``, kept in ``index/notes.json``) and, when
transcribed, what is said (``speech:``).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from opencine.core.time import fmt_ts
from opencine.index.model import ClipInfo, Scene, SceneIndex, load_index, load_subjects
from opencine.ingest.probe import low_res

if TYPE_CHECKING:
    from opencine.core.project import Project

STATIC_MOTION = 0.02  # below this mean motion the motion peak is noise: not printed
SUBJECT_MIN = 0.1  # subject motion below this is not printed (``subj``)
NOTE_CHARS = 200  # per scene note
COVER_AREA = 0.4  # frame share a moving thing covers in a scene's last second: a mask wipe


def _num(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def _fps(text: str) -> str | None:
    try:
        f = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    if f <= 0:
        return None
    return str(round(f)) if abs(f - round(f)) < 0.05 else _num(f, 2)


def shot_label(iso: str | None) -> str | None:
    """``01/10 06:12``: local wall-clock time of a ``ClipInfo.shot_at``."""
    if not iso:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(iso).strftime("%d/%m %H:%M")
    except ValueError:
        return None


def clip_line(c: ClipInfo) -> str:
    """``## C01 · 28.4 s · portrait 464×848 · 30 fps · … · shot 01/10 06:12``.

    ``low-res ×N``: the upscale a 1080×1920 final needs (chat apps compress videos sent as
    video). ``shot``: when it was recorded; ``(file name)`` = parsed from the file name."""
    parts = [c.id, f"{_num(c.duration_s, 1)} s",
             f"{_orientation(c.width, c.height)} {c.src_width}×{c.src_height}"]
    if fps := _fps(c.fps_avg or c.fps_nominal):
        parts.append(f"{fps} fps")
    if c.vfr:
        parts.append("VFR")
    if c.hdr:
        parts.append(f"HDR {c.hdr.upper()}")
    if c.audio is None:
        parts.append("no audio")
    if low_res(c.src_width, c.src_height):
        up = 1080 / min(c.src_width, c.src_height * 9 / 16)  # 9:16 cover of the frame
        parts.append(f"low-res ×{_num(up, 1)}")
    if shot := shot_label(c.shot_at):
        parts.append(f"shot {shot}" + (" (file name)" if c.shot_from == "name" else ""))
    return "## " + " · ".join(parts)


def scene_line(s: Scene) -> str:
    """``S03 00:12.4–00:16.0 · 3.6 s · motion 0.53 · cam → · subj 0.6 · peak +2.1 s · head ↓ tail →
    · yellow · audio +2.0 s · bright 0.48 · sharp 1.0``.

    ``cam``: camera move (pan/tilt arrow, zoom±, ``shake`` = handheld; absent = still);
    ``subj``: motion against the camera; ``peak``: motion peak; ``head``/``tail``: on-screen
    motion over the first/last second (``·`` = none); then the dominant colour and ``audio``:
    the strongest sound onset. Offsets are seconds into the scene (as in ``S03@2.1`` / ``in_s``);
    fields without data are omitted."""
    line = f"{s.id} {fmt_ts(s.start_s)}–{fmt_ts(s.end_s)} · {_num(s.duration_s, 1)} s · motion {_num(s.motion, 2)}"
    if s.cam_dir:
        line += f" · cam {s.cam_dir}"
    if s.subject_motion is not None and s.subject_motion >= SUBJECT_MIN:
        line += f" · subj {_num(s.subject_motion, 1)}"
    if s.motion_peak_s is not None and s.motion >= STATIC_MOTION:
        line += f" · peak +{_num(max(0.0, s.motion_peak_s - s.start_s), 1)} s"
    if s.head_dir and (s.head_dir, s.tail_dir) != ("·", "·"):
        line += f" · head {s.head_dir} tail {s.tail_dir or '·'}"
    if s.colour:
        line += f" · {s.colour}"
    if s.audio_peak_s is not None:
        line += f" · audio +{_num(max(0.0, s.audio_peak_s - s.start_s), 1)} s"
    return line + f" · bright {_num(s.brightness, 2)} · sharp {_num(s.sharpness, 1)}"


def covers_at_end(rows: list[dict] | None, s: Scene, rate: float) -> bool:
    """Something moving (subject-track motion boxes) covers ≥ ``COVER_AREA`` of the frame in the
    scene's last second: a natural mask wipe."""
    last = s.end_frame - round(rate)
    return any(r.get("kind") == "motion" and r["box"][2] * r["box"][3] >= COVER_AREA
               for r in rows or [] if max(s.start_frame, last) <= r["f"] < s.end_frame)


# ---- the Director's scene notes ---------------------------------------------------
def notes_path(project: Project) -> Path:
    return project.index_dir / "notes.json"


def load_notes(project: Project) -> dict[str, str]:
    p = notes_path(project)
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def save_notes(project: Project, notes: dict[str, str]) -> dict[str, str]:
    """Merge one-line notes per scene id (empty text deletes one) and refresh ``scenes.md``.

    Notes are keyed by scene id (stable when clips are added and re-ingested); all ids must exist.
    """
    index = load_index(project)
    order = {s.id: i for i, s in enumerate(index.scenes)}
    unknown = sorted(k for k in notes if k.strip().upper() not in order)
    if unknown:
        have = f"{index.scenes[0].id}–{index.scenes[-1].id}" if index.scenes else "no scenes"
        raise KeyError(f"unknown scene id(s) {', '.join(unknown)}; this project has {have}")
    cur = load_notes(project)
    for sid, text in notes.items():
        sid, text = sid.strip().upper(), " ".join(str(text or "").split())
        if not text:
            cur.pop(sid, None)
        else:
            cur[sid] = text if len(text) <= NOTE_CHARS else text[:NOTE_CHARS - 1] + "…"
    cur = {k: cur[k] for k in sorted(cur, key=lambda k: order.get(k, len(order)))}
    p = notes_path(project)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(p)
    scenes_md(project)
    return cur


def estimate_tokens(text: str) -> int:
    """Rough token count for symbol- and number-heavy text (~2 characters per token)."""
    return math.ceil(len(text) / 2)


def _orientation(w: int, h: int) -> str:
    return "portrait" if h > w else "landscape" if w > h else "square"


def describe_images(project: Project) -> list[dict[str, Any]]:
    """The project's stills (``I01``…) with their displayed size."""
    out = []
    for ref in project.meta.images:
        with Image.open(project.root / ref.file) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # EXIF says rotated 90°
                w, h = h, w
        out.append({"id": ref.id, "file": ref.file, "width": w, "height": h})
    return out


def l0_text(project: Project, index: SceneIndex) -> str:
    """The L0 text for ``index`` (no side effects): clips and scenes with notes and speech,
    match-cut hints, then the project's stills."""
    from opencine.speech.transcribe import load_transcripts, speech_line

    trs = load_transcripts(project)
    notes = load_notes(project)
    lines = [f"# {project.pid} · {len(index.clips)} clip · {len(index.scenes)} scene · {index.rate} fps"]
    dup = {c.id for c in index.clips if c.same_as}
    order = [c for c in index.shot_order() if c not in dup]
    if len(order) > 1 and order != [c.id for c in index.clips if c.id not in dup] \
            and any(c.shot_at for c in index.clips):
        lines.append("shooting order: " + " → ".join(order))
    for c in index.clips:
        if c.same_as:
            keep = index.clip(c.same_as)
            lines.append(f"## {c.id} = {keep.id} (sent twice; using {keep.id} {keep.src_width}×{keep.src_height})")
            continue
        lines.append(clip_line(c))
        subj = load_subjects(project, c.id)
        for s in index.scenes:
            if s.clip == c.id:
                lines.append(scene_line(s) + (" · tail: object covers frame → mask" if covers_at_end(
                    subj, s, float(index.rate)) else ""))
                if note := notes.get(s.id):
                    lines.append(f"  note: {note}")
                if said := speech_line(trs.get(c.id), s.start_s, s.end_s):
                    lines.append(said)
    from opencine.plan.match import l0_lines

    if pairs := l0_lines(project):
        lines.append("## Match-cut hints (last frame → first frame; match_candidates for more)")
        lines += pairs
    if images := describe_images(project):
        lines.append('## Images · use {"image": file} in an EditPlan or as a gen_image ref')
        lines += [f"{i['id']} {i['width']}×{i['height']} {_orientation(i['width'], i['height'])}"
                  f" · {i['file']}" for i in images]
    return "\n".join(lines) + "\n"


def scenes_md(project: Project) -> str:
    """Render L0 from ``scenes.json``, write ``index/scenes.md`` and return the text."""
    text = l0_text(project, load_index(project))
    (project.index_dir / "scenes.md").write_text(text, "utf-8")
    return text
