"""``position: auto`` for item text: the safe-area band (top, center, bottom) where the card hides
the least.

Each band's card box (measured with the shipped font) is compared every quarter second with the
picture through the item's crop and zoom: how much of the tracked subject it covers (face, else
motion, else saliency) and how busy the picture under it is. Near ties keep the style's usual
band; a band the voice captions hold is avoided; without tracking data the usual band is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from opencine.plan.schema import EditPlan, GenSource, ImageSource, SceneSource, TextOverlay
from opencine.render.text import BAND_ORDER, FONTS_DIR, margins

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.plan.media import Media

BANDS = ("top", "center", "bottom")
PREFER = BAND_ORDER
KIND_WEIGHT = {"face": 1.0, "motion": 0.8, "saliency": 0.5}  # how sure a box is the subject
STEP_S = 0.25  # sampling of the card's time on screen
ROW_GAP_S = 0.3  # a subject/sample row this close to a sampled moment stands for it
BUSY_WEIGHT = 0.25  # cost of fully busy picture under the card, vs. hiding the whole subject
RANK_STEP = 0.04  # cost per step down the style's band order (near ties keep the usual band)
BLOCKED = 10.0  # cost of the captions band while captions run
LINE_H = 1.1  # line pitch / font size as libass sets the shipped fonts, plus outline
HIDES = {"face": 0.1, "motion": 0.25, "saliency": 0.25}  # share of the subject's box covered
# that counts as hiding it at that moment (a face shows a cut forehead; a wide moving area less)
COVER_HINT = 0.3  # craft hint when the subject is hidden this share of the card's time


def fallback(style: str) -> str:
    """The style's usual band (``auto`` without tracking data)."""
    return PREFER[style][0]


@dataclass
class Placement:
    item: int
    band: str  # where the card goes
    auto: bool  # the plan said "auto"
    hidden: dict[str, float] = field(default_factory=dict)  # band → share of time the subject is hidden
    subject: str | None = None  # face | motion | saliency seen under it; None = no tracking data
    blocked: tuple[str, ...] = ()  # bands the captions hold meanwhile
    shown_s: float = 0.0


# ---- the card ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _font(path: str, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px)


def card_box(t: TextOverlay, band: str, width: int, height: int) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)``, fractions of the frame, of the card drawn in ``band``."""
    bottom, top, side = margins(width, height)
    px = max(1, round(t.font_size * height))
    lines = t.content.split("\n")
    face = FONTS_DIR / ("Montserrat-ExtraBold.ttf" if t.style == "title" else "Montserrat-Bold.ttf")
    try:
        font = _font(str(face), px)
        widths = [font.getlength(ln) for ln in lines]
    except OSError:  # no shipped font: an average glyph width
        widths = [0.55 * px * len(ln) for ln in lines]
    avail = width * (1 - 2 * side)
    n = sum(max(1, math.ceil(w / avail)) for w in widths)
    w, h = min(max(widths, default=0.0), avail), n * LINE_H * px
    if t.box or t.style == "label":
        pad = 2 * max(2, round(height / 110))
        w, h = w + pad, h + pad
    x0 = side * width if t.style == "time" else (width - w) / 2
    y0 = {"top": top * height, "center": (height - h) / 2, "bottom": height * (1 - bottom) - h}[band]
    return x0 / width, y0 / height, (x0 + w) / width, (y0 + h) / height


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


# ---- what the picture shows ----------------------------------------------------------------
def _detail(grid: list[int]) -> np.ndarray | None:
    import cv2

    side = math.isqrt(len(grid))
    if side < 4 or side * side != len(grid):
        return None
    g = np.asarray(grid, np.float32).reshape(side, side) / 255
    return np.abs(g - cv2.blur(g, (3, 3), borderType=cv2.BORDER_REPLICATE))


def _nearest(rows: list[dict], t: float, rate: float) -> dict | None:
    best = min(rows, key=lambda r: abs(r["f"] / rate - t), default=None)
    return best if best is not None and abs(best["f"] / rate - t) <= ROW_GAP_S else None


def _generated_rows(project: Project, m: Media, rate: Fraction) -> list[dict]:
    """Subject rows of a generated clip or still (frames of the file), cached by its identity."""
    from opencine.core.cache import cache_key
    from opencine.index import subjects

    st = m.path.stat()
    key = cache_key("text-subjects", str(m.path.resolve()), st.st_size, st.st_mtime_ns, str(rate),
                    subjects.params())
    f = project.cache_dir / "subjects" / f"{key[:16]}.json"
    if f.exists():
        return json.loads(f.read_text("utf-8"))
    try:
        if m.kind == "image":
            import cv2

            img = cv2.imdecode(np.fromfile(m.path, np.uint8), cv2.IMREAD_COLOR)
            found, kind = subjects.FaceFinder(subjects.face_model())(img), "face"
            if found is None:
                k = subjects.SAL_EDGE / max(img.shape[:2])
                small = cv2.resize(img, (max(1, round(img.shape[1] * k)), max(1, round(img.shape[0] * k))),
                                   interpolation=cv2.INTER_AREA)
                found, kind = subjects.saliency_box(small), "saliency"
            rows = [{"f": 0, "box": found[0], "kind": kind, "conf": found[1]}] if found else []
        else:
            rows = subjects.track(m.path, rate, m.hard_frames or 0, [])
    except Exception:  # noqa: BLE001 — unreadable: placed without tracking
        rows = []
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows), "utf-8")
    return rows


class _Picture:
    """Per output frame of one item: the subject box and the detail grid it shows, in output
    fractions."""

    def __init__(self, project: Project, plan: EditPlan, it, m: Media, index):
        from opencine.core.time import Rate
        from opencine.index.model import load_samples, load_subjects
        from opencine.render.segments import item_size

        self.it, self.m, self.rate = it, m, plan.fps
        W, H = plan.size
        self.fw, self.fh = item_size(it, W, H)
        self.col = ((W - self.fw) / 2 / W, self.fw / W)  # fit_blur: the column the picture fills
        n = it.length
        if it.ramp:  # source frame (from the source start) shown at each output frame
            from opencine.plan.ramp import shown

            self.src = [it.in_ + f for f in shown(it.ramp, max(1, it.out - it.in_), n)]
        else:
            self.src = [it.in_ + j * it.speed for j in range(n)]
        self.rows: list[dict] = []
        self.grids: list[dict] = []
        self.row_rate = float(self.rate)
        self.t0 = 0.0  # file seconds of source frame 0
        if isinstance(it.source, GenSource | ImageSource):
            self.rows = _generated_rows(project, m, self.rate)
            self.still = m.kind == "image"
        else:
            self.still = False
            clip = index.scene(it.source.scene).clip if isinstance(it.source, SceneSource) else it.source.clip
            self.rows = load_subjects(project, clip) or []
            self.grids = [g for g in load_samples(project, clip) or [] if g.get("grid") and not g.get("bad")]
            self.row_rate = float(Rate(index.rate))
            self.t0 = float(m.start_s)

    def at(self, j: int) -> tuple[tuple[float, float, float, float] | None, str | None, Any]:
        """(subject box, its kind, (window, detail grid)) at output frame ``j``."""
        from opencine.render.segments import source_window

        win = source_window(self.it, self.m, self.fw, self.fh, j)
        t = self.t0 + float(self.src[min(j, len(self.src) - 1)]) / float(self.rate)
        row = (self.rows[0] if self.rows else None) if self.still else _nearest(self.rows, t, self.row_rate)
        box = self._out(row["box"], win) if row else None
        g = _nearest(self.grids, t, self.row_rate) if self.grids else None
        return box, row["kind"] if box else None, (win, _detail(g["grid"]) if g else None)

    def _out(self, b: list[float], win) -> tuple[float, float, float, float] | None:
        x, y, w, h = win
        mw, mh = self.m.width, self.m.height
        x0, x1 = (b[0] * mw - x) / w, ((b[0] + b[2]) * mw - x) / w
        y0, y1 = (b[1] * mh - y) / h, ((b[1] + b[3]) * mh - y) / h
        box = (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))
        if _area(box) <= 0:
            return None  # the subject is outside the picture
        c0, cw = self.col
        return c0 + box[0] * cw, box[1], c0 + box[2] * cw, box[3]

    def busy(self, card: tuple[float, ...], view) -> float | None:
        """Fine detail under ``card`` relative to the whole picture's (0–1, 1/3 = average)."""
        win, det = view
        if det is None:
            return None
        c0, cw = self.col
        x, y, w, h = win
        mw, mh = self.m.width, self.m.height
        side = det.shape[0]
        fx0 = (x + max(0.0, (card[0] - c0) / cw) * w) / mw
        fx1 = (x + min(1.0, (card[2] - c0) / cw) * w) / mw
        fy0, fy1 = (y + card[1] * h) / mh, (y + card[3] * h) / mh
        r0, r1 = int(fy0 * side), max(int(fy0 * side) + 1, math.ceil(fy1 * side))
        k0, k1 = int(fx0 * side), max(int(fx0 * side) + 1, math.ceil(fx1 * side))
        cells = det[max(0, r0):min(side, r1), max(0, k0):min(side, k1)]
        if not cells.size:
            return None
        return min(3.0, float(cells.mean()) / (float(det.mean()) + 1e-6)) / 3


# ---- placing ---------------------------------------------------------------------------------
def _caption_spans(project: Project | None, plan: EditPlan) -> list[tuple[Fraction, Fraction]]:
    if project is None or plan.captions is None:
        return []
    from opencine.plan.captions import cues

    try:
        return [(c.start, c.end) for c in cues(project, plan)]
    except Exception:  # noqa: BLE001 — a missing words file is a caption hint, not a placing error
        return []


def place(project: Project | None, plan: EditPlan, i: int, start: int, medias: list | None = None,
          spans: list[tuple[Fraction, Fraction]] | None = None, index=None) -> Placement:
    """Where items[i]'s text goes; ``start`` = the item's timeline frame."""
    it = plan.items[i]
    t: TextOverlay = it.text
    rate, n = plan.fps, it.length
    a = min(t.start, n - 1)
    b = max(a + 1, min(n, t.end if t.end is not None else n))
    auto = t.position == "auto"
    order = PREFER[t.style]
    cap = plan.captions
    blocked = tuple(
        [cap.position] if cap is not None and any(
            s < Fraction(start + b) / rate and Fraction(start + a) / rate < e for s, e in spans or []) else [])
    W, H = plan.size
    cards = {band: card_box(t, band, W, H) for band in BANDS}
    weight = dict.fromkeys(BANDS, 0.0)
    hidden = dict.fromkeys(BANDS, 0)
    busy: dict[str, list[float]] = {band: [] for band in BANDS}
    kinds: list[str] = []
    samples = 0
    if project is not None:
        try:
            m = medias[i] if medias else _resolver(project, plan).media(it)
            pic = _Picture(project, plan, it, m, index if index is not None else _index(project))
        except Exception:  # noqa: BLE001 — missing media / old index: placed without tracking
            pic = None
        if pic is not None and (pic.rows or pic.grids):
            step = max(1, round(float(rate) * STEP_S))
            for j in range(a, b, step):
                box, kind, view = pic.at(j)
                samples += 1
                for band, card in cards.items():
                    if box is not None:
                        share = _overlap(card, box) / _area(box)
                        weight[band] += share * KIND_WEIGHT.get(kind, 0.5)
                        hidden[band] += share >= HIDES.get(kind, 0.25)
                    if (bz := pic.busy(card, view)) is not None:
                        busy[band].append(bz)
                if kind:
                    kinds.append(kind)
    cost = {}
    for band in BANDS:
        c = order.index(band) * RANK_STEP + (BLOCKED if band in blocked else 0.0)
        if samples:
            c += weight[band] / samples
        if busy[band]:
            c += BUSY_WEIGHT * sum(busy[band]) / len(busy[band])
        cost[band] = c
    band = min(BANDS, key=cost.__getitem__) if auto else t.position
    subject = max(set(kinds), key=lambda k: (kinds.count(k), KIND_WEIGHT.get(k, 0))) if kinds else None
    return Placement(i, band, auto, {k: round(v / samples, 2) for k, v in hidden.items()} if samples else {},
                     subject, blocked, float(Fraction(b - a) / rate))


def _resolver(project: Project, plan: EditPlan):
    from opencine.plan.media import Resolver

    return Resolver(project, plan.fps)


def _index(project: Project):
    from opencine.index.model import load_index

    try:
        return load_index(project)
    except Exception:  # noqa: BLE001 — not ingested: scene items cannot be tracked
        return None


def layout(project: Project | None, plan: EditPlan, medias: list | None = None) -> list[Placement]:
    """A placement for every item with text (``auto`` resolved, the others measured)."""
    spans = _caption_spans(project, plan)
    index = _index(project) if project is not None else None
    return [place(project, plan, i, s, medias, spans, index)
            for i, (s, it) in enumerate(zip(plan.starts(), plan.items, strict=True))
            if it.text is not None and it.text.content.strip()]


def resolve(project: Project | None, plan: EditPlan, medias: list | None = None) -> EditPlan:
    """``plan`` with every ``auto`` text position replaced by its band."""
    if not any(it.text is not None and it.text.position == "auto" for it in plan.items):
        return plan
    bands = {p.item: p.band for p in layout(project, plan, medias) if p.auto}
    items = [it.model_copy(update={"text": it.text.model_copy(update={"position": bands.get(
        i, fallback(it.text.style))})}) if it.text is not None and it.text.position == "auto" else it
        for i, it in enumerate(plan.items)]
    return plan.model_copy(update={"items": items})
