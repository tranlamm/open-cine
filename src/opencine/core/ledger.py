"""Cost/quota ledger (``costs.jsonl``) and the per-project image budget for the agent.

Each spend appends ``{ts, project, provider, kind, units, usd_est, quota_note, ...extra}``.
Images shown to the agent are ``provider="claude", kind="image_view"`` (units = image tokens);
self-review sheets use a separate ``image_review`` pool so footage viewing cannot starve them.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from opencine.core.config import get_settings

if TYPE_CHECKING:
    from opencine.core.project import Project

IMAGE_VIEW = "image_view"
IMAGE_REVIEW = "image_review"


def image_tokens(width: int, height: int) -> int:
    """Image cost estimate: ``⌈w/28⌉ × ⌈h/28⌉`` tokens."""
    return math.ceil(width / 28) * math.ceil(height / 28)


class BudgetExceeded(RuntimeError):
    """A spend would push the project over its USD cap."""


class ImageBudgetExceeded(RuntimeError):
    """The agent asked for more images/tokens than the project budget allows."""


class Ledger:
    def __init__(self, project: Project):
        self.project = project
        self.path = project.costs_path

    def record(
        self,
        provider: str,
        kind: str,
        units: float,
        *,
        usd_est: float = 0.0,
        quota_note: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "project": self.project.pid,
            "provider": provider,
            "kind": kind,
            "units": units,
            "usd_est": usd_est,
            "quota_note": quota_note,
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    # ---- money ----------------------------------------------------------
    def total_usd(self) -> float:
        return round(sum(e.get("usd_est") or 0.0 for e in self.entries()), 6)

    def check_cap(self, add_usd: float, cap_usd: float | None = None) -> None:
        cap = get_settings().cost_cap_usd if cap_usd is None else cap_usd
        spent = self.total_usd()
        if spent + add_usd > cap + 1e-9:
            raise BudgetExceeded(
                f"project {self.project.pid}: ${spent:.2f} spent + ${add_usd:.2f} > cap ${cap:.2f}"
            )

    def summary(self) -> dict[str, Any]:
        by: dict[str, dict[str, float]] = {}
        for e in self.entries():
            row = by.setdefault(f"{e['provider']}:{e['kind']}", {"calls": 0, "units": 0.0, "usd_est": 0.0})
            row["calls"] += 1
            row["units"] += e.get("units") or 0
            row["usd_est"] += e.get("usd_est") or 0
        count, tokens = self.image_usage()
        r_count, r_tokens = self.image_usage(IMAGE_REVIEW)
        return {
            "project": self.project.pid,
            "total_usd": self.total_usd(),
            "by_provider_kind": by,
            "images": {"count": count, "tokens": tokens},
            "review_images": {"count": r_count, "tokens": r_tokens},
        }

    # ---- image budget ---------------------------------------------------
    def image_usage(self, kind: str = IMAGE_VIEW) -> tuple[int, int]:
        views = [e for e in self.entries() if e["kind"] == kind]
        return len(views), int(sum(e["units"] for e in views))

    def grant_image(self, path: str, tokens: int, *, max_count: int | None = None,
                    max_tokens: int | None = None) -> dict[str, Any]:
        """Record an image shown to the agent, or raise if it would exceed the budget."""
        return self.grant_images([(path, tokens)], max_count=max_count, max_tokens=max_tokens)[0]

    def grant_images(self, images: list[tuple[str, int]], *, max_count: int | None = None,
                     max_tokens: int | None = None, kind: str = IMAGE_VIEW) -> list[dict[str, Any]]:
        """All-or-nothing ``grant_image`` for ``(path, tokens)`` pairs; a refusal charges nothing.

        ``kind=IMAGE_REVIEW`` charges the self-review pool. An image already granted with the
        same path, mtime and size is free: its earlier entry comes back with ``"reused": True``."""
        seen = {(e.get("path"), e.get("mtime_ns"), e.get("size")): e for e in self.entries()
                if e["kind"] == kind and e.get("mtime_ns") is not None}
        stamps = [_stamp(path) for path, _ in images]
        reused = {i: seen[(path, *st)] for i, ((path, _), st) in enumerate(zip(images, stamps, strict=True))
                  if st is not None and (path, *st) in seen}
        fresh = [(i, img) for i, img in enumerate(images) if i not in reused]
        out = self._charge([img for _, img in fresh], [stamps[i] for i, _ in fresh],
                           max_count=max_count, max_tokens=max_tokens, kind=kind)
        by_index = dict(zip([i for i, _ in fresh], out, strict=True))
        return [by_index[i] if i in by_index else reused[i] | {"reused": True} for i in range(len(images))]

    def _charge(self, images: list[tuple[str, int]], stamps: list[tuple[int, int] | None], *,
                max_count: int | None, max_tokens: int | None, kind: str) -> list[dict[str, Any]]:
        if not images:
            return []
        s = get_settings()
        review = kind == IMAGE_REVIEW
        if max_count is None:
            max_count = s.image_review_count if review else s.image_budget_count
        if max_tokens is None:
            max_tokens = s.image_review_tokens if review else s.image_budget_tokens
        count, used = self.image_usage(kind)
        need = sum(t for _, t in images)
        if count + len(images) > max_count or used + need > max_tokens:
            what = f"{len(images)} image(s) need {need} tokens" if len(images) > 1 else f"this image needs {need}"
            hint = ("Fix what the last review sheet showed, or open the gate with the known issues."
                    if review else "Work from L0 text and the L1 overview first.")
            raise ImageBudgetExceeded(
                f"{'review image' if review else 'image'} budget: {count}/{max_count} images, "
                f"{used}/{max_tokens} tokens used; {what}. {hint}"
            )
        out = []
        for (path, tokens), st in zip(images, stamps, strict=True):
            count, used = count + 1, used + tokens
            stamp = {"mtime_ns": st[0], "size": st[1]} if st else {}
            out.append(self.record("claude", kind, tokens, path=path,
                                   quota_note=f"{count}/{max_count} img, {used}/{max_tokens} tok", **stamp))
        return out


def _stamp(path: str) -> tuple[int, int] | None:
    """(mtime_ns, size) of an image file, or None if it is not a file (then always charged)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size
