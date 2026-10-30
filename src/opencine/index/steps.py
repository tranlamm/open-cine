"""``Cache`` with hit/miss counting and atomic writes, shared by the ingest steps and sheets."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from opencine.core.cache import Cache


class StepCache:
    def __init__(self, root: str | Path, *, force: bool = False):
        self.cache = Cache(root)
        self.force = force
        self.hits = 0
        self.misses = 0

    def _lookup(self, step: str, key: str, suffix: str) -> Path | None:
        hit = None if self.force else self.cache.hit(step, key, suffix)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        return hit

    def file(self, step: str, key: str, suffix: str, make: Callable[[Path], Any]) -> Path:
        """Cached file ``<step>/<key><suffix>``; ``make(tmp_path)`` writes it on a miss."""
        if hit := self._lookup(step, key, suffix):
            return hit
        out = self.cache.path(step, key, suffix)
        tmp = out.with_name(f"{key}.tmp{suffix}")
        make(tmp)
        os.replace(tmp, out)
        return out

    def json(self, step: str, key: str, make: Callable[[], Any]) -> Any:
        """Cached JSON value; written only after ``make`` returns, so it doubles as a commit marker."""
        if hit := self._lookup(step, key, ".json"):
            return json.loads(hit.read_text("utf-8"))
        value = make()
        out = self.cache.path(step, key, ".json")
        tmp = out.with_name(f"{key}.tmp.json")
        tmp.write_text(json.dumps(value, sort_keys=True), "utf-8")
        os.replace(tmp, out)
        return value

    def dir(self, step: str, key: str) -> Path:
        """Scratch directory for a step's side files (emptied when ``force``)."""
        d = self.cache.root / step / key
        if self.force and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        return d


def place(src: Path, dst: Path) -> Path:
    """Expose a cached artifact at ``dst`` (hard link, else copy) unless it is already there."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return dst
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst
