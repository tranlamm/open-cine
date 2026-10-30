"""Content-addressed cache keyed by ``(source sha256, step, params, version)``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_memo: dict[tuple[str, int, int], str] = {}


def sha256_file(path: str | Path) -> str:
    """sha256 of a file, memoised in-process by (path, size, mtime)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _memo:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        _memo[key] = h.hexdigest()
    return _memo[key]


def cache_key(*parts: Any) -> str:
    """Stable 16-hex key from JSON-serialisable parts (dict keys sorted)."""
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Cache:
    """Directory of artifacts named ``<step>/<key><suffix>``; a non-empty file is a hit."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, step: str, key: str, suffix: str = "") -> Path:
        p = self.root / step / f"{key}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def hit(self, step: str, key: str, suffix: str = "") -> Path | None:
        p = self.root / step / f"{key}{suffix}"
        return p if p.exists() and p.stat().st_size > 0 else None
