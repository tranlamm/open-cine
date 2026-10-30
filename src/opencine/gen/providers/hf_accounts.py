"""HF accounts the ``hf`` provider rotates through, each with a local ZeroGPU quota book.

Accounts are ``OPENCINE_HF_TOKENS`` in order (else ``HF_TOKEN``); each has its own quota (PRO:
40 min/day) and its own private duplicates, matched to ``OPENCINE_HF_SPACE_<ROLE>`` entries by
owner (user name, else an org). A single token with one entry per role is never looked up.
Identities come from ``whoami`` once per token and are cached in
``<gen_state_dir>/hf_accounts.json`` under a hash of the token, never the token itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from opencine.gen.base import QuotaExhausted
from opencine.gen.config import GenSettings, SpaceRef

WINDOW = timedelta(hours=24)


class QuotaBook:
    """Estimated ZeroGPU use in the 24 h window that starts at the first GPU use."""

    def __init__(self, path: Path, limit_s: float, now: Callable[[], datetime] = datetime.now):
        self.path, self.limit_s, self.now = path, limit_s, now

    def _load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        start = data.get("window_start")
        if start and self.now() >= datetime.fromisoformat(start) + WINDOW:
            data = {}
        return {"window_start": data.get("window_start"), "used_s": data.get("used_s", 0.0)}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), "utf-8")

    def status(self) -> dict[str, Any]:
        d = self._load()
        start = d["window_start"]
        reset = datetime.fromisoformat(start) + WINDOW if start else None
        return {"provider": "hf", "unit": "gpu_s", "limit": self.limit_s,
                "used": round(d["used_s"], 1), "left": round(max(self.limit_s - d["used_s"], 0), 1),
                "window_start": start, "reset_at": reset.isoformat(timespec="seconds") if reset else None}

    def check(self, reserve_s: float) -> None:
        st = self.status()
        if st["left"] < reserve_s:
            reset = datetime.fromisoformat(st["reset_at"]) if st["reset_at"] else None
            raise QuotaExhausted(f"HF quota (local book): {reserve_s:.0f}s reservation > "
                                 f"{st['left']:.0f}s left of {self.limit_s:.0f}s", reset_at=reset)

    def charge(self, seconds: float) -> None:
        d = self._load()
        d["window_start"] = d["window_start"] or self.now().isoformat(timespec="seconds")
        d["used_s"] += seconds
        self._save(d)

    def sync(self, left_s: float, reset_at: datetime | None, *, hub: bool = False) -> None:
        """Align with the server: a quota-exceeded message, or the Hub quota API (``hub``),
        where no reset time means the window has not started."""
        d = self._load()
        d["used_s"] = max(self.limit_s - left_s, 0.0)
        if reset_at:
            d["window_start"] = (reset_at - WINDOW).isoformat(timespec="seconds")
        elif hub:
            d["window_start"] = None
        self._save(d)


@dataclass(eq=False)
class Account:
    """One HF account: token, identity, quota book, and what this batch learnt about it."""

    token: str  # "" = no token (calls refuse; the book still reads)
    book: QuotaBook
    name: str = ""  # HF user name; "" = the only account, never looked up
    orgs: tuple[str, ...] = ()
    pro: bool | None = None
    refused: bool = False  # the server refused a call for quota (credits included)
    refused_until: datetime | None = None  # None while refused = until this batch ends
    hub_at: float | None = None  # monotonic time of the last Hub quota read
    hub_error: str | None = None  # why that read failed

    @property
    def label(self) -> str:
        return self.name or "default"

    def space(self, role: str, s: GenSettings) -> SpaceRef | None:
        """This account's duplicate for ``role``: its own entry, else an org's."""
        refs = s.hf_spaces(role)
        if not self.name:
            return refs[0] if refs else None
        return (next((r for r in refs if r.owner == self.name), None)
                or next((r for r in refs if r.owner in self.orgs), None))

    def serves(self, roles: Iterable[str], s: GenSettings) -> bool:
        return all(self.space(r, s) for r in roles)

    def out(self, now: datetime) -> bool:
        """Refused for quota and not back yet."""
        return self.refused and (self.refused_until is None or now < self.refused_until)

    def back_at(self) -> datetime | None:
        """When quota returns: the server's reset time, else the book's window end."""
        reset = self.book.status()["reset_at"]
        return self.refused_until or (datetime.fromisoformat(reset) if reset else None)


def token_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def identify(tokens: list[str], state_dir: Path, whoami: Callable[[str], dict[str, Any]],
             refresh: bool = False) -> list[dict[str, Any]]:
    """``{"name", "orgs", "pro"}`` of each token: from ``hf_accounts.json``, else ``whoami``."""
    path = state_dir / "hf_accounts.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}
    changed = False
    for t in tokens:
        if refresh or token_key(t) not in cache:
            me = whoami(t)
            cache[token_key(t)] = {"name": me["name"], "pro": bool(me.get("isPro")),
                                   "orgs": [o["name"] for o in me.get("orgs") or []]}
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2), "utf-8")
    return [cache[token_key(t)] for t in tokens]


def make_accounts(tokens: list[str], s: GenSettings, whoami: Callable[[str], dict[str, Any]],
                  now: Callable[[], datetime] = datetime.now, *, lookup: bool | None = None,
                  refresh: bool = False) -> list[Account]:
    """Accounts for ``tokens`` in order, one per HF user (a second token of the same user shares
    its quota and is dropped). Identities are looked up when ``lookup``; by default only when
    entries must be matched to owners (several tokens, or several entries for a role)."""
    from opencine.gen.providers.hf_space import SPACES

    tokens = list(dict.fromkeys(t for t in tokens if t))
    if not tokens:
        return [Account("", QuotaBook(s.gen_state_dir / "hf_quota.json", s.hf_quota_s, now))]
    if lookup is None:
        lookup = len(tokens) > 1 or any(len(s.hf_spaces(r)) > 1 for r in SPACES)
    ids = identify(tokens, s.gen_state_dir, whoami, refresh) if lookup else [{}] * len(tokens)
    out: list[Account] = []
    for token, me in zip(tokens, ids, strict=True):
        name = me.get("name", "")
        if name and any(a.name == name for a in out):
            continue
        book = QuotaBook(s.gen_state_dir / (f"hf_quota.{name}.json" if name else "hf_quota.json"),
                         s.hf_quota_s, now)
        out.append(Account(token, book, name, tuple(me.get("orgs", ())), me.get("pro")))
    return out
