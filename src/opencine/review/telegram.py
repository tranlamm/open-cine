"""Telegram Bot API (optional): post text, stills, clips and finals to the user's chat.

On when ``OPENCINE_TELEGRAM_BOT_TOKEN`` and ``OPENCINE_TELEGRAM_CHAT_ID`` are set. Photos and
videos go as albums of up to 10, other files as documents. The public Bot API takes uploads
≤ 50 MB (photos ≤ 10 MB); ``OPENCINE_TELEGRAM_API_URL`` can point at a local Bot API server.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from opencine.core.config import get_settings

API_URL = "https://api.telegram.org"
ALBUM_MAX = 10
TEXT_MAX = 4096
PHOTO_MAX = 10 * 2**20
UPLOAD_MAX = 50 * 2**20
LOCAL_UPLOAD_MAX = 2000 * 2**20
THUMB_EDGE = 320  # Bot API thumbnail: JPEG ≤ 320 px, ≤ 200 kB
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}


class TelegramError(RuntimeError):
    pass


def configured() -> bool:
    s = get_settings()
    return bool(s.telegram_bot_token and s.telegram_chat_id)


class Bot:
    def __init__(self, token: str, base_url: str = API_URL, transport: httpx.BaseTransport | None = None):
        self.token = token
        self.local = base_url.rstrip("/") != API_URL
        self.base = f"{base_url.rstrip('/')}/bot{token}"
        self.http = httpx.Client(timeout=3600.0 if self.local else 300.0, transport=transport)

    def call(self, method: str, data: dict[str, Any] | None = None,
             files: dict[str, Path] | None = None) -> Any:
        """One Bot API call; waits out a 429 ``retry_after``. Errors never carry the token."""
        for _ in range(4):
            handles = {k: (p.name, p.open("rb")) for k, p in (files or {}).items()}
            try:
                r = self.http.post(f"{self.base}/{method}", data=data, files=handles or None)
            except httpx.HTTPError as e:
                raise TelegramError(f"{method}: {str(e).replace(self.token, '<token>')}") from None
            finally:
                for _, f in handles.values():
                    f.close()
            try:
                body = r.json()
            except ValueError:
                raise TelegramError(f"{method}: HTTP {r.status_code}") from None
            if body.get("ok"):
                return body["result"]
            wait = (body.get("parameters") or {}).get("retry_after")
            if r.status_code == 429 and wait:
                time.sleep(float(wait))
                continue
            raise TelegramError(f"{method}: {body.get('error_code')} {body.get('description')}")
        raise TelegramError(f"{method}: still rate-limited")


def split_text(text: str, size: int = TEXT_MAX) -> list[str]:
    """``text`` in pieces of ≤ ``size`` chars, cut at line ends where possible."""
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size and cur:
            out.append(cur)
            cur = ""
        while len(line) > size:
            out.append(line[:size])
            line = line[size:]
        cur += line
    return [c for c in [*out, cur] if c.strip()]


def album_pages(n: int) -> list[int]:
    """``n`` items as balanced albums of ≤ 10 (never a lone last item when n ≥ 2)."""
    if n <= 0:
        return []
    pages = math.ceil(n / ALBUM_MAX)
    base, extra = divmod(n, pages)
    return [base + (i < extra) for i in range(pages)]


def _video_fields(path: Path, tmp: Path, key: str, files: dict[str, Path]) -> dict[str, Any]:
    """Display size, duration and a thumbnail: without them Telegram plays a video in a square frame."""
    from opencine.core.ffmpeg import probe_json, run
    from opencine.ingest.probe import stream_rotation

    try:
        info = probe_json(path)
    except Exception:  # noqa: BLE001 — sent without them
        return {}
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v or not v.get("width") or not v.get("height"):
        return {}
    w, h = int(v["width"]), int(v["height"])
    if stream_rotation(v) in (90, 270):
        w, h = h, w
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    out: dict[str, Any] = {"width": w, "height": h, "duration": max(1, round(dur))}
    thumb = tmp / f"{key}.jpg"
    try:
        run(["-y", "-ss", f"{min(1.0, dur / 2):.2f}", "-i", str(path), "-frames:v", "1", "-vf",
             f"scale={THUMB_EDGE}:{THUMB_EDGE}:force_original_aspect_ratio=decrease", "-q:v", "5", str(thumb)])
    except Exception:  # noqa: BLE001 — Telegram makes its own
        return out
    if thumb.is_file() and thumb.stat().st_size <= 200 * 1024:
        files[key] = thumb
        out["thumbnail"] = f"attach://{key}"
    return out


def send(text: str, paths: list[Path], *, bot: Bot | None = None) -> dict[str, Any]:
    """Post ``text``, then the files in order. Several files are captioned 1, 2, 3… (the review
    item numbers); a file over the upload cap is skipped with its reason."""
    s = get_settings()
    if bot is None:
        if not configured():
            raise TelegramError("Telegram is off: set OPENCINE_TELEGRAM_BOT_TOKEN and "
                                "OPENCINE_TELEGRAM_CHAT_ID (opencine setup --telegram)")
        bot = Bot(s.telegram_bot_token, s.telegram_api_url)
    chat = s.telegram_chat_id
    cap = LOCAL_UPLOAD_MAX if bot.local else UPLOAD_MAX
    media, docs, skipped = [], [], []
    for n, p in enumerate(paths, 1):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        size, ext = p.stat().st_size, p.suffix.lower()
        if size > cap:
            skipped.append({"path": str(p), "reason": f"{size / 2**20:.0f} MB > {cap // 2**20} MB"})
            continue
        kind = "photo" if ext in PHOTO_EXTS and size <= PHOTO_MAX else "video" if ext in VIDEO_EXTS else None
        (media if kind else docs).append((n, kind or "document", p))
    numbered = len(paths) > 1
    ids: list[int] = []
    for chunk in split_text(text):
        ids.append(bot.call("sendMessage", {"chat_id": chat, "text": chunk})["message_id"])

    with tempfile.TemporaryDirectory() as tmp:
        start = 0
        for size in album_pages(len(media)):
            page, start = media[start:start + size], start + size
            files: dict[str, Path] = {}
            items = []
            for i, (n, kind, p) in enumerate(page):
                files[f"f{i}"] = p
                item = {"type": kind, "media": f"attach://f{i}"} | ({"caption": str(n)} if numbered else {})
                if kind == "video":
                    item |= _video_fields(p, Path(tmp), f"t{i}", files)
                items.append(item)
            if len(items) == 1:  # an album needs 2–10 items
                n, kind, p = page[0]
                fields = {k: str(v) for k, v in items[0].items() if k not in ("type", "media")}
                files[kind] = files.pop("f0")
                res = [bot.call(f"send{kind.title()}", {"chat_id": chat, **fields}, files)]
            else:
                res = bot.call("sendMediaGroup", {"chat_id": chat, "media": json.dumps(items)}, files)
            ids += [m["message_id"] for m in res]
        for n, _, p in docs:
            data = {"chat_id": chat} | ({"caption": str(n)} if numbered else {})
            ids.append(bot.call("sendDocument", data, {"document": p})["message_id"])
    return {"sent": len(ids), "message_ids": ids, "skipped": skipped}


def latest_chat(bot: Bot) -> str | None:
    """The chat of the newest message the bot received (``getUpdates``), for setup."""
    for u in reversed(bot.call("getUpdates", {"timeout": "0"}) or []):
        msg = u.get("message") or u.get("channel_post") or {}
        if msg.get("chat", {}).get("id") is not None:
            return str(msg["chat"]["id"])
    return None
