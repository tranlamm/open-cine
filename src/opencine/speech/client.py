"""HTTP clients for self-hosted speech gateways (STT and TTS) that implement the API below.

STT ``POST /v1/audio/transcriptions``: raw media body (not multipart), ``?language&timestamps=true
&timeline_offset_s``; returns ``{text, language, duration, segments[{start, end, text, words?}]}``.
TTS ``POST /v1/audio/speech/file`` → one WAV; ``/v1/audio/speech/batch`` ``{"items": [...]}`` →
ZIP of ``0000.wav``, ``0001.wav``… in input order. Both use ``Authorization: Bearer``; 429/503
``Retry-After`` is honoured; errors are ``{"error": …}`` or FastAPI ``{"detail": …}``.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from opencine.core.config import get_settings

RETRY_STATUSES = {429, 503}
MAX_RETRY_WAIT_S = 60.0


class SpeechError(RuntimeError):
    pass


class SpeechUnavailable(SpeechError):
    """Not configured, or the gateway does not answer (e.g. its GPU instance is stopped)."""


class _Gateway:
    name = ""

    def __init__(self, url: str | None, token: str | None, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 900.0, retries: int = 4, sleep: Callable[[float], None] = time.sleep):
        if not url:
            raise SpeechUnavailable(f"{self.name} is not configured (set OPENCINE_{self.name.upper()}_URL)")
        self.url, self.token = url.rstrip("/"), token or ""
        self.http = httpx.Client(transport=transport, timeout=httpx.Timeout(timeout, connect=10.0))
        self.retries, self.sleep = retries, sleep

    def _post(self, path: str, **kw) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.token}", **kw.pop("headers", {})}
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(f"{self.url}{path}", headers=headers, **kw)
            except httpx.TransportError as e:
                raise SpeechUnavailable(f"{self.name} at {self.url} does not answer: {e}") from e
            if r.status_code in RETRY_STATUSES and attempt < self.retries:
                self.sleep(min(float(r.headers.get("Retry-After") or 2 ** attempt), MAX_RETRY_WAIT_S))
                if hasattr(kw.get("content"), "seek"):
                    kw["content"].seek(0)
                continue
            if r.status_code >= 400:
                raise SpeechError(f"{self.name} {path}: HTTP {r.status_code} {_reason(r)}")
            return r
        raise AssertionError("unreachable")

    def status(self, timeout: float = 5.0) -> dict[str, Any]:
        """``{url, healthy, ready, detail}`` without raising (for `opencine speech status`)."""
        out: dict[str, Any] = {"url": self.url, "healthy": False, "ready": False}
        try:
            out["healthy"] = self.http.get(f"{self.url}/healthz", timeout=timeout).status_code == 200
            r = self.http.get(f"{self.url}/readyz", timeout=timeout,
                              headers={"Authorization": f"Bearer {self.token}"})
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            out["ready"] = r.status_code == 200 and bool(body.get("ready", True))
            out["detail"] = body
        except httpx.HTTPError as e:
            out["detail"] = f"{type(e).__name__}: {e}"
        return out

    def require_up(self) -> None:
        if not (st := self.status())["healthy"]:
            raise SpeechUnavailable(f"{self.name} at {self.url} does not answer ({st.get('detail')}); "
                                    "is the GPU instance running?")


def _reason(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:300]
    return str(body.get("error") or body.get("detail") or body)[:300]


class SttClient(_Gateway):
    name = "stt"

    @classmethod
    def from_settings(cls, **kw) -> SttClient:
        s = get_settings()
        return cls(s.stt_url, s.stt_token, **kw)

    def transcribe(self, audio: Path, *, language: str | None = None, offset_s: float = 0.0,
                   context: str | None = None, timestamps: bool = True) -> dict[str, Any]:
        """``timestamps`` requests word times (the gateway's CTC alignment)."""
        params: dict[str, Any] = {"timestamps": str(timestamps).lower(), "timeline_offset_s": offset_s}
        if language:
            params["language"] = language
        if context:
            params["context"] = context[:2000]
        with Path(audio).open("rb") as f:
            r = self._post("/v1/audio/transcriptions", params=params, content=f,
                           headers={"Content-Type": "application/octet-stream"})
        return r.json()


class TtsClient(_Gateway):
    name = "tts"

    @classmethod
    def from_settings(cls, **kw) -> TtsClient:
        s = get_settings()
        return cls(s.tts_url, s.tts_token, **kw)

    def speech(self, item: dict[str, Any]) -> bytes:
        """One complete WAV for ``{text, language?, sample_rate_hz?}``."""
        return self._post("/v1/audio/speech/file", json=_request(item)).content

    def batch(self, items: list[dict[str, Any]]) -> list[bytes]:
        """WAVs in input order (one ZIP request)."""
        r = self._post("/v1/audio/speech/batch", json={"items": [_request(i) for i in items]})
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".wav"))
            if len(names) != len(items):
                raise SpeechError(f"tts batch returned {len(names)} files for {len(items)} items")
            return [z.read(n) for n in names]


def _request(item: dict[str, Any]) -> dict[str, Any]:
    return {"text": item["text"], "voice": "default",
            "language": item.get("language") or get_settings().speech_language,
            "response_format": "wav", "stream": False,
            "sample_rate_hz": int(item.get("sample_rate_hz") or 48000)}
