"""Minimal Ollama JSON client implemented with the Python standard library."""

from __future__ import annotations

import json
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    """Raised when an Ollama-compatible request fails safely."""


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 30.0
    allow_remote: bool = False
    max_response_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("An Ollama model name is required.")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama base_url must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials embedded in Ollama URLs are not allowed.")
        if parsed.query or parsed.fragment:
            raise ValueError("Ollama base_url must not contain a query or fragment.")
        if not _is_loopback_host(parsed.hostname) and not self.allow_remote:
            raise ValueError(
                "Remote Ollama-compatible endpoints require explicit opt-in."
            )
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError("Ollama timeout must be between 0 and 120 seconds.")
        if not 1_024 <= self.max_response_bytes <= 10_000_000:
            raise ValueError(
                "Ollama max_response_bytes must be between 1024 and 10000000."
            )

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request one bounded non-streaming JSON response."""
        body = {
            "model": self.model,
            "stream": False,
            "format": json_schema or "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read(self.max_response_bytes + 1)
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(
                "The model endpoint timed out before returning a valid plan."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(
                "The model endpoint is unavailable or rejected the request."
            ) from exc

        if len(response_bytes) > self.max_response_bytes:
            raise OllamaError("The model endpoint response exceeded the size limit.")
        try:
            envelope = json.loads(response_bytes.decode("utf-8"))
            content = envelope["message"]["content"]
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaError(
                "The model endpoint returned an invalid structured response."
            ) from exc

        if not isinstance(payload, dict):
            raise OllamaError("The model endpoint response must be a JSON object.")
        return payload
