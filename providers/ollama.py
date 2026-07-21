"""Minimal Ollama JSON client implemented with the Python standard library."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    """Raised when a local Ollama request fails or returns invalid data."""


@dataclass(frozen=True)
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("An Ollama model name is required.")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must use http:// or https://.")

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request one non-streaming JSON response from the local model."""
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
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_response = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(
                f"Local model {self.model!r} did not respond within "
                f"{self.timeout_seconds:g} seconds."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(
                "Cannot reach local Ollama. Start Ollama and confirm the model "
                f"{self.model!r} is installed."
            ) from exc

        try:
            envelope = json.loads(raw_response)
            content = envelope["message"]["content"]
            payload = json.loads(content)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an invalid JSON response.") from exc

        if not isinstance(payload, dict):
            raise OllamaError("Ollama JSON response must be an object.")
        return payload
