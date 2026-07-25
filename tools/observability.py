"""Structured, value-free runtime event logging."""

from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger("mas_ds")


def log_event(event: str, **metadata: Any) -> None:
    """Emit one stable JSON event without dataframe or prompt content."""
    payload = {"event": event, **metadata}
    LOGGER.info(json.dumps(payload, sort_keys=True, default=str))
