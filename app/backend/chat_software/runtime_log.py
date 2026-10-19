"""Structured runtime events shared by the UI stream and server terminal."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping


_LOGGER = logging.getLogger("uvicorn.error")
_SENSITIVE_TEXT_FIELDS = {"content", "message", "old_text", "new_text", "response"}


def compact_runtime_value(value: Any) -> Any:
    """Keep runtime events useful without dumping whole file contents."""

    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SENSITIVE_TEXT_FIELDS and isinstance(item, str):
                compact[key] = f"<{len(item.encode('utf-8'))} bytes>"
            else:
                compact[key] = compact_runtime_value(item)
        return compact
    if isinstance(value, list):
        items = [compact_runtime_value(item) for item in value[:30]]
        if len(value) > 30:
            items.append({"truncated_count": len(value) - 30})
        return items
    if isinstance(value, str) and len(value) > 2_000:
        return value[:2_000] + f"... [{len(value) - 2_000} more chars]"
    return value


def record_runtime_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Timestamp and print a structured event, returning the UI-safe payload."""

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **dict(event),
    }
    _LOGGER.info(
        "[runtime] %s",
        json.dumps(compact_runtime_value(payload), default=str, ensure_ascii=False),
    )
    return payload
