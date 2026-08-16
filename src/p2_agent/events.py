from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import UUID


class ProgressEventStore:
    """Thread-safe in-process event buffer; Redis Streams replaces it later."""

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = Lock()

    def append(self, task_id: UUID | str, event: str, **payload: Any) -> dict[str, Any]:
        item = {
            "task_id": str(task_id),
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
        with self._lock:
            self._events.setdefault(str(task_id), []).append(item)
        return item

    def list(self, task_id: UUID | str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(str(task_id), []))
