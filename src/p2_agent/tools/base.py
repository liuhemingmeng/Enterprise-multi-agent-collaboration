from __future__ import annotations

from typing import Protocol


class ToolProtocol(Protocol):
    """Contract every registered tool must satisfy."""

    name: str
    timeout_seconds: float
    cost_per_call: float

    def validate_params(self, params: dict) -> dict: ...
    def execute(self, params: dict) -> list: ...
