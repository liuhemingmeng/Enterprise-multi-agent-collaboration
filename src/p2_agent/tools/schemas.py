from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A request to invoke a registered tool."""

    model_config = ConfigDict(extra="forbid")

    call_id: UUID = Field(default_factory=uuid4)
    tool_name: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    caller: str = Field(default="retriever", max_length=64)


class ToolResult(BaseModel):
    """The outcome of a tool invocation."""

    model_config = ConfigDict(extra="forbid")

    call_id: UUID
    tool_name: str
    success: bool
    data: list[Any] = Field(default_factory=list)
    error: str | None = None
    cost: float = Field(default=0.0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)
    timed_out: bool = False


class ToolErrorRecord(BaseModel):
    """A persisted error for audit purposes."""

    model_config = ConfigDict(extra="forbid")

    error_id: UUID = Field(default_factory=uuid4)
    call_id: UUID
    tool_name: str
    error_type: Literal[
        "not_whitelisted",
        "param_validation",
        "timeout",
        "budget_exceeded",
        "execution_error",
    ]
    error_message: str
    params: dict[str, Any] = Field(default_factory=dict)
    caller: str = "unknown"
