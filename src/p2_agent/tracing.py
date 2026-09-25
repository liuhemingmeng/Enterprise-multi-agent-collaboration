from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from p2_agent.llm import reset_usage_scope, take_usage_scope


class Span(BaseModel):
    """One executed graph node. The basis of the execution trace / timeline.

    Designed to map cleanly onto OpenTelemetry spans later: ``node`` is the
    span name, ``started_at``/``ended_at`` are timestamps, ``duration_ms`` is
    the duration, and ``tokens``/``cost_usd`` carry the LLM economics that a
    production evaluator cares about.  Under the deterministic stub the token
    and cost figures are 0 / tool-only, but the *schema* is production-shaped
    so swapping in a real LLM requires no migration.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    node: str
    status: Literal["ok", "error", "skipped"] = "ok"
    started_at: str
    ended_at: str
    duration_ms: float
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class TracingStore:
    """In-process span buffer keyed by task_id.

    This is the MVP equivalent of an OpenTelemetry collector / LangSmith
    trace backend.  It is intentionally memory-only: spans live for the life
    of the process and are aggregated on read.  Swapping in a real backend
    (OTel exporter, PostgreSQL) is a drop-in change behind this interface.
    """

    def __init__(self) -> None:
        self._spans: dict[str, list[Span]] = {}
        self._costs: dict[str, float] = {}
        self._lock = Lock()

    def add(self, span: Span) -> None:
        with self._lock:
            self._spans.setdefault(span.task_id, []).append(span)

    def record_tool_cost(self, task_id: str, amount: float) -> None:
        """Accumulate tool-call cost (e.g. KB search) per task."""
        with self._lock:
            self._costs[str(task_id)] = self._costs.get(str(task_id), 0.0) + amount

    def list(self, task_id: str) -> list[Span]:
        with self._lock:
            return list(self._spans.get(str(task_id), []))

    def summary(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            spans = list(self._spans.get(str(task_id), []))
            cost = self._costs.get(str(task_id), 0.0)
        per_node: dict[str, dict[str, Any]] = {}
        total_ms = 0.0
        total_tokens = 0
        llm_cost = 0.0
        for s in spans:
            total_ms += s.duration_ms
            total_tokens += s.tokens
            llm_cost += s.cost_usd
            bucket = per_node.setdefault(
                s.node,
                {
                    "node": s.node,
                    "count": 0,
                    "total_ms": 0.0,
                    "errors": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["total_ms"] = round(bucket["total_ms"] + s.duration_ms, 2)
            bucket["tokens"] += s.tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + s.cost_usd, 8)
            if s.status == "error":
                bucket["errors"] += 1
        return {
            "task_id": str(task_id),
            "span_count": len(spans),
            "total_duration_ms": round(total_ms, 2),
            "total_tokens": total_tokens,
            "tool_cost_usd": round(cost, 6),
            "llm_cost_usd": round(llm_cost, 6),
            "total_cost_usd": round(cost + llm_cost, 6),
            "per_node": list(per_node.values()),
        }

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()
            self._costs.clear()


tracing_store = TracingStore()


def instrumented(node_name: str, fn: Callable) -> Callable:
    """Wrap a graph node to record a :class:`Span` on every invocation.

    The span is written in *both* success and failure paths (the failure path
    re-raises so the graph's normal error handling is untouched).  The node's
    own return value is passed through unchanged — instrumentation is purely
    observational and never mutates workflow state.
    """

    def wrapper(state: Any) -> dict:
        start = time.perf_counter()
        started = datetime.now(UTC).isoformat()
        reset_usage_scope()
        try:
            out = fn(state)
        except Exception as exc:  # noqa: BLE001 - record then propagate
            ended = datetime.now(UTC).isoformat()
            usage = take_usage_scope()
            tracing_store.add(
                Span(
                    task_id=str(state.task_id),
                    node=node_name,
                    status="error",
                    started_at=started,
                    ended_at=ended,
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    # Tokens already spent before the failure are real spend and
                    # must stay visible — a node that burns budget then throws is
                    # exactly the case a cost dashboard has to surface.
                    tokens=usage.total_tokens,
                    cost_usd=usage.cost_usd,
                    error=str(exc),
                )
            )
            raise
        ended = datetime.now(UTC).isoformat()
        usage = take_usage_scope()
        tracing_store.add(
            Span(
                task_id=str(state.task_id),
                node=node_name,
                status="ok",
                started_at=started,
                ended_at=ended,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                tokens=usage.total_tokens,
                cost_usd=usage.cost_usd,
            )
        )
        return out if isinstance(out, dict) else {}

    return wrapper
