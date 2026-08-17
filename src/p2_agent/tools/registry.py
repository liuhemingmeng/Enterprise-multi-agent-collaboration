from __future__ import annotations

from threading import Lock

from p2_agent.tools.schemas import ToolCall, ToolErrorRecord, ToolResult


class CostBudget:
    """Track cumulative spending across tool calls within one task.

    If cumulative cost exceeds ``max_cost``, subsequent calls are rejected
    with ``budget_exceeded``.  This prevents runaway agents from burning
    API credits.
    """

    def __init__(self, max_cost: float = 1.0) -> None:
        self.max_cost = max_cost
        self._spent: dict[str, float] = {}
        self._lock = Lock()

    def can_spend(self, amount: float, task_id: str = "default") -> bool:
        with self._lock:
            current = self._spent.get(task_id, 0.0)
            return current + amount <= self.max_cost

    def charge(self, amount: float, task_id: str = "default") -> None:
        with self._lock:
            self._spent[task_id] = self._spent.get(task_id, 0.0) + amount

    def remaining(self, task_id: str = "default") -> float:
        with self._lock:
            return max(self.max_cost - self._spent.get(task_id, 0.0), 0.0)


class ToolRegistry:
    """Whitelist-enforcing tool catalogue.

    Only tools explicitly registered here can be invoked.  Every call goes
    through ``call()`` which enforces (in order):

    1. whitelist check
    2. parameter validation
    3. timeout guard
    4. cost budget
    5. execution

    Any failure is archived via ``error_archive`` and returned as a
    ``ToolResult`` with ``success=False`` — the graph never crashes.
    """

    def __init__(self, budget: CostBudget | None = None) -> None:
        self._tools: dict[str, object] = {}
        self.budget = budget or CostBudget()
        self.error_archive: ErrorArchive | None = None

    def register(self, tool: object) -> None:
        name = getattr(tool, "name", None)
        if not name:
            raise ValueError("tool must have a non-empty 'name'")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        for attr in ("timeout_seconds", "cost_per_call", "validate_params", "execute"):
            if not hasattr(tool, attr):
                raise ValueError(f"tool '{name}' missing required attribute: {attr}")
        self._tools[name] = tool

    def is_allowed(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> object:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def call(self, call: ToolCall, task_id: str = "default") -> ToolResult:
        import time

        tool_name = call.tool_name
        tool = self._tools.get(tool_name)

        # 1. whitelist
        if tool is None:
            self._archive_error(call, "not_whitelisted", f"tool not in whitelist: {tool_name}")
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"tool not in whitelist: {tool_name}",
            )

        # 2. param validation
        try:
            validated = tool.validate_params(call.params)
        except (ValueError, TypeError) as exc:
            self._archive_error(call, "param_validation", str(exc))
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"param validation failed: {exc}",
            )

        # 3. budget check
        if not self.budget.can_spend(tool.cost_per_call, task_id):
            remaining = self.budget.remaining(task_id)
            msg = (
                f"budget exceeded: need {tool.cost_per_call}, "
                f"remaining {remaining:.2f}"
            )
            self._archive_error(call, "budget_exceeded", msg)
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=False,
                error=msg,
            )

        # 4. timeout + execution
        start = time.monotonic()
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.execute, validated)
                data = future.result(timeout=tool.timeout_seconds)
            elapsed = (time.monotonic() - start) * 1000
            self.budget.charge(tool.cost_per_call, task_id)
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=True,
                data=data if isinstance(data, list) else [data],
                cost=tool.cost_per_call,
                duration_ms=round(elapsed, 2),
            )
        except TimeoutError:
            self._archive_error(call, "timeout", f"timed out after {tool.timeout_seconds}s")
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"timed out after {tool.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as exc:
            self._archive_error(call, "execution_error", str(exc))
            return ToolResult(
                call_id=call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"execution error: {exc}",
            )

    def _archive_error(self, call: ToolCall, error_type: str, message: str) -> None:
        if self.error_archive is None:
            return
        from p2_agent.tools.schemas import ToolErrorRecord

        record = ToolErrorRecord(
            call_id=call.call_id,
            tool_name=call.tool_name,
            error_type=error_type,
            error_message=message,
            params=call.params,
            caller=call.caller,
        )
        self.error_archive.save(record)


class ErrorArchive:
    """SQLite table for persisting tool-call errors for audit."""

    def __init__(self, path: str = "data/p2_errors.sqlite3") -> None:
        from pathlib import Path

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_errors (
                    error_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    caller TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save(self, record: ToolErrorRecord) -> None:
        import json

        payload = json.dumps(record.params, ensure_ascii=False, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_errors
                    (error_id, call_id, tool_name, error_type, error_message,
                     params_json, caller)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.error_id),
                    str(record.call_id),
                    record.tool_name,
                    record.error_type,
                    record.error_message,
                    payload,
                    record.caller,
                ),
            )
            conn.commit()

    def list_errors(
        self, tool_name: str | None = None, limit: int = 100
    ) -> list[dict]:
        with self._connect() as conn:
            if tool_name:
                rows = conn.execute(
                    """
                    SELECT * FROM tool_errors
                    WHERE tool_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tool_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM tool_errors
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]
