"""Stage 5 tests: tool whitelist, parameter validation, timeout, budget, error archive."""
from __future__ import annotations

import time

from p2_agent.graph.workflow import build_workflow
from p2_agent.schemas import WorkflowState
from p2_agent.service import create_tool_registry
from p2_agent.tools.kb_search import KBSearchTool
from p2_agent.tools.registry import CostBudget, ErrorArchive, ToolRegistry
from p2_agent.tools.schemas import ToolCall

# ── Whitelist enforcement ──────────────────────────────────────────


def test_registry_rejects_unregistered_tool():
    registry = ToolRegistry(CostBudget(max_cost=100))
    call = ToolCall(tool_name="shell_exec", params={"cmd": "rm -rf /"})
    result = registry.call(call)
    assert not result.success
    assert "not in whitelist" in result.error


def test_registry_rejects_duplicate_registration():
    registry = ToolRegistry()
    registry.register(KBSearchTool())
    try:
        registry.register(KBSearchTool())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration must fail")


def test_registry_rejects_tool_missing_interface():
    class BadTool:
        name = "bad_tool"

    registry = ToolRegistry()
    try:
        registry.register(BadTool())
    except ValueError as exc:
        assert "missing required attribute" in str(exc)
    else:
        raise AssertionError("incomplete tool must be rejected")


# ── Parameter validation ────────────────────────────────────────────


def test_kb_search_rejects_blank_query():
    tool = KBSearchTool()
    try:
        tool.validate_params({"query": "", "top_k": 3})
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("blank query must be rejected")


def test_kb_search_rejects_oversized_query():
    tool = KBSearchTool()
    try:
        tool.validate_params({"query": "x" * 501, "top_k": 3})
    except ValueError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("oversized query must be rejected")


def test_kb_search_rejects_top_k_out_of_range():
    tool = KBSearchTool()
    try:
        tool.validate_params({"query": "valid", "top_k": 0})
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("top_k=0 must be rejected")

    try:
        tool.validate_params({"query": "valid", "top_k": 11})
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("top_k=11 must be rejected")


def test_kb_search_rejects_non_integer_top_k():
    tool = KBSearchTool()
    try:
        tool.validate_params({"query": "valid", "top_k": "three"})
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("non-integer top_k must be rejected")


# ── Successful tool call ───────────────────────────────────────────


def test_kb_search_success_returns_evidence():
    registry = ToolRegistry(CostBudget(max_cost=100))
    registry.register(KBSearchTool())
    call = ToolCall(tool_name="kb_search", params={"query": "预测性维护", "top_k": 2})
    result = registry.call(call)
    assert result.success
    assert len(result.data) == 2
    assert result.cost == 0.02
    assert result.duration_ms >= 0


# ── Cost budget enforcement ───────────────────────────────────────


def test_budget_exceeded_rejects_subsequent_calls():
    budget = CostBudget(max_cost=0.03)
    registry = ToolRegistry(budget=budget)
    registry.register(KBSearchTool())

    # First call succeeds (cost 0.02, total 0.02)
    call1 = ToolCall(tool_name="kb_search", params={"query": "test1", "top_k": 1})
    result1 = registry.call(call1)
    assert result1.success

    # Second call succeeds (cost 0.02, total 0.04 > 0.03? no, 0.04 <= 0.03 is false)
    # Actually 0.02 + 0.02 = 0.04 > 0.03, so second should be rejected
    call2 = ToolCall(tool_name="kb_search", params={"query": "test2", "top_k": 1})
    result2 = registry.call(call2)
    assert not result2.success
    assert "budget exceeded" in result2.error


def test_budget_remaining_decreases_after_charge():
    budget = CostBudget(max_cost=1.0)
    assert budget.remaining() == 1.0
    budget.charge(0.3)
    assert budget.remaining() == 0.7


# ── Error archive ──────────────────────────────────────────────────


def test_error_archive_persists_errors(tmp_path):
    archive = ErrorArchive(str(tmp_path / "errors.sqlite3"))
    registry = ToolRegistry(CostBudget(max_cost=100))
    registry.error_archive = archive
    registry.register(KBSearchTool())

    # Trigger a whitelist rejection
    bad_call = ToolCall(tool_name="dangerous_tool", params={})
    registry.call(bad_call)

    errors = archive.list_errors()
    assert len(errors) == 1
    assert errors[0]["tool_name"] == "dangerous_tool"
    assert errors[0]["error_type"] == "not_whitelisted"


def test_error_archive_filters_by_tool_name(tmp_path):
    archive = ErrorArchive(str(tmp_path / "errors.sqlite3"))
    registry = ToolRegistry(CostBudget(max_cost=100))
    registry.error_archive = archive
    registry.register(KBSearchTool())

    # Trigger whitelist rejection
    registry.call(ToolCall(tool_name="unknown_tool", params={}))
    # Trigger param validation error
    registry.call(ToolCall(tool_name="kb_search", params={"query": ""}))

    all_errors = archive.list_errors()
    assert len(all_errors) == 2

    filtered = archive.list_errors(tool_name="kb_search")
    assert len(filtered) == 1
    assert filtered[0]["error_type"] == "param_validation"


def test_error_archive_includes_params_snapshot(tmp_path):
    archive = ErrorArchive(str(tmp_path / "errors.sqlite3"))
    registry = ToolRegistry(CostBudget(max_cost=100))
    registry.error_archive = archive
    registry.register(KBSearchTool())

    registry.call(
        ToolCall(
            tool_name="kb_search",
            params={"query": "test", "top_k": 99},
        )
    )
    errors = archive.list_errors()
    assert len(errors) == 1
    import json

    params = json.loads(errors[0]["params_json"])
    assert params["query"] == "test"


# ── End-to-end workflow with registry ──────────────────────────────


def test_workflow_completes_with_tool_registry(tmp_path):
    from p2_agent.persistence import SQLiteStateStore

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    registry = create_tool_registry(store.path)
    graph = build_workflow(registry)
    result = graph.invoke(WorkflowState(user_goal="预测性维护方案设计"))
    assert result["status"] == "completed"
    assert result["evidence"]


def test_backward_compatible_run_task_without_registry():
    """Stages 1–4 path: build_workflow() with no arg uses direct KB."""
    state = build_workflow().invoke(WorkflowState(user_goal="向后兼容测试"))
    assert state["status"] == "completed"


# ── Timeout simulation ─────────────────────────────────────────────


def test_timeout_rejected_and_archived(tmp_path):
    """A tool that sleeps beyond its timeout returns a timeout result."""
    archive = ErrorArchive(str(tmp_path / "errors.sqlite3"))
    budget = CostBudget(max_cost=100)
    registry = ToolRegistry(budget=budget)
    registry.error_archive = archive

    class SlowTool:
        name = "slow_tool"
        timeout_seconds = 0.1
        cost_per_call = 0.01

        def validate_params(self, params: dict) -> dict:
            return params

        def execute(self, params: dict) -> list:
            time.sleep(0.5)
            return []

    registry.register(SlowTool())
    call = ToolCall(tool_name="slow_tool", params={})
    result = registry.call(call)
    assert not result.success
    assert result.timed_out
    assert "timed out" in result.error

    errors = archive.list_errors(tool_name="slow_tool")
    assert len(errors) == 1
    assert errors[0]["error_type"] == "timeout"


def test_execution_error_archived(tmp_path):
    """A tool that raises during execution is caught and archived."""
    archive = ErrorArchive(str(tmp_path / "errors.sqlite3"))
    registry = ToolRegistry(CostBudget(max_cost=100))
    registry.error_archive = archive

    class CrashingTool:
        name = "crash_tool"
        timeout_seconds = 5.0
        cost_per_call = 0.01

        def validate_params(self, params: dict) -> dict:
            return params

        def execute(self, params: dict) -> list:
            raise RuntimeError("simulated crash")

    registry.register(CrashingTool())
    call = ToolCall(tool_name="crash_tool", params={})
    result = registry.call(call)
    assert not result.success
    assert "execution error" in result.error

    errors = archive.list_errors(tool_name="crash_tool")
    assert len(errors) == 1
    assert errors[0]["error_type"] == "execution_error"
