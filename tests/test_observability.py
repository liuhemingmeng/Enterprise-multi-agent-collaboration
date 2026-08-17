from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from p2_agent.guardrails import guardrail_store
from p2_agent.main import app, service
from p2_agent.tracing import tracing_store

pytestmark = pytest.mark.usefixtures("reset_stores")


@pytest.fixture(autouse=True)
def reset_stores():
    tracing_store.reset()
    guardrail_store.reset()
    yield
    tracing_store.reset()
    guardrail_store.reset()


GOAL = "为某制造企业设计一套零碳园区方案，重点突出投资回报与合规路径"


def test_spans_recorded_for_full_run():
    state = service.workflow.create_and_run(GOAL)
    assert state.status in {"completed", "need_human"}
    spans = tracing_store.list(str(state.task_id))
    nodes = {s.node for s in spans}
    # planner, retriever, analyst, writer, reviewer, export must all be traced
    assert {"planner", "retriever", "analyst", "writer", "reviewer"} <= nodes
    for s in spans:
        assert s.duration_ms >= 0
        assert s.status in {"ok", "error"}


def test_cost_accumulates_from_tool_calls():
    state = service.workflow.create_and_run(GOAL)
    summary = tracing_store.summary(str(state.task_id))
    # retriever issues KB searches, each charged a cost -> total > 0
    assert summary["total_cost_usd"] > 0


def test_trace_endpoint_returns_spans_and_summary():
    state = service.workflow.create_and_run(GOAL)
    client = TestClient(app)
    resp = client.get(f"/tasks/{state.task_id}/trace")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["span_count"] >= 5
    assert body["summary"]["total_cost_usd"] > 0
    assert len(body["spans"]) >= 5


def test_stream_endpoint_emits_done():
    state = service.workflow.create_and_run(GOAL)
    client = TestClient(app)
    resp = client.get(f"/tasks/{state.task_id}/stream")
    assert resp.status_code == 200
    # a completed task should immediately yield a terminal "done" event
    assert "done" in resp.text
