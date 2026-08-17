import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from p2_agent.async_service import AsyncWorkflowService
from p2_agent.eval.dataset import build_evaluation_set, save_dataset
from p2_agent.eval.runner import run_comparison
from p2_agent.guardrails import guardrail_store
from p2_agent.tracing import tracing_store

service = AsyncWorkflowService()

app = FastAPI(title="P2 Agent Workbench", version="0.1.0")


class TaskRequest(BaseModel):
    user_goal: str = Field(min_length=1, max_length=2000)
    require_human_approval: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "p2-agent-workbench"}


@app.post("/tasks", status_code=202)
def create_task(request: TaskRequest) -> dict:
    try:
        state = service.submit(request.user_goal, request.require_human_approval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return state.public_dict()


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    state = service.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="task not found")
    return state.public_dict()


@app.post("/tasks/{task_id}/resume")
def resume_task(task_id: str) -> dict:
    try:
        state = service.workflow.resume(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.public_dict()


@app.get("/tasks/{task_id}/events")
def get_task_events(task_id: str) -> dict:
    if service.get(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "events": service.events_for(task_id)}


@app.get("/tools")
def list_tools() -> dict:
    """Return the tool whitelist and per-tool metadata."""
    registry = service.workflow.registry
    tools = []
    for name in registry.names():
        tool = registry.get(name)
        tools.append(
            {
                "name": name,
                "timeout_seconds": getattr(tool, "timeout_seconds", 0),
                "cost_per_call": getattr(tool, "cost_per_call", 0),
            }
        )
    return {
        "tools": tools,
        "budget_max": registry.budget.max_cost,
    }


@app.get("/tools/errors")
def list_tool_errors(
    tool_name: str | None = Query(default=None, description="filter by tool name"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    """Return archived tool-call errors for audit."""
    archive = service.workflow.registry.error_archive
    if archive is None:
        return {"errors": []}
    return {"errors": archive.list_errors(tool_name=tool_name, limit=limit)}


@app.get("/eval/dataset")
def get_eval_dataset() -> dict:
    """Return the deterministic evaluation set (size + a small sample)."""
    tasks = build_evaluation_set()
    return {
        "size": len(tasks),
        "sample": [t.to_dict() for t in tasks[:3]],
    }


@app.post("/eval/run")
def run_eval() -> dict:
    """Run single-agent vs multi-agent comparison over the evaluation set.

    With deterministic stubs this is fast; it runs synchronously and returns
    the full comparison report (metrics, deltas, per-category breakdown).
    """
    report = run_comparison()
    return report


@app.post("/eval/dataset/save")
def save_eval_dataset() -> dict:
    """Persist the evaluation set to eval/tasks_sample.json."""
    from pathlib import Path

    path = save_dataset(Path("eval/tasks_sample.json"))
    return {"path": str(path), "size": len(build_evaluation_set())}


_FRONTEND_HTML = (
    Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"
)


class HumanDecisionRequest(BaseModel):
    decision: Literal["approve", "revise"]


@app.post("/tasks/{task_id}/human-decision")
def decide_task(task_id: str, body: HumanDecisionRequest) -> dict:
    """Apply a human decision on a task paused at ``need_human``."""
    try:
        state = service.human_decision(task_id, body.decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.public_dict()


@app.get("/insight")
@app.get("/")
def insight_ui() -> HTMLResponse:
    """Serve the single-page workbench UI."""
    html = _FRONTEND_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/tasks/{task_id}/trace")
def get_trace(task_id: str) -> dict:
    """Return the execution trace (per-node spans) and a cost/time summary."""
    if service.get(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task_id,
        "spans": [s.model_dump() for s in tracing_store.list(task_id)],
        "summary": tracing_store.summary(task_id),
    }


@app.get("/tasks/{task_id}/guardrails")
def get_guardrails(task_id: str) -> dict:
    """Return guardrail findings recorded for a task."""
    if service.get(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task_id,
        "findings": [f.model_dump() for f in guardrail_store.list(task_id)],
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    """Server-Sent-Events stream of progress, spans and guardrail findings.

    Pushes incremental updates every 0.5s until the task reaches a terminal
    state (completed / failed / need_human), then emits a final ``done`` event.
    """
    if service.get(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")

    async def event_generator():
        sent_events = sent_spans = sent_guard = 0
        for _ in range(300):  # ~150s ceiling
            state = service.get(task_id)
            if state is None:
                yield _sse({"type": "not_found"})
                return
            events = service.events_for(task_id)
            spans = tracing_store.list(task_id)
            guards = guardrail_store.list(task_id)
            for e in events[sent_events:]:
                yield _sse({"type": "event", "data": e})
            for s in spans[sent_spans:]:
                yield _sse({"type": "span", "data": s.model_dump()})
            for g in guards[sent_guard:]:
                yield _sse({"type": "guardrail", "data": g.model_dump()})
            sent_events, sent_spans, sent_guard = len(events), len(spans), len(guards)
            if state.status in {"completed", "failed", "need_human"}:
                yield _sse({"type": "done", "status": state.status})
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
