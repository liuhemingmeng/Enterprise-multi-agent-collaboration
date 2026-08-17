from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from p2_agent.async_service import AsyncWorkflowService

service = AsyncWorkflowService()

app = FastAPI(title="P2 Agent Workbench", version="0.1.0")


class TaskRequest(BaseModel):
    user_goal: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "p2-agent-workbench"}


@app.post("/tasks", status_code=202)
def create_task(request: TaskRequest) -> dict:
    try:
        state = service.submit(request.user_goal)
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
