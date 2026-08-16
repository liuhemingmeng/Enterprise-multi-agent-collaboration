from fastapi import FastAPI, HTTPException
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
