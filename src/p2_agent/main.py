from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from p2_agent.service import WorkflowService

service = WorkflowService()

app = FastAPI(title="P2 Agent Workbench", version="0.1.0")


class TaskRequest(BaseModel):
    user_goal: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "p2-agent-workbench"}


@app.post("/tasks")
def create_task(request: TaskRequest) -> dict:
    try:
        state = service.create_and_run(request.user_goal)
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
        state = service.resume(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.public_dict()
