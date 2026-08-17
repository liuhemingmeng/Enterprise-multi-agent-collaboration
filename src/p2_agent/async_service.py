from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from uuid import UUID

from p2_agent.events import ProgressEventStore
from p2_agent.persistence import SQLiteStateStore
from p2_agent.schemas import WorkflowState
from p2_agent.service import WorkflowService


class AsyncWorkflowService:
    """MVP background executor. Replace the executor with Redis/RQ later."""

    def __init__(self, store: SQLiteStateStore | None = None) -> None:
        self.store = store or SQLiteStateStore()
        self.workflow = WorkflowService(self.store)
        self.events = ProgressEventStore()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="p2-worker")
        self._futures: dict[str, Future[WorkflowState]] = {}
        self._lock = Lock()

    def submit(
        self, user_goal: str, require_human_approval: bool = False
    ) -> WorkflowState:
        state = WorkflowState(
            user_goal=user_goal,
            status="queued",
            require_human_approval=require_human_approval,
        )
        self.store.save(state)
        self.events.append(state.task_id, "queued", status="queued")
        future = self.executor.submit(self._run, state)
        with self._lock:
            self._futures[str(state.task_id)] = future
        return state

    def _run(self, state: WorkflowState) -> WorkflowState:
        self.events.append(state.task_id, "started", status="running")
        running = state.model_copy(update={"status": "running"})
        self.store.save(running)
        try:
            result = self.workflow._run_and_save(running)
        except Exception as exc:
            self.events.append(state.task_id, "failed", status="failed", error=str(exc))
            raise
        self.events.append(
            state.task_id,
            "completed" if result.status == "completed" else "stopped",
            status=result.status,
            trace=result.trace,
        )
        return result

    def get(self, task_id: UUID | str) -> WorkflowState | None:
        return self.workflow.get(task_id)

    def events_for(self, task_id: UUID | str) -> list[dict]:
        return self.events.list(task_id)

    def human_decision(self, task_id: UUID | str, decision: str) -> WorkflowState:
        return self.workflow.human_decision(task_id, decision)

    def wait(self, task_id: UUID | str) -> WorkflowState:
        with self._lock:
            future = self._futures.get(str(task_id))
        if future is None:
            state = self.get(task_id)
            if state is None:
                raise KeyError(f"task not found: {task_id}")
            return state
        return future.result(timeout=30)
