from __future__ import annotations

from pathlib import Path
from uuid import UUID

from p2_agent.graph.workflow import build_workflow, route_from_checkpoint
from p2_agent.persistence import SQLiteStateStore
from p2_agent.schemas import WorkflowState
from p2_agent.tools.kb_search import KBSearchTool
from p2_agent.tools.registry import CostBudget, ErrorArchive, ToolRegistry


def create_tool_registry(store_path: Path | None = None) -> ToolRegistry:
    """Build the default tool whitelist with KB search registered.

    Every tool that the retriever can invoke must be registered here.
    Adding a new tool means adding it to this function — that is the
    security boundary.
    """
    budget = CostBudget(max_cost=5.0)
    registry = ToolRegistry(budget=budget)
    registry.register(KBSearchTool())
    if store_path:
        error_path = store_path.parent / "p2_errors.sqlite3"
    else:
        error_path = Path("data/p2_errors.sqlite3")
    registry.error_archive = ErrorArchive(str(error_path))
    return registry


class WorkflowService:
    def __init__(self, store: SQLiteStateStore | None = None) -> None:
        self.store = store or SQLiteStateStore(Path("data/p2_state.sqlite3"))
        self.registry = create_tool_registry(self.store.path)
        self.graph = build_workflow(self.registry)

    def create_and_run(self, user_goal: str) -> WorkflowState:
        state = WorkflowState(user_goal=user_goal)
        self.store.save(state)
        return self._run_and_save(state)

    def resume(self, task_id: UUID | str) -> WorkflowState:
        state = self.store.get(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status in {"completed", "need_human", "failed"}:
            return state
        return self._run_and_save(state)

    def get(self, task_id: UUID | str) -> WorkflowState | None:
        return self.store.get(task_id)

    def _run_and_save(self, state: WorkflowState) -> WorkflowState:
        try:
            if route_from_checkpoint(state) == "__end__":
                return state
            result = self.graph.invoke(state)
            final = (
                result
                if isinstance(result, WorkflowState)
                else WorkflowState.model_validate(result)
            )
        except Exception as exc:
            failed = state.model_copy(
                update={"status": "failed", "errors": [*state.errors, str(exc)]}
            )
            self.store.save(failed)
            raise
        self.store.save(final)
        return final
