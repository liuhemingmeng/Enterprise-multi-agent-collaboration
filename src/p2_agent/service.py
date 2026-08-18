from __future__ import annotations

from pathlib import Path
from uuid import UUID

from p2_agent.graph.workflow import build_workflow, route_from_checkpoint
from p2_agent.llm import LLMClient
from p2_agent.persistence import SQLiteStateStore
from p2_agent.schemas import WorkflowState
from p2_agent.settings import LLM_ENABLED, P1_ENABLED
from p2_agent.tools.kb_search import KBSearchTool, P1SearchTool
from p2_agent.tools.registry import CostBudget, ErrorArchive, ToolRegistry


def create_tool_registry(
    store_path: Path | None = None, *, use_p1: bool | None = None
) -> ToolRegistry:
    """Build the default tool whitelist with KB search registered.

    Every tool that the retriever can invoke must be registered here.
    Adding a new tool means adding it to this function — that is the
    security boundary.

    When ``use_p1`` (or, by default, the ``P1_ENABLED`` runtime switch) is
    true, the real :class:`P1SearchTool` is registered; otherwise the
    deterministic :class:`KBSearchTool` stub is used.  Either way the tool
    name is ``kb_search``, so the rest of the workflow is unchanged.
    """
    if use_p1 is None:
        use_p1 = P1_ENABLED
    budget = CostBudget(max_cost=5.0)
    registry = ToolRegistry(budget=budget)
    registry.register(P1SearchTool() if use_p1 else KBSearchTool())
    if store_path:
        error_path = store_path.parent / "p2_errors.sqlite3"
    else:
        error_path = Path("data/p2_errors.sqlite3")
    registry.error_archive = ErrorArchive(str(error_path))
    return registry


class WorkflowService:
    def __init__(
        self,
        store: SQLiteStateStore | None = None,
        *,
        llm: LLMClient | None = None,
        use_p1: bool | None = None,
    ) -> None:
        self.store = store or SQLiteStateStore(Path("data/p2_state.sqlite3"))
        self.registry = create_tool_registry(self.store.path, use_p1=use_p1)
        self.llm = llm if llm is not None else (LLMClient() if LLM_ENABLED else None)
        self.graph = build_workflow(self.registry, llm=self.llm)

    def create_and_run(self, user_goal: str) -> WorkflowState:
        from p2_agent.guardrails import (
            Severity,
            check_input,
            guardrail_store,
        )

        state = WorkflowState(user_goal=user_goal)
        findings = check_input(user_goal)
        critical = [f for f in findings if f.severity == Severity.critical]
        if critical:
            for f in findings:
                f.task_id = str(state.task_id)
                guardrail_store.add(f)
            failed = state.model_copy(
                update={"status": "failed", "errors": [f.message for f in critical]}
            )
            self.store.save(failed)
            return failed
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

    def human_decision(self, task_id: UUID | str, decision: str) -> WorkflowState:
        """Apply a human decision on a task paused at ``need_human``.

        ``approve`` resumes the export path; ``revise`` sends the draft back to
        the writer.  No-op when the task is not waiting for human input.
        """
        state = self.store.get(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status != "need_human":
            return state
        updated = state.model_copy(update={"human_decision": decision})
        return self._run_and_save(updated)

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
