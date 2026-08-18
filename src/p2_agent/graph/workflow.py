from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from p2_agent.agents.stubs import (
    DeterministicKnowledgeBase,
    analyst_node,
    planner_node,
    retriever_node,
    reviewer_node,
    writer_node,
)
from p2_agent.guardrails import run_node_guardrails
from p2_agent.llm import LLMClient
from p2_agent.schemas import WorkflowState
from p2_agent.tools.registry import ToolRegistry
from p2_agent.tracing import instrumented

MAX_RETRY = 3


def guarded(node_name: str, fn):
    """Wrap a graph node with tracing + detection-only guardrails.

    ``instrumented`` records a span (success and failure paths); afterward we
    run node-level guardrails (plan/draft/review shape, tool-result injection)
    which only *record* findings — they never mutate the workflow state.  The
    only blocking guardrail is the input check at the service boundary.
    """

    base = instrumented(node_name, fn)

    def wrapper(state):
        out = base(state)
        run_node_guardrails(node_name, state, out if isinstance(out, dict) else {})
        return out

    return wrapper


def route_from_checkpoint(state: WorkflowState) -> str:
    """Choose the next graph node from the last durable node snapshot."""
    if state.status in {"completed", "failed"}:
        return "__end__"
    if state.status == "need_human":
        if state.human_decision == "approve":
            return "human_approval"
        if state.human_decision == "revise":
            return "revise"
        return "__end__"
    last = state.trace[-1] if state.trace else ""
    if last == "reviewer":
        route = route_after_review(state)
        return {
            "approved": "human_approval",
            "revise": "revise",
            "insufficient": "insufficient",
            "human": "human_queue",
        }[route]
    return {
        "": "planner",
        "planner": "retriever",
        "retriever": "analyst",
        "analyst": "writer",
        "writer": "reviewer",
        "revise": "writer",
        "insufficient_fallback": "retriever",
        "human_approval": "export",
        "export": "__end__",
        "human_queue": "__end__",
    }.get(last, "planner")


def route_after_review(
    state: WorkflowState,
) -> Literal["approved", "revise", "insufficient", "human"]:
    if state.review is None:
        return "human"
    if state.retry_count >= MAX_RETRY:
        return "human"
    if state.review.decision == "approved" and state.require_human_approval:
        return "human"
    return {
        "approved": "approved",
        "revise": "revise",
        "insufficient_evidence": "insufficient",
        "human_review": "human",
    }[state.review.decision]


def revise_node(state: WorkflowState) -> dict:
    return {"retry_count": state.retry_count + 1, "trace": [*state.trace, "revise"]}


def insufficient_node(state: WorkflowState) -> dict:
    """Fallback after the reviewer finds evidence insufficient.

    If every retrieval query has already returned empty, re-running the same
    retrieval cannot help — short-circuit straight to human review instead of
    burning more budget on identical queries.  Otherwise bump the retry counter
    and let the graph re-enter the retriever.
    """
    if state.plan is not None:
        all_queries = [
            query
            for subtask in state.plan.subtasks
            for query in subtask.retrieval_queries
        ]
        known_empty = set(state.empty_queries)
        if all_queries and all(q in known_empty for q in all_queries):
            return {"status": "need_human", "trace": [*state.trace, "human_queue"]}
    return {
        "retry_count": state.retry_count + 1,
        "trace": [*state.trace, "insufficient_fallback"],
    }


def route_after_insufficient(state: WorkflowState) -> Literal["retriever", "human_queue"]:
    if state.status == "need_human":
        return "human_queue"
    return "retriever"


def human_queue_node(state: WorkflowState) -> dict:
    return {
        "status": "need_human",
        "human_decision": "",
        "trace": [*state.trace, "human_queue"],
    }


def approval_node(state: WorkflowState) -> dict:
    return {
        "status": "approved",
        "human_decision": "",
        "trace": [*state.trace, "human_approval"],
    }


def export_node(state: WorkflowState) -> dict:
    return {"status": "completed", "trace": [*state.trace, "export"]}


def build_workflow(
    registry: ToolRegistry | None = None,
    llm: LLMClient | None = None,
):
    """Build the compiled LangGraph workflow.

    * ``registry`` — when provided, the retriever routes searches through the
      tool whitelist.  When ``None`` (backward-compatible for stages 1–4),
      falls back to direct KB access.
    * ``llm`` — when provided, the generative agents (planner / analyst /
      writer / reviewer) call a real LLM; otherwise they use deterministic
      stubs.  Passing ``None`` keeps CI / tests fully offline and reproducible.
    """
    if registry is not None:
        builder = StateGraph(WorkflowState)
        builder.add_node("planner", guarded("planner", lambda s: planner_node(s, llm=llm)))
        builder.add_node(
            "retriever",
            guarded("retriever", lambda state: retriever_node(state, registry=registry)),
        )
    else:
        kb = DeterministicKnowledgeBase()
        builder = StateGraph(WorkflowState)
        builder.add_node("planner", guarded("planner", lambda s: planner_node(s, llm=llm)))
        builder.add_node(
            "retriever", guarded("retriever", lambda state: retriever_node(state, kb=kb))
        )
    builder.add_node("analyst", guarded("analyst", lambda s: analyst_node(s, llm=llm)))
    builder.add_node("writer", guarded("writer", lambda s: writer_node(s, llm=llm)))
    builder.add_node("reviewer", guarded("reviewer", lambda s: reviewer_node(s, llm=llm)))
    builder.add_node("revise", guarded("revise", revise_node))
    builder.add_node("insufficient", guarded("insufficient", insufficient_node))
    builder.add_node("human_queue", guarded("human_queue", human_queue_node))
    builder.add_node("human_approval", guarded("human_approval", approval_node))
    builder.add_node("export", guarded("export", export_node))

    builder.add_conditional_edges(
        START,
        route_from_checkpoint,
        {
            "planner": "planner",
            "retriever": "retriever",
            "analyst": "analyst",
            "writer": "writer",
            "reviewer": "reviewer",
            "revise": "revise",
            "insufficient": "insufficient",
            "human_approval": "human_approval",
            "export": "export",
            "human_queue": "human_queue",
            "__end__": END,
        },
    )
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "approved": "human_approval",
            "revise": "revise",
            "insufficient": "insufficient",
            "human": "human_queue",
        },
    )
    builder.add_edge("revise", "writer")
    builder.add_conditional_edges(
        "insufficient",
        route_after_insufficient,
        {"retriever": "retriever", "human_queue": "human_queue"},
    )
    builder.add_edge("human_approval", "export")
    builder.add_edge("export", END)
    builder.add_edge("human_queue", END)
    return builder.compile()


def run_task(user_goal: str, *, initial_state: WorkflowState | None = None) -> WorkflowState:
    initial = initial_state or WorkflowState(user_goal=user_goal)
    result = build_workflow().invoke(initial)
    if isinstance(result, WorkflowState):
        return result
    return WorkflowState.model_validate(result)
