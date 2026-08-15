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
from p2_agent.schemas import WorkflowState

MAX_RETRY = 3


def route_from_checkpoint(state: WorkflowState) -> str:
    """Choose the next graph node from the last durable node snapshot."""
    if state.status in {"completed", "failed", "need_human"}:
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
    return {
        "approved": "approved",
        "revise": "revise",
        "insufficient_evidence": "insufficient",
        "human_review": "human",
    }[state.review.decision]


def revise_node(state: WorkflowState) -> dict:
    return {"retry_count": state.retry_count + 1, "trace": [*state.trace, "revise"]}


def insufficient_node(state: WorkflowState) -> dict:
    return {
        "retry_count": state.retry_count + 1,
        "trace": [*state.trace, "insufficient_fallback"],
    }


def human_queue_node(state: WorkflowState) -> dict:
    return {"status": "need_human", "trace": [*state.trace, "human_queue"]}


def approval_node(state: WorkflowState) -> dict:
    return {"status": "approved", "trace": [*state.trace, "human_approval"]}


def export_node(state: WorkflowState) -> dict:
    return {"status": "completed", "trace": [*state.trace, "export"]}


def build_workflow():
    kb = DeterministicKnowledgeBase()
    builder = StateGraph(WorkflowState)
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", lambda state: retriever_node(state, kb))
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("revise", revise_node)
    builder.add_node("insufficient", insufficient_node)
    builder.add_node("human_queue", human_queue_node)
    builder.add_node("human_approval", approval_node)
    builder.add_node("export", export_node)

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
    builder.add_edge("insufficient", "retriever")
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
