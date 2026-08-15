from p2_agent.graph.workflow import MAX_RETRY, build_workflow, run_task
from p2_agent.schemas import Plan, Subtask, WorkflowState


def test_end_to_end_workflow_completes_with_trace_and_citations():
    state = run_task("为制造业客户设计预测性维护方案")
    assert state.status == "completed"
    assert state.review is not None
    assert state.review.decision == "approved"
    assert state.evidence
    assert state.analysis is not None
    assert state.analysis.cited_evidence_ids
    assert state.trace == [
        "planner",
        "retriever",
        "analyst",
        "writer",
        "reviewer",
        "human_approval",
        "export",
    ]


def test_revise_branch_retries_then_hands_off_to_human():
    state = run_task("需要修订的预测性维护方案")
    assert state.status == "need_human"
    assert state.retry_count == 2
    assert state.trace.count("revise") == 2
    assert state.trace[-1] == "human_queue"


def test_insufficient_evidence_branch_is_bounded():
    initial = WorkflowState(
        user_goal="__no_evidence__",
        plan=Plan(
            goal="__no_evidence__",
            subtasks=[
                Subtask(id="s1", title="缺证据任务", retrieval_queries=["__no_evidence__"])
            ],
        ),
        retry_count=MAX_RETRY,
    )
    state = run_task("__no_evidence__", initial_state=initial)
    assert state.status == "need_human"
    assert state.retry_count == MAX_RETRY
    assert state.trace[-1] == "human_queue"


def test_route_from_completed_state_does_not_restart_work():
    initial = WorkflowState(user_goal="测试", status="completed")
    state = run_task("测试", initial_state=initial)
    assert state.status == "completed"


def test_workflow_is_compiled_and_accepts_typed_state():
    result = build_workflow().invoke(WorkflowState(user_goal="测试目标"))
    assert result["status"] == "completed"


def test_blank_goal_is_rejected_before_graph_execution():
    try:
        run_task("")
    except Exception as exc:
        assert "at least 1 character" in str(exc)
    else:
        raise AssertionError("blank goal should fail")
