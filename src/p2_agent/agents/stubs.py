from __future__ import annotations

from p2_agent.schemas import Analysis, Evidence, Plan, Review, Subtask, WorkflowState
from p2_agent.tools.schemas import ToolCall


class DeterministicKnowledgeBase:
    """Local substitute for P1. It is intentionally deterministic for testing."""

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if "__no_evidence__" in query:
            return []
        top_k = min(max(top_k, 1), 10)
        return [
            Evidence(
                evidence_id=f"ev-{index}",
                doc_id="stub-doc-001",
                doc_name="公开制造业预测性维护白皮书（模拟）",
                chunk_id=f"chunk-{index}",
                page=index,
                text=(
                    f"证据片段 {index}：预测性维护可围绕设备状态监测、"
                    f"异常预警和维护决策设计。查询词：{query}"
                ),
                score=round(1 - index * 0.1, 2),
            )
            for index in range(1, top_k + 1)
        ]


def planner_node(state: WorkflowState) -> dict:
    goal = state.user_goal.strip()
    if not goal:
        raise ValueError("user_goal cannot be blank")
    plan = Plan(
        goal=goal,
        subtasks=[
            Subtask(
                id="s1",
                title="明确业务目标与约束",
                retrieval_queries=[goal],
            ),
            Subtask(
                id="s2",
                title="整理技术方案与实施风险",
                depends_on=["s1"],
                retrieval_queries=[f"{goal} 技术方案 风险"],
            ),
        ],
    )
    return {"plan": plan, "status": "running", "trace": [*state.trace, "planner"]}


def retriever_node(
    state: WorkflowState,
    kb: DeterministicKnowledgeBase | None = None,
    registry=None,
) -> dict:
    """Retrieve evidence through the tool registry (whitelist + budget + timeout).

    If ``registry`` is provided, every search goes through ``registry.call()``
    which enforces whitelist, parameter validation, timeout and cost budget.
    If ``registry`` is ``None`` (backward-compatible path for stages 1–4),
    fall back to direct KB access.
    """
    if state.plan is None:
        raise ValueError("plan is required before retrieval")

    evidence: list[Evidence] = []
    if registry is not None:
        for subtask in state.plan.subtasks:
            for query in subtask.retrieval_queries:
                call = ToolCall(
                    tool_name="kb_search",
                    params={"query": query, "top_k": 2},
                    caller="retriever",
                )
                result = registry.call(call, task_id=str(state.task_id))
                if result.success and result.data:
                    evidence.extend(result.data)
    else:
        local_kb = kb or DeterministicKnowledgeBase()
        for subtask in state.plan.subtasks:
            for query in subtask.retrieval_queries:
                evidence.extend(local_kb.search(query, top_k=2))

    unique = {item.evidence_id: item for item in evidence}
    return {"evidence": list(unique.values()), "trace": [*state.trace, "retriever"]}


def analyst_node(state: WorkflowState) -> dict:
    if not state.evidence:
        return {
            "analysis": Analysis(open_questions=["缺少可验证资料"], cited_evidence_ids=[]),
            "trace": [*state.trace, "analyst"],
        }
    ids = [item.evidence_id for item in state.evidence]
    return {
        "analysis": Analysis(
            facts=["资料显示方案应覆盖状态监测、异常预警和维护决策"],
            assumptions=["当前成本收益需结合客户真实设备数据复核"],
            risks=["模拟资料不能替代客户现场验证"],
            open_questions=["客户现有设备数据接口是否可用"],
            cited_evidence_ids=ids,
        ),
        "trace": [*state.trace, "analyst"],
    }


def writer_node(state: WorkflowState) -> dict:
    if state.analysis is None:
        raise ValueError("analysis is required before writing")
    citations = " ".join(f"[{item.evidence_id}, p.{item.page}]" for item in state.evidence)
    fact = state.analysis.facts[0] if state.analysis.facts else "暂无充分事实"
    draft = (
        f"# 企业方案初稿\n\n## 目标\n{state.user_goal}\n\n"
        f"## 事实依据\n- {fact} {citations}\n\n"
        f"## 实施建议\n先建设设备状态监测、异常预警和维护决策闭环，再通过现场数据验证收益。\n\n"
        f"## 风险与待确认\n- {'; '.join(state.analysis.risks + state.analysis.open_questions)}\n"
    )
    return {"draft": draft, "trace": [*state.trace, "writer"]}


def reviewer_node(state: WorkflowState) -> dict:
    if not state.evidence:
        review = Review(
            decision="insufficient_evidence",
            reasons=["没有证据，不能生成可审核方案"],
        )
    elif "需要修订" in state.user_goal and state.retry_count < 2:
        review = Review(decision="revise", reasons=["模拟规则：方案需要补充边界条件"])
    elif "需要修订" in state.user_goal and state.retry_count >= 2:
        review = Review(decision="human_review", reasons=["自动修订次数达到人工接管阈值"])
    elif not state.draft.strip():
        review = Review(decision="revise", reasons=["草稿为空"])
    else:
        review = Review(
            decision="approved",
            reasons=["草稿包含证据引用、风险和待确认事项"],
            checked_evidence_ids=[item.evidence_id for item in state.evidence],
        )
    return {"review": review, "trace": [*state.trace, "reviewer"]}
