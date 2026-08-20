from __future__ import annotations

import hashlib

from p2_agent.llm import LLMClient, extract_json
from p2_agent.schemas import Analysis, Evidence, Plan, Review, Subtask, WorkflowState
from p2_agent.tools.schemas import ToolCall


class DeterministicKnowledgeBase:
    """Local substitute for P1. It is intentionally deterministic for testing."""

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if "__no_evidence__" in query:
            return []
        # A stable short slug keeps evidence ids distinct per query so that the
        # multi-agent retriever (which issues several queries) collects different
        # evidence than a single broad search. Without this, every query returned
        # the same ev-1/ev-2 ids and de-duplication collapsed the advantage.
        slug = hashlib.md5(query.strip().encode("utf-8")).hexdigest()[:6]
        top_k = min(max(top_k, 1), 10)
        return [
            Evidence(
                evidence_id=f"ev-{slug}-{index}",
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


# ---------------------------------------------------------------------------
# Deterministic fallbacks (used when no LLM client is configured)
# ---------------------------------------------------------------------------

def _deterministic_plan(goal: str) -> Plan:
    return Plan(
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


def _deterministic_analysis(state: WorkflowState) -> Analysis:
    if not state.evidence:
        return Analysis(open_questions=["缺少可验证资料"], cited_evidence_ids=[])
    ids = [item.evidence_id for item in state.evidence]
    return Analysis(
        facts=["资料显示方案应覆盖状态监测、异常预警和维护决策"],
        assumptions=["当前成本收益需结合客户真实设备数据复核"],
        risks=["模拟资料不能替代客户现场验证"],
        open_questions=["客户现有设备数据接口是否可用"],
        cited_evidence_ids=ids,
    )


def _deterministic_draft(state: WorkflowState) -> str:
    if state.analysis is None:
        raise ValueError("analysis is required before writing")
    citations = " ".join(f"[{item.evidence_id}, p.{item.page}]" for item in state.evidence)
    fact = state.analysis.facts[0] if state.analysis.facts else "暂无充分事实"
    return (
        f"# 企业方案初稿\n\n## 目标\n{state.user_goal}\n\n"
        f"## 事实依据\n- {fact} {citations}\n\n"
        f"## 实施建议\n先建设设备状态监测、异常预警和维护决策闭环，再通过现场数据验证收益。\n\n"
        f"## 风险与待确认\n- {'; '.join(state.analysis.risks + state.analysis.open_questions)}\n"
    )


def _deterministic_review(state: WorkflowState) -> Review:
    if not state.evidence:
        return Review(decision="insufficient_evidence", reasons=["没有证据，不能生成可审核方案"])
    if "需要修订" in state.user_goal and state.retry_count < 2:
        return Review(decision="revise", reasons=["模拟规则：方案需要补充边界条件"])
    if "需要修订" in state.user_goal and state.retry_count >= 2:
        return Review(decision="human_review", reasons=["自动修订次数达到人工接管阈值"])
    if not state.draft.strip():
        return Review(decision="revise", reasons=["草稿为空"])
    return Review(
        decision="approved",
        reasons=["草稿包含证据引用、风险和待确认事项"],
        checked_evidence_ids=[item.evidence_id for item in state.evidence],
    )


# ---------------------------------------------------------------------------
# LLM-driven variants (used when an LLMClient is provided)
# ---------------------------------------------------------------------------

_DECISION_SYNONYMS = {
    "approved": "approved",
    "approve": "approved",
    "通过": "approved",
    "revise": "revise",
    "修改": "revise",
    "重写": "revise",
    "insufficient_evidence": "insufficient_evidence",
    "insufficient": "insufficient_evidence",
    "证据不足": "insufficient_evidence",
    "human_review": "human_review",
    "human": "human_review",
    "人工": "human_review",
}


def _normalize_decision(value: object) -> str:
    if not isinstance(value, str):
        return "human_review"
    return _DECISION_SYNONYMS.get(value.strip().lower(), "human_review")


def _fmt_evidence(evidence, *, limit: int = 8, max_chars: int = 600) -> str:
    """Compact evidence block for LLM prompts.

    Capping both the *number* of items and the *length* per item keeps the
    prompt small and the generation fast — the live deepseek-v4-flash endpoint
    is slow on long-context / long-output calls (a full 20-item proposal draft
    regularly blows the request timeout). The retriever still fetches everything;
    only what we send to the LLM is trimmed.
    """
    items = evidence[:limit]
    lines = []
    for e in items:
        text = (e.text or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"- [{e.evidence_id}, p.{e.page}] {e.doc_name}: {text}")
    return "\n".join(lines)


def planner_node(state: WorkflowState, llm: LLMClient | None = None) -> dict:
    goal = state.user_goal.strip()
    if not goal:
        raise ValueError("user_goal cannot be blank")
    plan = None
    if llm is not None:
        try:
            plan = _llm_plan(goal, llm)
        except Exception:
            plan = None  # fall back to deterministic
    if plan is None:
        plan = _deterministic_plan(goal)
    return {"plan": plan, "status": "running", "trace": [*state.trace, "planner"]}


def _llm_plan(goal: str, llm: LLMClient) -> Plan:
    system = (
        "你是资深企业解决方案架构师。负责把客户的宏观目标拆解为"
        "可检索、可执行的研究子任务。只输出 JSON，不要解释。"
    )
    user = (
        f"客户目标：{goal}\n\n"
        "请拆解为 2-4 个子任务并输出 JSON，格式：\n"
        '{"goal": str, "subtasks": [{"id": str, "title": str, '
        '"depends_on": [str], "retrieval_queries": [str]}]}\n'
        "每个子任务的 retrieval_queries 是该子任务需要的检索关键词（2-3 个）。"
    )
    data = extract_json(llm.chat(system=system, user=user))
    return Plan.model_validate(data)


def retriever_node(
    state: WorkflowState,
    kb: DeterministicKnowledgeBase | None = None,
    registry=None,
) -> dict:
    """Retrieve evidence through the tool registry (whitelist + budget + timeout).

    If ``registry`` is provided, every search goes through ``registry.call()``
    which enforces whitelist, parameter validation, timeout and cost budget.
    When P1 is wired up, the registry contains ``P1SearchTool`` so the same
    code path performs real retrieval.  If ``registry`` is ``None`` (backward
    compatible path), fall back to direct KB access.
    """
    if state.plan is None:
        raise ValueError("plan is required before retrieval")

    evidence: list[Evidence] = []
    empty_queries: list[str] = []
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
                    empty_queries.append(query)
    else:
        local_kb = kb or DeterministicKnowledgeBase()
        for subtask in state.plan.subtasks:
            for query in subtask.retrieval_queries:
                found = local_kb.search(query, top_k=2)
                if found:
                    evidence.extend(found)
                else:
                    empty_queries.append(query)

    unique = {item.evidence_id: item for item in evidence}
    merged_empty = [*state.empty_queries, *empty_queries]
    return {
        "evidence": list(unique.values()),
        "empty_queries": merged_empty,
        "trace": [*state.trace, "retriever"],
    }


def analyst_node(state: WorkflowState, llm: LLMClient | None = None) -> dict:
    analysis = None
    if llm is not None and state.evidence:
        try:
            analysis = _llm_analysis(state, llm)
        except Exception:
            analysis = None
    if analysis is None:
        analysis = _deterministic_analysis(state)
    return {"analysis": analysis, "trace": [*state.trace, "analyst"]}


def _llm_analysis(state: WorkflowState, llm: LLMClient) -> Analysis:
    evidence_block = _fmt_evidence(state.evidence)
    system = "你是严谨的方案分析师，基于证据做分析，不臆造。只输出 JSON。"
    user = (
        f"客户目标：{state.user_goal}\n\n检索到的证据：\n{evidence_block}\n\n"
        "输出 JSON：\n"
        '{"facts": [str], "assumptions": [str], "risks": [str], '
        '"open_questions": [str], "cited_evidence_ids": [str]}'
    )
    data = extract_json(llm.chat(system=system, user=user))
    return Analysis.model_validate(data)


def writer_node(state: WorkflowState, llm: LLMClient | None = None) -> dict:
    if state.analysis is None:
        raise ValueError("analysis is required before writing")
    draft = None
    if llm is not None:
        try:
            draft = _llm_draft(state, llm)
        except Exception:
            draft = None
    if draft is None:
        draft = _deterministic_draft(state)
    return {"draft": draft, "trace": [*state.trace, "writer"]}


def _llm_draft(state: WorkflowState, llm: LLMClient) -> str:
    analysis = state.analysis
    facts = "; ".join(analysis.facts) if analysis else ""
    evidence_block = _fmt_evidence(state.evidence)
    system = (
        "你是企业方案撰写专家。用 Markdown 撰写方案初稿，"
        "凡引用证据必须标注 [evidence_id, p.x]，不要编造未提供的数据。"
        "内容控制在 500 字以内，聚焦结论与可执行建议，避免冗长铺垫。"
    )
    user = (
        f"客户目标：{state.user_goal}\n\n关键事实：{facts}\n\n"
        f"可用证据：\n{evidence_block}\n\n请撰写方案初稿（含目标、事实依据、"
        "实施建议、风险与待确认），务必简练。"
    )
    return llm.chat(system=system, user=user).strip()


def reviewer_node(state: WorkflowState, llm: LLMClient | None = None) -> dict:
    review = None
    if llm is not None and state.draft.strip():
        try:
            review = _llm_review(state, llm)
        except Exception:
            review = None
    if review is None:
        review = _deterministic_review(state)
    return {"review": review, "trace": [*state.trace, "reviewer"]}


def _llm_review(state: WorkflowState, llm: LLMClient) -> Review:
    evidence_block = "\n".join(
        f"- [{e.evidence_id}, p.{e.page}] {e.doc_name}" for e in state.evidence
    )
    system = (
        "你是方案评审专家，严格把关质量与引用。只输出 JSON。"
        "decision 取值：approved / revise / insufficient_evidence / human_review。"
    )
    user = (
        f"客户目标：{state.user_goal}\n\n草稿：\n{state.draft}\n\n"
        f"可用证据：\n{evidence_block}\n\n"
        "输出 JSON：\n"
        '{"decision": str, "reasons": [str], "checked_evidence_ids": [str]}'
    )
    data = extract_json(llm.chat(system=system, user=user))
    data["decision"] = _normalize_decision(data.get("decision"))
    return Review.model_validate(data)
