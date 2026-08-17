from __future__ import annotations

import time

from p2_agent.eval.metrics import EvalOutcome
from p2_agent.schemas import Evidence
from p2_agent.tools.registry import ToolRegistry
from p2_agent.tools.schemas import ToolCall


def single_agent_run(
    *, goal: str, category: str, domain: str, task_id: str, registry: ToolRegistry
) -> EvalOutcome:
    """Baseline: one LLM-style call that searches once and self-approves.

    This models the common "just ask one big model to write the whole proposal
    in one shot" approach.  It deliberately:
      * issues a single broad retrieval (top_k=3) instead of per-subtask search;
      * cites only the top-1 evidence in the facts section and leaves the
        recommendation / risk sections as generic, uncited prose;
      * self-approves without a reviewer gate, so it never routes hard cases
        to a human.
    The multi-agent pipeline is the system under test; this is the baseline we
    compare against for citation coverage, cost and robustness.
    """
    start = time.monotonic()
    evidence: list[Evidence] = []
    call = ToolCall(
        tool_name="kb_search",
        params={"query": goal, "top_k": 3},
        caller="single_agent",
    )
    result = registry.call(call, task_id=task_id)
    if result.success and result.data:
        evidence = [item if isinstance(item, Evidence) else Evidence.model_validate(item)
                    for item in result.data]

    cited = [evidence[0].evidence_id] if evidence else []
    first = evidence[0] if evidence else None
    citation_text = f"[{first.evidence_id}, p.{first.page}]" if first else ""
    fact_text = first.text if first else "暂无可验证资料"

    draft = (
        "# 企业方案（单Agent直接生成）\n\n"
        f"## 目标\n{goal}\n\n"
        f"## 事实依据\n- {fact_text} {citation_text}\n\n"
        "## 实施建议\n建议尽快落地，先小范围试点后再逐步推广，注意控制投入节奏。\n\n"
        "## 风险\n注意此类项目常见的落地风险，必要时由人工复核关键假设。\n"
    )

    duration_ms = round((time.monotonic() - start) * 1000, 2)
    cost = registry.budget.spent(task_id)
    coverage = (len(cited) / len(evidence)) if evidence else 0.0

    return EvalOutcome(
        task_id=task_id,
        goal=goal,
        category=category,
        domain=domain,
        pipeline="single_agent",
        status="completed",
        draft=draft,
        evidence_count=len(evidence),
        cited_evidence_ids=cited,
        citation_coverage=coverage,
        cost=cost,
        duration_ms=duration_ms,
        reached_terminal=True,
    )
