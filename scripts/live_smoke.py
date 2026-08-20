"""真实链路冒烟测试：用 .env 里的真实 P1 + LLM 跑完整工作流。

用于阶段十联调，验证：
- P1 检索客户端能拿到真实证据（非确定性桩）
- planner/analyst/writer/reviewer 走真实 LLM（非确定性桩回退）
- trace / guardrails / 状态机路由正常

不写入任何仓库文件，仅打印报告。
"""
from __future__ import annotations

import time

from p2_agent.service import WorkflowService
from p2_agent.settings import LLM_ENABLED, P1_ENABLED

_t0 = time.perf_counter()


def log(*a) -> None:
    msg = " ".join(str(x) for x in a)
    print(f"[{time.perf_counter() - _t0:6.1f}s] {msg}", flush=True)


def _is_deterministic_plan(plan) -> bool:
    # 确定性桩的 subtask id 固定为 s1/s2，标题固定
    ids = {s.id for s in plan.subtasks}
    return ids == {"s1", "s2"}


def main() -> None:
    log("=== 运行开关 ===")
    log(f"P1_ENABLED={P1_ENABLED}  LLM_ENABLED={LLM_ENABLED}")
    if not (P1_ENABLED and LLM_ENABLED):
        raise SystemExit("开关未全部开启，请确认 .env 已填真实密钥")

    svc = WorkflowService()  # 读 .env 的真实配置
    goal = "为一家科技公司设计基于检索增强生成(RAG)的企业知识库问答方案，重点说明检索策略、引用溯源与评测方法"

    log("开始 create_and_run ...")
    t0 = time.perf_counter()
    state = svc.create_and_run(goal)
    elapsed = time.perf_counter() - t0
    log(f"create_and_run 返回，status={state.status}，总耗时 {elapsed:.1f}s")

    print("\n=== 工作流结果 ===")
    print("task_id :", state.task_id)
    print("status   :", state.status)
    print("trace    :", state.trace)
    print("retry    :", state.retry_count)

    print("\n=== Planner（是否真实 LLM）===")
    if state.plan is None:
        print("plan 为 None！")
    else:
        print("deterministic_fallback:", _is_deterministic_plan(state.plan))
        print("goal     :", state.plan.goal)
        for s in state.plan.subtasks:
            print(f"  - {s.id} {s.title} | deps={s.depends_on} | queries={s.retrieval_queries}")

    print("\n=== Retriever（是否真实 P1 证据）===")
    print("evidence 条数:", len(state.evidence))
    print("empty_queries:", state.empty_queries)
    for e in state.evidence[:5]:
        print(f"  - [{e.evidence_id}] score={e.score:.3f} {e.doc_name} p{e.page}: {e.text[:60]}...")

    print("\n=== Analyst ===")
    if state.analysis is None:
        print("analysis 为 None")
    else:
        print("facts        :", state.analysis.facts[:3])
        print("risks        :", state.analysis.risks[:3])
        print("cited_ids    :", state.analysis.cited_evidence_ids[:5])

    print("\n=== Writer（草稿前 400 字）===")
    print((state.draft or "")[:400])

    print("\n=== Reviewer ===")
    if state.review is None:
        print("review 为 None")
    else:
        print("decision:", state.review.decision)
        print("reasons :", state.review.reasons[:3])
        print("checked :", (state.review.checked_evidence_ids or [])[:5])

    print("\n=== 耗时 ===")
    print(f"总耗时 {elapsed:.1f}s")

    print("\n=== Trace spans ===")
    from p2_agent.tracing import tracing_store
    summ = tracing_store.summary(str(state.task_id))
    print(
        f"  span_count={summ['span_count']} total={summ['total_duration_ms']}ms "
        f"cost=${summ['total_cost_usd']}"
    )
    for s in tracing_store.list(str(state.task_id)):
        print(f"  - {s.node} {s.status} dur={s.duration_ms}ms")

    print("\n=== Guardrails ===")
    from p2_agent.guardrails import guardrail_store
    findings = guardrail_store.list(str(state.task_id))
    print("findings 条数:", len(findings))
    for f in findings:
        print(f"  - [{f.severity}] {f.category}: {f.message[:80]}")


if __name__ == "__main__":
    main()
