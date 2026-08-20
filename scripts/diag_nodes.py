"""逐节点计时：直接串行调用各 agent 节点，定位真实链路慢/挂点。"""
from __future__ import annotations

import time

from p2_agent.agents.stubs import (
    analyst_node,
    planner_node,
    retriever_node,
    reviewer_node,
    writer_node,
)
from p2_agent.llm import LLMClient
from p2_agent.schemas import WorkflowState
from p2_agent.service import create_tool_registry


def tcall(label, fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    print(f"[{time.perf_counter()-t:6.2f}s] {label}", flush=True)
    return out

def main():
    llm = LLMClient()
    registry = create_tool_registry(use_p1=True)
    goal = "为一家汽车零部件制造企业设计基于大模型的智能质检方案，目标是降低漏检率并缩短质检周期"
    state = WorkflowState(user_goal=goal)

    out = tcall("planner", planner_node, state, llm=llm)
    plan = out["plan"]
    print("   plan subtasks:", [s.title for s in plan.subtasks], flush=True)
    state = state.model_copy(update={"plan": plan})

    out = tcall("retriever", retriever_node, state, registry=registry)
    ev = out["evidence"]
    print(f"   evidence {len(ev)} 条, empty={out['empty_queries']}", flush=True)
    state = state.model_copy(update={"evidence": ev})

    out = tcall("analyst", analyst_node, state, llm=llm)
    state = state.model_copy(update={"analysis": out["analysis"]})

    out = tcall("writer", writer_node, state, llm=llm)
    state = state.model_copy(update={"draft": out["draft"]})

    out = tcall("reviewer", reviewer_node, state, llm=llm)
    print("   review decision:", out["review"].decision if out.get("review") else None, flush=True)

    print("ALL NODES DONE", flush=True)

if __name__ == "__main__":
    main()
