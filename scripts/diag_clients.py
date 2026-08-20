"""单独给两个外部客户端计时，定位真实链路耗时瓶颈。"""
from __future__ import annotations

import time

from p2_agent.llm import LLMClient
from p2_agent.retrieval import P1RetrievalClient


def main():
    # --- P1 retrieval ---
    p1 = P1RetrievalClient()
    t = time.perf_counter()
    try:
        ev = p1.search("大模型 检索增强生成 评测", top_k=2)
        print(f"[P1] OK 耗时 {time.perf_counter()-t:.2f}s, 证据 {len(ev)} 条")
        for e in ev[:2]:
            print(f"    - score={e.score:.3f} {e.doc_name} p{e.page}")
    except Exception as e:
        print(f"[P1] FAIL 耗时 {time.perf_counter()-t:.2f}s: {e!r}")

    # --- LLM planner-style JSON ---
    llm = LLMClient()
    system = "你是资深企业解决方案架构师。把目标拆为子任务。只输出 JSON，不要解释。"
    user = ('客户目标：为汽车零部件厂设计大模型智能质检方案\n\n'
            '输出 JSON：{"goal": str, "subtasks": [{"id": str, "title": str, '
            '"depends_on": [str], "retrieval_queries": [str]}]}')
    t = time.perf_counter()
    try:
        out = llm.chat(system=system, user=user)
        print(f"[LLM] OK 耗时 {time.perf_counter()-t:.2f}s")
        print("    回复前 200 字:", out[:200].replace("\n", " "))
    except Exception as e:
        print(f"[LLM] FAIL 耗时 {time.perf_counter()-t:.2f}s: {e!r}")

if __name__ == "__main__":
    main()
