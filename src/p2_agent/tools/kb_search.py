from __future__ import annotations

from p2_agent.agents.stubs import DeterministicKnowledgeBase
from p2_agent.schemas import Evidence


class KBSearchTool:
    """Wrap DeterministicKnowledgeBase as a registered, validated tool.

    This is the first tool in the whitelist.  When P1 HTTP client arrives,
    replace ``self.kb`` with an HTTP client that implements the same
    ``search`` contract.
    """

    name = "kb_search"
    timeout_seconds: float = 5.0
    cost_per_call: float = 0.02

    def __init__(self, kb: DeterministicKnowledgeBase | None = None) -> None:
        self.kb = kb or DeterministicKnowledgeBase()

    def validate_params(self, params: dict) -> dict:
        query = params.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("param 'query' must be a non-empty string")
        if len(query) > 500:
            raise ValueError("param 'query' must be <= 500 characters")

        top_k = params.get("top_k", 3)
        if not isinstance(top_k, int):
            raise ValueError("param 'top_k' must be an integer")
        if top_k < 1 or top_k > 10:
            raise ValueError("param 'top_k' must be in [1, 10]")
        return {"query": query.strip(), "top_k": top_k}

    def execute(self, params: dict) -> list[Evidence]:
        return self.kb.search(params["query"], top_k=params["top_k"])
