from __future__ import annotations

from p2_agent.agents.stubs import DeterministicKnowledgeBase
from p2_agent.retrieval import P1RetrievalClient
from p2_agent.schemas import Evidence


def validate_kb_params(params: dict) -> dict:
    """Shared parameter validation for both KB search tools."""
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


class KBSearchTool:
    """Wrap DeterministicKnowledgeBase as a registered, validated tool.

    This is the deterministic stub used when P1 is not wired up (tests, CI,
    and local development without a RAG key).  It satisfies the exact same
    tool contract as :class:`P1SearchTool`, so the registry / budget / tracing
    machinery is identical for both.
    """

    name = "kb_search"
    timeout_seconds: float = 5.0
    cost_per_call: float = 0.02

    def __init__(self, kb: DeterministicKnowledgeBase | None = None) -> None:
        self.kb = kb or DeterministicKnowledgeBase()

    def validate_params(self, params: dict) -> dict:
        return validate_kb_params(params)

    def execute(self, params: dict) -> list[Evidence]:
        return self.kb.search(params["query"], top_k=params["top_k"])


class P1SearchTool:
    """Real retrieval tool backed by the P1 RAG API.

    Drop-in replacement for :class:`KBSearchTool`: same ``name`` / validation /
    execute contract, so the tool whitelist, cost budget and tracing all apply
    unchanged.  The heavy lifting (auth header, 429 backoff, score threshold,
    field mapping) lives in :class:`P1RetrievalClient`.
    """

    name = "kb_search"
    timeout_seconds: float = 8.0
    cost_per_call: float = 0.02

    def __init__(self, client: P1RetrievalClient | None = None) -> None:
        self.client = client or P1RetrievalClient()

    def validate_params(self, params: dict) -> dict:
        return validate_kb_params(params)

    def execute(self, params: dict) -> list[Evidence]:
        return self.client.search(params["query"], top_k=params["top_k"])
