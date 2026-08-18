from __future__ import annotations

import time

import httpx

from p2_agent.schemas import Evidence
from p2_agent.settings import (
    P1_RAG_BASE_URL,
    P1_RAG_LIMIT,
    P1_RAG_SCORE_THRESHOLD,
    P1_RAG_TIMEOUT,
    RAG_API_KEY,
)


class P1RetrievalError(RuntimeError):
    """Raised when P1 cannot be reached or returns a non-recoverable status."""


class P1RetrievalClient:
    """HTTP client for the P1 RAG ``/corpus/search`` endpoint.

    Contract (confirmed with the P1 team, see 对接文档 §9.4):

    * ``GET {base_url}/corpus/search?query=<q>&limit=<5-10>``
    * auth header ``X-API-Key: <RAG_API_KEY>``
    * response items: ``content, page_number, section_title, score,
      paper_id, paper_title``
    * status ``409`` corpus empty / not indexed -> treat as no evidence
    * status ``422`` similarity <= 0 (rare) -> treat as no evidence
    * status ``429`` rate-limited (5 r/s + burst 10 by source IP) -> backoff retry
    * ``5xx`` / connection error -> backoff retry, then raise P1RetrievalError

    P1 **never returns an empty array** for a query when the corpus is indexed;
    even nonsense queries return a top-k (possibly low-score). Therefore the
    client filters by ``score_threshold`` and only reports "no evidence" when
    *every* returned item is below threshold.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        limit: int | None = None,
        score_threshold: float | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or P1_RAG_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else RAG_API_KEY
        self.limit = limit or P1_RAG_LIMIT
        self.score_threshold = (
            score_threshold if score_threshold is not None else P1_RAG_SCORE_THRESHOLD
        )
        self.timeout = timeout or P1_RAG_TIMEOUT
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = http_client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def search(self, query: str, top_k: int | None = None) -> list[Evidence]:
        k = min(top_k or self.limit, 10)
        url = f"{self.base_url}/corpus/search"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        params = {"query": query, "limit": k}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._get_client().get(url, headers=headers, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise P1RetrievalError(f"P1 request failed: {exc}") from exc

            if resp.status_code == 429:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise P1RetrievalError("P1 rate limited (429) after retries")
            if resp.status_code in (409, 422):
                # corpus empty / not indexed / no similarity -> no evidence
                return []
            if resp.status_code >= 500:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise P1RetrievalError(f"P1 server error {resp.status_code}")
            if resp.status_code != 200:
                raise P1RetrievalError(
                    f"P1 unexpected status {resp.status_code}: {resp.text[:200]}"
                )
            return self._to_evidence(resp.json())

        raise P1RetrievalError(f"P1 retrieval failed: {last_exc}")

    def _to_evidence(self, payload: object) -> list[Evidence]:
        if isinstance(payload, dict):
            items = payload.get("results") or payload.get("data") or payload.get("items")
            if items is None:
                # Some services return a bare list under a single key.
                items = [v for v in payload.values() if isinstance(v, list)]
                items = items[0] if items else []
        else:
            items = payload
        if not isinstance(items, list):
            items = []

        out: list[Evidence] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0) or 0.0)
            if score < self.score_threshold:
                continue
            text = (item.get("content") or "").strip()
            if not text:
                continue
            paper_id = str(item.get("paper_id") or item.get("doc_id") or f"doc{i}")
            out.append(
                Evidence(
                    evidence_id=f"p1-{paper_id}-{i}",
                    source_type="local",
                    doc_id=paper_id,
                    doc_name=str(item.get("paper_title") or "未知文档"),
                    chunk_id=f"{paper_id}-{i}",
                    page=max(int(item.get("page_number") or 1), 1),
                    text=text[:5000],
                    score=score,
                    url=None,
                )
            )
        return out
