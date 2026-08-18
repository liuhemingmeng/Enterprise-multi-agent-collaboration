from __future__ import annotations

import pytest

from p2_agent.retrieval import P1RetrievalClient, P1RetrievalError
from p2_agent.tools.kb_search import P1SearchTool


class FakeResp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class FakeClient:
    """Minimal httpx.Client stand-in that replays queued responses in order."""

    def __init__(self, responses: list[FakeResp]):
        self._queue = list(responses)
        self.calls: list[tuple] = []

    def get(self, url, *, headers=None, params=None):
        self.calls.append(("GET", url, params))
        return self._queue.pop(0)

    def post(self, url, *, headers=None, json=None):
        self.calls.append(("POST", url, json))
        return self._queue.pop(0)


def _p1_item(**overrides) -> dict:
    base = {
        "content": "预测性维护通过传感器监测设备状态实现异常预警。",
        "page_number": 3,
        "section_title": "方法论",
        "score": 0.91,
        "paper_id": "doc-001",
        "paper_title": "制造业预测性维护白皮书",
    }
    base.update(overrides)
    return base


def test_maps_p1_payload_to_evidence():
    client = P1RetrievalClient(
        api_key="k",
        http_client=FakeClient([FakeResp(200, [_p1_item()])]),
    )
    ev = client.search("预测性维护")
    assert len(ev) == 1
    e = ev[0]
    assert e.evidence_id == "p1-doc-001-0"
    assert e.doc_id == "doc-001"
    assert e.doc_name == "制造业预测性维护白皮书"
    assert e.page == 3
    assert e.score == 0.91
    assert e.source_type == "local"


def test_score_threshold_filters_low_quality():
    items = [_p1_item(score=0.95), _p1_item(score=0.2, paper_id="doc-002")]
    client = P1RetrievalClient(
        api_key="k",
        score_threshold=0.5,
        http_client=FakeClient([FakeResp(200, items)]),
    )
    ev = client.search("x")
    assert [e.doc_id for e in ev] == ["doc-001"]


def test_all_below_threshold_returns_empty():
    items = [_p1_item(score=0.1), _p1_item(score=0.15, paper_id="doc-002")]
    client = P1RetrievalClient(
        api_key="k", score_threshold=0.3, http_client=FakeClient([FakeResp(200, items)])
    )
    assert client.search("x") == []


def test_page_zero_clamped_to_one():
    client = P1RetrievalClient(
        api_key="k", http_client=FakeClient([FakeResp(200, [_p1_item(page_number=0)])])
    )
    assert client.search("x")[0].page == 1


def test_empty_text_skipped():
    client = P1RetrievalClient(
        api_key="k",
        http_client=FakeClient([FakeResp(200, [_p1_item(content="  ")])]),
    )
    assert client.search("x") == []


def test_409_treated_as_no_evidence():
    client = P1RetrievalClient(
        api_key="k", http_client=FakeClient([FakeResp(409, text="corpus empty")])
    )
    assert client.search("x") == []


def test_422_treated_as_no_evidence():
    client = P1RetrievalClient(
        api_key="k", http_client=FakeClient([FakeResp(422, text="no similarity")])
    )
    assert client.search("x") == []


def test_429_is_retried_then_succeeds():
    client = P1RetrievalClient(
        api_key="k",
        backoff=0.01,
        http_client=FakeClient(
            [FakeResp(429), FakeResp(429), FakeResp(200, [_p1_item()])]
        ),
    )
    ev = client.search("x")
    assert len(ev) == 1
    assert len(client._client.calls) == 3


def test_429_exhausted_raises():
    client = P1RetrievalClient(
        api_key="k",
        backoff=0.01,
        http_client=FakeClient([FakeResp(429), FakeResp(429), FakeResp(429), FakeResp(429)]),
    )
    with pytest.raises(P1RetrievalError):
        client.search("x")


def test_5xx_raises_after_retries():
    client = P1RetrievalClient(
        api_key="k",
        backoff=0.01,
        http_client=FakeClient([FakeResp(503), FakeResp(503), FakeResp(503), FakeResp(503)]),
    )
    with pytest.raises(P1RetrievalError):
        client.search("x")


def test_p1_search_tool_wraps_client():
    fake_client = FakeClient([FakeResp(200, [_p1_item()])])
    tool = P1SearchTool(client=P1RetrievalClient(api_key="k", http_client=fake_client))
    assert tool.validate_params({"query": "a", "top_k": 2}) == {"query": "a", "top_k": 2}
    ev = tool.execute({"query": "a", "top_k": 2})
    assert len(ev) == 1
    assert ev[0].doc_id == "doc-001"
