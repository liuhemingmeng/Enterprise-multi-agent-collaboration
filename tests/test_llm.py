from __future__ import annotations

import pytest

from p2_agent.agents.stubs import planner_node, reviewer_node, writer_node
from p2_agent.llm import LLMClient, LLMError, extract_json
from p2_agent.schemas import Plan, Review, WorkflowState


class FakeResp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResp]):
        self._queue = list(responses)
        self.calls: list[tuple] = []

    def post(self, url, *, headers=None, json=None):
        self.calls.append((url, json))
        return self._queue.pop(0)


def _chat_resp(content: str) -> FakeResp:
    return FakeResp(200, {"choices": [{"message": {"content": content}}]})


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_extract_json_prose_wrapped():
    text = "好的，这是结果：\n{\"a\": 1}\n以上。"
    assert extract_json(text) == {"a": 1}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_chat_returns_content():
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        http_client=FakeClient([_chat_resp("hello")]),
    )
    assert llm.chat(system="s", user="u") == "hello"


def test_chat_429_retried():
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        backoff=0.01,
        http_client=FakeClient([FakeResp(429), FakeResp(429), _chat_resp("ok")]),
    )
    assert llm.chat(system="s", user="u") == "ok"
    assert len(llm._client.calls) == 3


def test_chat_5xx_raises():
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        backoff=0.01,
        http_client=FakeClient([FakeResp(503), FakeResp(503), FakeResp(503), FakeResp(503)]),
    )
    with pytest.raises(LLMError):
        llm.chat(system="s", user="u")


def test_planner_node_uses_llm():
    plan_json = (
        '{"goal":"做方案","subtasks":['
        '{"id":"s1","title":"目标","retrieval_queries":["a","b"]},'
        '{"id":"s2","title":"技术","depends_on":["s1"],"retrieval_queries":["c"]}]}'
    )
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        http_client=FakeClient([_chat_resp(plan_json)]),
    )
    state = WorkflowState(user_goal="为某工厂做预测性维护方案")
    out = planner_node(state, llm=llm)
    plan = out["plan"]
    assert isinstance(plan, Plan)
    assert len(plan.subtasks) == 2
    assert plan.subtasks[1].depends_on == ["s1"]


def test_planner_node_falls_back_on_llm_failure():
    # LLM returns garbage JSON -> node must fall back to deterministic plan.
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        http_client=FakeClient([_chat_resp("not valid json at all")]),
    )
    state = WorkflowState(user_goal="为某工厂做预测性维护方案")
    out = planner_node(state, llm=llm)
    assert isinstance(out["plan"], Plan)
    assert out["plan"].subtasks[0].id == "s1"


def test_writer_node_uses_llm():
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        http_client=FakeClient([_chat_resp("# 由LLM生成的企业方案初稿")]),
    )
    state = WorkflowState(
        user_goal="g",
        analysis=None,  # type: ignore[arg-type]
    )
    # writer requires analysis; build a minimal valid state
    from p2_agent.schemas import Analysis

    state.analysis = Analysis(facts=["f"], cited_evidence_ids=[])
    state.evidence = []
    out = writer_node(state, llm=llm)
    assert "LLM生成" in out["draft"]


def test_reviewer_node_uses_llm_and_normalizes_decision():
    review_json = '{"decision":"通过","reasons":["引用完整"],"checked_evidence_ids":["e1"]}'
    llm = LLMClient(
        base_url="https://api.example.com/v1",
        api_key="sk",
        model="m",
        http_client=FakeClient([_chat_resp(review_json)]),
    )
    state = WorkflowState(user_goal="g", draft="draft text", evidence=[])
    out = reviewer_node(state, llm=llm)
    review = out["review"]
    assert isinstance(review, Review)
    assert review.decision == "approved"
