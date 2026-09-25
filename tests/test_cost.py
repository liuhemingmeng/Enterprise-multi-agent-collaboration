"""Token accounting: usage parsing, unit-price conversion, span attribution."""
from __future__ import annotations

import threading

import httpx
import pytest

from p2_agent.llm import (
    LLMClient,
    compute_cost,
    parse_usage,
    peek_usage_scope,
    reset_usage_scope,
    take_usage_scope,
    unit_prices,
)
from p2_agent.tracing import Span, TracingStore, instrumented, tracing_store


def _payload(usage: dict | None) -> dict:
    return {
        "choices": [{"message": {"content": "ok"}}],
        **({"usage": usage} if usage is not None else {}),
    }


class _FakeState:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


# --- usage parsing -------------------------------------------------------


def test_parse_usage_reads_standard_fields() -> None:
    u = parse_usage(
        _payload({"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}),
        "deepseek-v4-flash",
    )
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (120, 30, 150)
    assert u.calls == 1


def test_parse_usage_missing_block_is_zero() -> None:
    u = parse_usage(_payload(None), "deepseek-v4-flash")
    assert u.total_tokens == 0
    assert u.cost_usd == 0.0


def test_parse_usage_falls_back_to_total_when_split_absent() -> None:
    """Some providers only report a total; keep the tokens rather than drop them."""
    u = parse_usage(_payload({"total_tokens": 90}), "deepseek-v4-flash")
    assert u.total_tokens == 90


def test_parse_usage_reads_reasoning_token_details() -> None:
    """Reasoning models report output under completion_tokens_details."""
    u = parse_usage(
        _payload(
            {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "completion_tokens_details": {"reasoning_tokens": 40},
            }
        ),
        "deepseek-r1",
    )
    assert u.completion_tokens == 40
    assert u.total_tokens == 50


# --- price conversion ----------------------------------------------------


def test_compute_cost_uses_per_million_rates() -> None:
    in_rate, out_rate = unit_prices("deepseek-v4-flash")
    expected = (1_000 / 1_000_000) * in_rate + (500 / 1_000_000) * out_rate
    assert compute_cost("deepseek-v4-flash", 1_000, 500) == round(expected, 8)


def test_compute_cost_output_costs_more_than_input() -> None:
    in_rate, out_rate = unit_prices("deepseek-v4-flash")
    assert out_rate > in_rate
    assert compute_cost("deepseek-v4-flash", 0, 100) > compute_cost(
        "deepseek-v4-flash", 100, 0
    )


def test_unknown_model_falls_back_to_default_rate() -> None:
    in_rate, out_rate = unit_prices("some-unlisted-model")
    assert in_rate > 0 and out_rate > 0


def test_env_overrides_unit_price(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PRICE_IN_PER_M", "2.0")
    monkeypatch.setenv("LLM_PRICE_OUT_PER_M", "8.0")
    # 1M input tokens at $2/1M == $2.00
    assert compute_cost("deepseek-v4-flash", 1_000_000, 0) == pytest.approx(2.0)


def test_invalid_env_price_keeps_table_rate(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PRICE_IN_PER_M", "not-a-number")
    in_rate, _ = unit_prices("deepseek-v4-flash")
    assert in_rate > 0


# --- client records usage ------------------------------------------------


def test_client_records_usage_into_scope(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(self, url, headers=None, json=None, **kwargs):
        captured["body"] = json
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 50},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    reset_usage_scope()
    client = LLMClient(base_url="http://x", api_key="k", model="deepseek-v4-flash")
    client.chat(system="s", user="u")

    u = take_usage_scope()
    assert (u.prompt_tokens, u.completion_tokens) == (200, 50)
    assert u.calls == 1
    assert u.cost_usd > 0


def test_client_without_usage_block_records_nothing(monkeypatch) -> None:
    def fake_post(self, url, headers=None, json=None, **kwargs):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    reset_usage_scope()
    LLMClient(base_url="http://x", api_key="k").chat(system="s", user="u")
    assert take_usage_scope().total_tokens == 0


# --- span attribution ----------------------------------------------------


def test_span_carries_tokens_consumed_by_node(monkeypatch) -> None:
    def fake_post(self, url, headers=None, json=None, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 100},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    store = TracingStore()
    monkeypatch.setattr("p2_agent.tracing.tracing_store", store)

    def node(state):
        LLMClient(base_url="http://x", api_key="k").chat(system="s", user="u")
        return {}

    instrumented("writer", node)(_FakeState("t-1"))
    spans = store.list("t-1")
    assert len(spans) == 1
    assert spans[0].tokens == 400
    assert spans[0].cost_usd > 0


def test_failed_node_keeps_tokens_already_spent(monkeypatch) -> None:
    """A node that spends tokens then throws must still show the spend."""

    def fake_post(self, url, headers=None, json=None, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{bad json"}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 20},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    store = TracingStore()
    monkeypatch.setattr("p2_agent.tracing.tracing_store", store)

    def node(state):
        LLMClient(base_url="http://x", api_key="k").chat(system="s", user="u")
        raise RuntimeError("parse failed")

    with pytest.raises(RuntimeError):
        instrumented("analyst", node)(_FakeState("t-2"))

    span = store.list("t-2")[0]
    assert span.status == "error"
    assert span.tokens == 520
    assert span.cost_usd > 0


def test_usage_scope_is_thread_local() -> None:
    """Two concurrent tasks must not see each other's tokens."""
    from p2_agent.llm import _record_usage

    results: dict[str, int] = {}
    barrier = threading.Barrier(2)

    def run(tag: str, tokens: int) -> None:
        reset_usage_scope()
        barrier.wait()  # both threads are now inside their own scope
        for _ in range(5):
            _record_usage("deepseek-v4-flash", tokens, 0)
        results[tag] = peek_usage_scope().total_tokens

    t1 = threading.Thread(target=run, args=("a", 10))
    t2 = threading.Thread(target=run, args=("b", 999))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results == {"a": 50, "b": 4995}


# --- summary aggregation -------------------------------------------------


def test_summary_sums_llm_and_tool_cost() -> None:
    store = TracingStore()
    store.add(
        Span(
            task_id="t-3",
            node="writer",
            started_at="a",
            ended_at="b",
            duration_ms=10,
            tokens=1000,
            cost_usd=0.002,
        )
    )
    store.add(
        Span(
            task_id="t-3",
            node="analyst",
            started_at="a",
            ended_at="b",
            duration_ms=5,
            tokens=500,
            cost_usd=0.001,
        )
    )
    store.record_tool_cost("t-3", 0.01)

    s = store.summary("t-3")
    assert s["total_tokens"] == 1500
    assert s["llm_cost_usd"] == pytest.approx(0.003)
    assert s["tool_cost_usd"] == pytest.approx(0.01)
    assert s["total_cost_usd"] == pytest.approx(0.013)


def test_summary_per_node_breaks_down_tokens() -> None:
    store = TracingStore()
    for _ in range(3):
        store.add(
            Span(
                task_id="t-4",
                node="writer",
                started_at="a",
                ended_at="b",
                duration_ms=1,
                tokens=100,
                cost_usd=0.0002,
            )
        )
    s = store.summary("t-4")
    writer = [n for n in s["per_node"] if n["node"] == "writer"][0]
    assert writer["count"] == 3
    assert writer["tokens"] == 300


def test_tracing_store_default_instance_exists() -> None:
    assert isinstance(tracing_store, TracingStore)
