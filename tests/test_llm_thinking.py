"""Request shaping for reasoning-capable endpoints (DeepSeek thinking mode).

The client must stay provider-neutral: it only sends ``thinking`` /
``reasoning_effort`` when explicitly configured, because stricter
OpenAI-compatible endpoints reject unknown fields.
"""
from __future__ import annotations

import httpx
import pytest

from p2_agent.llm import LLMClient

_OK = {
    "choices": [{"message": {"content": "ok"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


@pytest.fixture
def capture(monkeypatch) -> dict:
    """Patch httpx so we can inspect the exact request body."""
    box: dict = {}

    def fake_post(self, url, headers=None, json=None, **kwargs):
        box["body"] = json
        return httpx.Response(200, json=_OK)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return box


def _client(**kw) -> LLMClient:
    return LLMClient(base_url="http://x", api_key="k", model="deepseek-flash", **kw)


def test_thinking_disabled_is_sent_explicitly(capture) -> None:
    _client(thinking="disabled").chat(system="s", user="u")
    assert capture["body"]["thinking"] == {"type": "disabled"}


def test_thinking_enabled_is_sent_explicitly(capture) -> None:
    _client(thinking="enabled").chat(system="s", user="u")
    assert capture["body"]["thinking"] == {"type": "enabled"}


def test_no_thinking_field_when_unset(capture) -> None:
    """Default posture: send nothing, so other providers stay compatible."""
    _client(thinking="").chat(system="s", user="u")
    assert "thinking" not in capture["body"]


def test_garbage_thinking_value_is_not_forwarded(capture) -> None:
    """A typo must not become a 400 from the upstream API."""
    _client(thinking="diabled").chat(system="s", user="u")
    assert "thinking" not in capture["body"]


def test_temperature_sent_when_thinking_disabled(capture) -> None:
    _client(thinking="disabled", temperature=0.3).chat(system="s", user="u")
    assert capture["body"]["temperature"] == 0.3


def test_temperature_omitted_when_thinking_enabled(capture) -> None:
    """Reasoning endpoints drop sampling controls; do not pretend otherwise."""
    _client(thinking="enabled", temperature=0.3).chat(system="s", user="u")
    assert "temperature" not in capture["body"]


def test_reasoning_effort_only_with_thinking_enabled(capture) -> None:
    _client(thinking="enabled", reasoning_effort="low").chat(system="s", user="u")
    assert capture["body"]["reasoning_effort"] == "low"

    _client(thinking="disabled", reasoning_effort="low").chat(system="s", user="u")
    assert "reasoning_effort" not in capture["body"]


def test_invalid_reasoning_effort_dropped(capture) -> None:
    _client(thinking="enabled", reasoning_effort="turbo").chat(system="s", user="u")
    assert "reasoning_effort" not in capture["body"]


def test_max_tokens_always_sent(capture) -> None:
    """A reasoning model can burn the whole budget on CoT and return empty
    content — the cap must be present so callers control that tradeoff."""
    _client(thinking="enabled", max_tokens=4096).chat(system="s", user="u")
    assert capture["body"]["max_tokens"] == 4096
