from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass

import httpx

from p2_agent.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns an unusable response."""


# --------------------------------------------------------------------------
# Token accounting
# --------------------------------------------------------------------------
# Unit prices are USD per 1M tokens and are *configuration, not measurement*.
# The token counts in a Span are what the provider actually reported; the cost
# is derived from these rates.  Providers move prices and many (including
# subscription/coding-plan endpoints) do not bill per token at all, so the
# rates stay overridable via environment variables and the README states the
# conversion explicitly rather than implying a metered bill.

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model -> (input USD / 1M tokens, output USD / 1M tokens)
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
    "gpt-4o-mini": (0.15, 0.60),
    "qwen-plus": (0.40, 1.20),
    "glm-4-flash": (0.00, 0.00),
}
DEFAULT_PRICING = (0.30, 1.20)


def _env_price(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def unit_prices(model: str) -> tuple[float, float]:
    """Return (input, output) USD-per-1M-token rates for ``model``.

    Environment overrides ``LLM_PRICE_IN_PER_M`` / ``LLM_PRICE_OUT_PER_M`` win
    over the table, so a price change never requires a code deploy.
    """
    base_in, base_out = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (
        _env_price("LLM_PRICE_IN_PER_M", base_in),
        _env_price("LLM_PRICE_OUT_PER_M", base_out),
    )


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Convert a token count into USD using the configured unit prices."""
    in_rate, out_rate = unit_prices(model)
    return round(
        (prompt_tokens / 1_000_000) * in_rate
        + (completion_tokens / 1_000_000) * out_rate,
        8,
    )


@dataclass
class Usage:
    """Token/cost totals for one or more LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class _UsageScope(threading.local):
    """Per-thread accumulator so concurrent tasks never mix their usage.

    Graph nodes run synchronously inside one thread, and ``instrumented``
    resets/reads the scope around each node — so a Span always carries exactly
    the tokens that node consumed, even when several tasks run in parallel.
    """

    def __init__(self) -> None:
        self.usage = Usage()


_scope = _UsageScope()


def reset_usage_scope() -> None:
    _scope.usage = Usage()


def take_usage_scope() -> Usage:
    current = _scope.usage
    _scope.usage = Usage()
    return current


def peek_usage_scope() -> Usage:
    return _scope.usage


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    u = _scope.usage
    u.prompt_tokens += prompt_tokens
    u.completion_tokens += completion_tokens
    u.cost_usd = round(u.cost_usd + compute_cost(model, prompt_tokens, completion_tokens), 8)
    u.calls += 1


def parse_usage(payload: dict, model: str) -> Usage:
    """Extract usage from an OpenAI-compatible chat response.

    Providers differ on which fields they populate (some omit ``total_tokens``,
    some nest under ``usage.completion_tokens_details``), so every field is
    optional and falls back to whatever is available.
    """
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if not completion:
        details = usage.get("completion_tokens_details") or {}
        completion = int(details.get("reasoning_tokens") or 0) + int(
            details.get("accepted_prediction_tokens") or 0
        )
    if not prompt and not completion:
        total = int(usage.get("total_tokens") or 0)
        if total:
            # Unknown split: attribute the whole total to completion.
            completion = total
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=compute_cost(model, prompt, completion),
        calls=1,
    )


def extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from an LLM reply.

    Handles fenced code blocks (```json ... ```) and prose wrapping.  Raises
    ``ValueError`` if no JSON object can be found.
    """
    if not text:
        raise ValueError("empty LLM output")
    cleaned = text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("no JSON object found in LLM output")


class LLMClient:
    """OpenAI-compatible chat client.

    Works with any provider exposing ``POST {base_url}/chat/completions``
    (DeepSeek, 通义千问, 智谱 GLM, Moonshot, OpenAI, ...).  Retries 429/5xx
    with exponential backoff; raises :class:`LLMError` after exhausting retries.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else LLM_API_KEY
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.timeout = timeout or LLM_TIMEOUT
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = http_client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def chat(self, *, system: str, user: str, temperature: float | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._get_client().post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise LLMError(f"LLM request failed: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise LLMError(f"LLM error {resp.status_code} after retries")
            if resp.status_code != 200:
                raise LLMError(
                    f"LLM unexpected status {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            parsed = parse_usage(data, self.model)
            _record_usage(self.model, parsed.prompt_tokens, parsed.completion_tokens)
            return data["choices"][0]["message"]["content"]

        raise LLMError(f"LLM call failed: {last_exc}")
