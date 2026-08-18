from __future__ import annotations

import json
import re
import time

import httpx

from p2_agent.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns an unusable response."""


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
        max_retries: int = 3,
        backoff: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else LLM_API_KEY
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.timeout = timeout or LLM_TIMEOUT
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
            return data["choices"][0]["message"]["content"]

        raise LLMError(f"LLM call failed: {last_exc}")
