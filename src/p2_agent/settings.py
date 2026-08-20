from __future__ import annotations

"""Centralised runtime configuration for P2.

P2 needs two external services at runtime (both optional in tests / CI):

1. **P1 Retrieval API** — the already-deployed RAG service. It only does
   retrieval + citation. Controlled by ``RAG_API_KEY``.
2. **A real LLM API** — drives the generative agents (planner / analyst /
   writer / reviewer). It is OpenAI-compatible, so any provider that exposes
   ``/chat/completions`` works. Controlled by ``LLM_API_KEY`` + ``LLM_BASE_URL``
   + ``LLM_MODEL``.

When a key is absent the corresponding client is disabled and the workflow
falls back to deterministic stubs, so the test-suite and the 100-query
evaluation never touch the network and stay reproducible + free.
"""

import os  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; real env vars also work.
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


# --- P1 Retrieval API -------------------------------------------------------
P1_RAG_BASE_URL: str = _get("P1_RAG_BASE_URL", "http://101.245.96.205") or "http://101.245.96.205"
RAG_API_KEY: str = _get("RAG_API_KEY", "") or ""
P1_RAG_SCORE_THRESHOLD: float = float(_get("P1_RAG_SCORE_THRESHOLD", "0.3") or "0.3")
P1_RAG_LIMIT: int = int(_get("P1_RAG_LIMIT", "8") or "8")
P1_RAG_TIMEOUT: float = float(_get("P1_RAG_TIMEOUT", "10") or "10")

# --- LLM API (OpenAI-compatible) -------------------------------------------
LLM_BASE_URL: str = _get("LLM_BASE_URL", "") or ""
LLM_API_KEY: str = _get("LLM_API_KEY", "") or ""
LLM_MODEL: str = _get("LLM_MODEL", "") or ""
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.3") or "0.3")
LLM_TIMEOUT: float = float(_get("LLM_TIMEOUT", "120") or "120")
LLM_MAX_TOKENS: int | None = int(_get("LLM_MAX_TOKENS", "1024") or 1024)

# --- Derived switches -------------------------------------------------------
P1_ENABLED: bool = bool(RAG_API_KEY)
LLM_ENABLED: bool = bool(LLM_API_KEY and LLM_BASE_URL and LLM_MODEL)
