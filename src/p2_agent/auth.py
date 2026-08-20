"""X-API-Key authentication for P2, shared with P1 (one key for both services).

The authority for the API key is **P1**: P1 owns ``RAG_API_KEY`` and the
admin rotation endpoint (``POST /admin/rotate-key`` gated by ``ADMIN_KEY``).
P2 therefore proxies every protected request to P1's ``/corpus/search``
endpoint using the caller's key — P1's 200/401 decides validity. This makes
a rotation in P1 take effect for P2 **immediately**, with zero sync between
the two deployments: there is no copied key to drift.

Notes
-----
* When ``RAG_API_KEY`` / ``P1_RAG_BASE_URL`` are not configured (local dev,
  CI, stub mode) authentication is disabled, keeping tests fully offline.
* The last successfully verified key is cached in-process and used as the
  credential for P2's *own* outbound retrieval calls to P1, so a task
  submitted right after a rotation still retrieves with the fresh key.
* ``/tasks/{id}/stream`` is consumed by ``EventSource`` which cannot set
  headers, so the key may alternatively be passed as ``?api_key=...``.
"""

from __future__ import annotations

import threading

import httpx
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from p2_agent.settings import P1_RAG_BASE_URL, RAG_API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Reachable without a key so the entry portal, static assets, health probe,
# docs and the key-verification endpoint load for any visitor. All data
# routes stay protected.
_PUBLIC_EXACT = {
    "/",
    "/portal",
    "/workbench",
    "/insight",
    "/verify-key",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}
_PUBLIC_PREFIXES = ("/static",)

_lock = threading.Lock()
_current_key: str | None = None


def get_current_key() -> str | None:
    """Most recently verified API key (used for P2's own calls to P1)."""
    with _lock:
        return _current_key


def set_current_key(key: str) -> None:
    global _current_key
    with _lock:
        _current_key = key


def auth_enabled() -> bool:
    """Auth is on only when both the key and P1 endpoint are configured."""
    return bool((RAG_API_KEY or "").strip()) and bool((P1_RAG_BASE_URL or "").strip())


def _proxy_verify(key: str) -> bool:
    """Ask P1 whether ``key`` is valid: 200 -> valid, anything else -> invalid.

    Intentionally performs a tiny real query (``limit=1``) so a rotating key
    is validated against the *current* P1 state on every request — no cache,
    no window where a rotated-out key still works.
    """
    url = f"{P1_RAG_BASE_URL.rstrip('/')}/corpus/search"
    try:
        resp = httpx.get(
            url,
            headers={"X-API-Key": key},
            params={"query": "__auth_check__", "limit": 1},
            timeout=10,
        )
    except httpx.HTTPError:
        return False
    if resp.status_code == 200:
        set_current_key(key)
        return True
    return False


def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
    api_key_q: str | None = Security(api_key_query),
) -> None:
    """FastAPI dependency: require a valid X-API-Key on non-public routes."""
    path = request.url.path
    if path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIXES):
        return
    if not auth_enabled():
        return
    key = (api_key or api_key_q or "").strip()
    if not key or not _proxy_verify(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


__all__ = [
    "api_key_header",
    "get_current_key",
    "set_current_key",
    "auth_enabled",
    "verify_api_key",
]
