"""Auth layer tests: shared P1 key, proxy verification, public paths.

Note: the test environment runs with RAG_API_KEY/P1_RAG_BASE_URL cleared,
so authentication is disabled by default and existing API tests stay
offline. These tests explicitly re-enable it via monkeypatch.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

import p2_agent.auth as auth
from p2_agent.auth import auth_enabled, verify_api_key


def _fake_request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "headers": []})


def _make_app() -> FastAPI:
    app = FastAPI(dependencies=[Depends(verify_api_key)])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/protected")
    def protected() -> dict:
        return {"secret": 42}

    return app


def test_auth_disabled_when_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "")
    assert auth_enabled() is False


def test_auth_enabled_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    assert auth_enabled() is True


def test_public_paths_bypass_auth(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    monkeypatch.setattr(auth, "_proxy_verify", lambda key: False)  # would reject
    client = TestClient(_make_app())
    for path in ("/health", "/docs"):
        assert client.get(path).status_code == 200
    assert _path_public("/") and _path_public("/portal") and _path_public("/verify-key")


def _path_public(path: str) -> bool:
    return path in auth._PUBLIC_EXACT or path.startswith(auth._PUBLIC_PREFIXES)


def test_proxy_accepts_valid_key(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    monkeypatch.setattr(auth, "_proxy_verify", lambda key: True)
    client = TestClient(_make_app())
    resp = client.get("/protected", headers={"X-API-Key": "valid"})
    assert resp.status_code == 200
    assert resp.json() == {"secret": 42}


def test_proxy_rejects_invalid_key(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    monkeypatch.setattr(auth, "_proxy_verify", lambda key: False)
    client = TestClient(_make_app())
    resp = client.get("/protected", headers={"X-API-Key": "bad"})
    assert resp.status_code == 401


def test_missing_key_rejected_when_auth_on(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    monkeypatch.setattr(auth, "_proxy_verify", lambda key: True)
    client = TestClient(_make_app())
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_query_key_accepted_for_sse(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    monkeypatch.setattr(auth, "_proxy_verify", lambda key: True)
    client = TestClient(_make_app())
    resp = client.get("/protected?api_key=viaquery")
    assert resp.status_code == 200


def test_proxy_verify_sets_current_key(monkeypatch) -> None:
    """A successful proxy check caches the live key for P2's own calls."""
    import httpx

    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    auth.set_current_key(None)
    assert auth.get_current_key() is None

    def fake_get(url, headers=None, params=None, **kwargs):
        assert headers["X-API-Key"] == "fresh"
        return httpx.Response(200, json=[])

    monkeypatch.setattr(httpx, "get", fake_get)
    assert auth._proxy_verify("fresh") is True
    assert auth.get_current_key() == "fresh"

    def fake_get_401(url, headers=None, params=None, **kwargs):
        return httpx.Response(401, json={"detail": "Invalid or missing API key"})

    monkeypatch.setattr(httpx, "get", fake_get_401)
    assert auth._proxy_verify("stale") is False
    assert auth.get_current_key() == "fresh"  # unchanged


def test_current_key_used_by_retrieval_headers(monkeypatch) -> None:
    """P2's own outbound calls prefer the last verified key over .env copy."""
    import httpx

    from p2_agent.retrieval import P1RetrievalClient

    captured: dict = {}

    def fake_get(url, headers=None, params=None, **kwargs):
        captured["key"] = (headers or {}).get("X-API-Key")
        item = {"score": 0.9, "content": "x", "paper_title": "t", "paper_id": 1, "page_number": 3}
        return httpx.Response(200, json=[item])

    client = P1RetrievalClient(base_url="http://p1", api_key="stale-copy")
    fake_http = type("C", (), {"get": staticmethod(fake_get)})()
    monkeypatch.setattr(client, "_get_client", lambda: fake_http)
    auth.set_current_key("rotated-live-key")
    client.search("q")
    assert captured["key"] == "rotated-live-key"


def test_verify_key_endpoint_public(monkeypatch) -> None:
    import p2_agent.main as main

    monkeypatch.setattr(auth, "RAG_API_KEY", "k1")
    monkeypatch.setattr(auth, "P1_RAG_BASE_URL", "http://p1")
    # main.py bound _proxy_verify at import time; patch it there
    monkeypatch.setattr(main, "_proxy_verify", lambda key: key == "good")
    client = TestClient(main.app)
    assert client.post("/verify-key", json={"api_key": "good"}).json() == {"valid": True}
    assert client.post("/verify-key", json={"api_key": "bad"}).json() == {"valid": False}
