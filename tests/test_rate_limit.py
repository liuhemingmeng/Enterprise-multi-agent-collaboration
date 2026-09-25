"""Rate limiting: window accounting, method scoping, headers, isolation."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import p2_agent.rate_limit as rate_limit
from p2_agent.rate_limit import (
    SlidingWindowLimiter,
    build_middleware,
    client_key,
)
from p2_agent.rate_limit import (
    limiter as global_limiter,
)


def _app() -> FastAPI:
    """Minimal app with the same middleware the real service installs."""
    app = FastAPI()
    app.middleware("http")(build_middleware())

    @app.get("/read")
    def read() -> dict:
        return {"ok": True}

    @app.post("/write")
    def write() -> dict:
        return {"ok": True}

    return app


# --- window accounting -----------------------------------------------------


def test_allows_up_to_limit_then_blocks() -> None:
    lim = SlidingWindowLimiter(limit=3, window=60)
    for _ in range(3):
        allowed, _, _ = lim.check("1.2.3.4", now=100.0)
        assert allowed
    allowed, retry_after, remaining = lim.check("1.2.3.4", now=100.0)
    assert not allowed
    assert remaining == 0
    assert retry_after > 0


def test_window_slides_old_hits_expire() -> None:
    lim = SlidingWindowLimiter(limit=2, window=60)
    lim.check("k", now=0.0)
    lim.check("k", now=1.0)
    allowed, _, _ = lim.check("k", now=2.0)
    assert not allowed
    # Oldest hits age out once the window has passed.
    allowed, _, remaining = lim.check("k", now=62.0)
    assert allowed
    assert remaining == 1


def test_remaining_counts_down_within_window() -> None:
    lim = SlidingWindowLimiter(limit=3, window=60)
    _, _, r1 = lim.check("k", now=0.0)
    _, _, r2 = lim.check("k", now=0.1)
    assert (r1, r2) == (2, 1)


def test_zero_limit_disables_throttling() -> None:
    lim = SlidingWindowLimiter(limit=0, window=60)
    for _ in range(50):
        allowed, _, _ = lim.check("k", now=1.0)
        assert allowed


def test_keys_are_counted_independently() -> None:
    lim = SlidingWindowLimiter(limit=1, window=60)
    assert lim.check("a", now=0.0)[0]
    assert not lim.check("a", now=0.0)[0]
    assert lim.check("b", now=0.0)[0]  # different client, fresh budget


def test_eviction_prevents_unbounded_memory() -> None:
    lim = SlidingWindowLimiter(limit=5, window=60, max_keys=10)
    for i in range(10):
        lim.check(f"ip-{i}", now=0.0)
    # Fill past capacity; idle buckets must be dropped rather than kept forever.
    for i in range(10, 40):
        lim.check(f"ip-{i}", now=60.0 + i)
    assert len(lim._hits) <= 10


# --- middleware behaviour --------------------------------------------------


def test_write_requests_are_throttled(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=2, window=60))
    client = TestClient(_app())
    assert client.post("/write").status_code == 200
    assert client.post("/write").status_code == 200
    blocked = client.post("/write")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


def test_read_requests_are_never_throttled(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=1, window=60))
    client = TestClient(_app())
    for _ in range(10):
        assert client.get("/read").status_code == 200


def test_disabled_switch_lets_everything_through(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=1, window=60))
    client = TestClient(_app())
    for _ in range(10):
        assert client.post("/write").status_code == 200


def test_successful_response_carries_limit_headers(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=5, window=60))
    res = TestClient(_app()).post("/write")
    assert res.status_code == 200
    assert res.headers["X-RateLimit-Limit"] == "5"
    assert res.headers["X-RateLimit-Remaining"] == "4"


def test_forwarded_for_header_separates_clients(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=1, window=60))
    client = TestClient(_app())
    first = client.post("/write", headers={"X-Forwarded-For": "203.0.113.9"})
    second = client.post("/write", headers={"X-Forwarded-For": "203.0.113.9"})
    other = client.post("/write", headers={"X-Forwarded-For": "198.51.100.7"})
    assert first.status_code == 200
    assert second.status_code == 429
    assert other.status_code == 200


def test_429_body_explains_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "limiter", SlidingWindowLimiter(limit=1, window=60))
    client = TestClient(_app())
    client.post("/write")
    body = client.post("/write").json()
    assert "Rate limit exceeded" in body["detail"]


# --- helpers ---------------------------------------------------------------


def test_client_key_prefers_forwarded_header() -> None:
    class _Req:
        headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2"}
        client = None

    assert client_key(_Req()) == "1.1.1.1"


def test_client_key_falls_back_to_peer_ip() -> None:
    class _Client:
        host = "10.0.0.5"

    class _Req:
        headers: dict[str, str] = {}
        client = _Client()

    assert client_key(_Req()) == "10.0.0.5"


def test_global_limiter_uses_configured_defaults() -> None:
    assert global_limiter.limit >= 0
    assert global_limiter.window > 0


def test_reset_clears_state() -> None:
    lim = SlidingWindowLimiter(limit=1, window=60)
    lim.check("k", now=0.0)
    assert not lim.check("k", now=0.0)[0]
    lim.reset()
    assert lim.check("k", now=0.0)[0]


def test_middleware_does_not_mask_application_errors() -> None:
    """The limiter must not swallow or rewrite a genuine 500."""
    app = FastAPI()
    app.middleware("http")(build_middleware())

    @app.post("/boom")
    def boom():
        raise ValueError("node failure")

    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/boom").status_code == 500
