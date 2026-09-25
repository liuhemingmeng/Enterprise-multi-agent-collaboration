from __future__ import annotations

"""Per-client rate limiting for state-changing requests.

Scope note (deliberate, and worth being able to defend in an interview): this
is an **in-process sliding window**, which is correct for the single-container
deployment this project runs on.  It is *not* correct once the API is scaled to
several replicas — each replica would keep its own counters and the effective
limit would multiply.  The fix at that point is moving the counter to Redis
(or doing it in the gateway); the ``Limiter`` protocol below is the seam where
that swap happens, so no caller changes.
"""

import time  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Callable  # noqa: E402
from threading import Lock  # noqa: E402

from p2_agent.settings import (  # noqa: E402
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)

# Only requests that change state or kick off work are metered.  Polling and
# SSE are what a browser hits continuously; throttling them would break the
# live progress view rather than protect the backend.
METERED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class RateLimitExceeded(Exception):
    """Raised internally when a client is over budget."""

    def __init__(self, retry_after: float, limit: int) -> None:
        super().__init__(f"rate limit exceeded: {limit}/window")
        self.retry_after = retry_after
        self.limit = limit


class SlidingWindowLimiter:
    """Fixed-capacity sliding window counter, one deque of timestamps per key."""

    def __init__(
        self,
        limit: int = RATE_LIMIT_PER_MINUTE,
        window: float = RATE_LIMIT_WINDOW_SECONDS,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = max(0, limit)
        self.window = window
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, float, int]:
        """Consume one unit for ``key``.

        Returns ``(allowed, retry_after_seconds, remaining)``.  When ``allowed``
        is False, ``retry_after`` is how long until the oldest hit ages out.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                if len(self._hits) >= self.max_keys:
                    self._evict_locked(now)
                bucket = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if self.limit and len(bucket) >= self.limit:
                retry_after = max(0.0, bucket[0] + self.window - now)
                return False, retry_after, 0
            bucket.append(now)
            return True, 0.0, max(0, self.limit - len(bucket))

    def _evict_locked(self, now: float) -> None:
        """Drop idle buckets so a spoofed-IP flood cannot exhaust memory."""
        cutoff = now - self.window
        stale = [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
        if len(self._hits) >= self.max_keys:
            # Still full: shed the least-recently-active keys.
            ordered = sorted(self._hits.items(), key=lambda kv: kv[1][-1])
            for k, _ in ordered[: max(1, len(ordered) // 10)]:
                del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def client_key(request) -> str:
    """Identify a caller: forwarded-for header if behind a proxy, else peer IP."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    forwarded = request.headers.get("x-real-ip")
    if forwarded:
        return forwarded.strip()
    return request.client.host if request.client else "unknown"


def is_metered(request) -> bool:
    return RATE_LIMIT_ENABLED and request.method in METERED_METHODS


def build_middleware() -> Callable:
    """Return an ASGI-agnostic FastAPI middleware guarding write endpoints."""
    from fastapi.responses import JSONResponse

    async def rate_limit_middleware(request, call_next):  # noqa: ANN001
        if not is_metered(request):
            return await call_next(request)
        allowed, retry_after, remaining = limiter.check(client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {limiter.limit} write "
                        f"requests per {int(RATE_LIMIT_WINDOW_SECONDS)}s."
                    )
                },
                headers={
                    "Retry-After": str(max(1, int(retry_after) + 1)),
                    "X-RateLimit-Limit": str(limiter.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    return rate_limit_middleware
