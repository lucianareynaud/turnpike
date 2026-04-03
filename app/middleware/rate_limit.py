"""Rate limiting middleware for the gateway.

Implements per-caller sliding window rate limiting using in-memory deque.
For multi-instance deployments, replace deque with Redis ZRANGEBYSCORE + ZADD.
"""

import os
import time
from collections import deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

# Module-level singleton — shared across all middleware instances within the process.
# Tests reset this directly via reset_rate_limit_windows() rather than walking the
# middleware stack, consistent with the singleton pattern used for cache and
# circuit breaker.
_windows: dict[str, deque[float]] = {}


def reset_rate_limit_windows() -> None:
    """Clear all per-caller rate limit windows. Call from test fixtures only."""
    _windows.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces per-caller rate limiting.

    Uses a sliding window algorithm backed by the module-level _windows singleton.
    Health endpoints (/healthz, /readyz) are exempt from rate limiting.
    """

    _EXEMPT_PATHS = {"/healthz", "/readyz"}

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware."""
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        caller_id = getattr(request.state, "caller_id", "__anonymous__")

        # Read at dispatch time so monkeypatch.setenv works in tests.
        rpm = int(os.environ.get("RATE_LIMIT_RPM", "60"))

        now = time.monotonic()
        window = _windows.setdefault(caller_id, deque())

        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= rpm:
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        window.append(now)
        return await call_next(request)
