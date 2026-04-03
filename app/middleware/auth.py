"""Authentication middleware and identity resolution for the gateway.

This module provides admission control via API key validation and establishes
seams for client-specific identity resolution. Both core functions are pure
and independently testable without middleware instantiation.
"""

import logging
import os
import secrets

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


def authenticate(provided_key: str, expected_key: str) -> bool:
    """Return True if provided_key matches expected_key.

    Uses secrets.compare_digest for constant-time comparison to prevent
    timing-based key enumeration attacks. Neither key value is ever logged.

    Args:
        provided_key: The API key provided by the caller.
        expected_key: The expected API key from configuration.

    Returns:
        True if keys match, False otherwise.
    """
    return secrets.compare_digest(provided_key, expected_key)


def resolve_caller(headers: dict[str, str]) -> tuple[str, str]:
    """Resolve (caller_id, tenant_id) from normalized request headers.

    Args:
        headers: dict with header names normalized to lowercase.

    Returns:
        (caller_id, tenant_id) where:
        - caller_id: the x-api-key value when present, otherwise "default".
          Used as the rate-limit window key. Not an identity claim.
        - tenant_id: always "default" in this spec. This is a transport seam
          for spec 010a — not a multi-tenancy claim and not suitable for
          billing, policy, or cost attribution until 010a wires it properly.

    Client-specific implementations replace this function to extract identity
    from JWT claims, OAuth tokens, or other auth artifacts. The middleware
    does not change when this function is replaced.

    This function is auth-method-agnostic: it resolves context from available
    headers after admission, it does not validate identity.
    """
    caller_id = headers.get("x-api-key", "") or "default"
    return (caller_id, "default")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key authentication.

    Validates X-API-Key header against APP_API_KEY environment variable.
    Health endpoints (/healthz, /readyz) are exempt from authentication.
    """

    _EXEMPT_PATHS = {"/healthz", "/readyz"}

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware and validate required configuration.

        Raises:
            ValueError: If APP_API_KEY environment variable is not set.
        """
        super().__init__(app)
        key = os.environ.get("APP_API_KEY")
        if not key:
            raise ValueError("APP_API_KEY environment variable is required")
        self._key = key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through authentication layer.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response from downstream handler or 401 Unauthorized.
        """
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")

        if not authenticate(provided, self._key):
            logger.debug("auth=fail path=%s", request.url.path)
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        logger.debug("auth=ok path=%s", request.url.path)

        # Normalize headers to lowercase before passing to resolve_caller.
        normalized = {k.lower(): v for k, v in request.headers.items()}
        caller_id, tenant_id = resolve_caller(normalized)

        # Store as runtime-local convenience for RateLimitMiddleware.
        # These are NOT core envelope fields — spec 010a receives resolved
        # values as plain Python arguments, not by reading request.state.
        request.state.caller_id = caller_id
        request.state.tenant_id = tenant_id

        return await call_next(request)
