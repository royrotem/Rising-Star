"""
Simple in-memory rate limiting middleware for UAIE.

Uses a very high default limit (10,000 requests per minute per IP)
to protect against accidental abuse without affecting normal usage.
No external dependencies required — uses a simple sliding window counter.

Implemented as pure ASGI middleware (not BaseHTTPMiddleware) to avoid
the known Starlette issue with stacked BaseHTTPMiddleware corrupting
response bodies.
"""

import logging
import time
import json
from collections import defaultdict
from typing import Dict

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("uaie.middleware.rate_limiter")

# Default: 10,000 requests per 60 seconds per IP — very permissive
DEFAULT_RATE_LIMIT = 10_000
DEFAULT_WINDOW_SECONDS = 60


class RateLimiterMiddleware:
    """
    Simple sliding-window rate limiter per client IP.

    Uses very high limits by default — intended as a safety net
    against runaway clients, not as a hard usage restriction.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # ip -> list of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxies."""
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode().split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _cleanup_old_entries(self, ip: str, now: float) -> None:
        """Remove request timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]
        if not self._requests[ip]:
            del self._requests[ip]

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Skip rate limiting for health checks
        if path in ("/health", "/docs", "/openapi.json"):
            await self.app(scope, receive, send)
            return

        client_ip = self._get_client_ip(scope)
        now = time.time()

        self._cleanup_old_entries(client_ip, now)

        request_count = len(self._requests.get(client_ip, []))

        if request_count >= self.max_requests:
            logger.warning(
                "Rate limit exceeded for %s: %d/%d requests in %ds window",
                client_ip,
                request_count,
                self.max_requests,
                self.window_seconds,
            )
            body = json.dumps({
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after_seconds": self.window_seconds,
            }).encode("utf-8")

            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                    [b"retry-after", str(self.window_seconds).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        # Record this request
        self._requests[client_ip].append(now)

        await self.app(scope, receive, send)
