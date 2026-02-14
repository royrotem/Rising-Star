"""
Simple in-memory rate limiting middleware for UAIE.

Uses a very high default limit (10,000 requests per minute per IP)
to protect against accidental abuse without affecting normal usage.
No external dependencies required — uses a simple sliding window counter.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("uaie.middleware.rate_limiter")

# Default: 10,000 requests per 60 seconds per IP — very permissive
DEFAULT_RATE_LIMIT = 10_000
DEFAULT_WINDOW_SECONDS = 60


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window rate limiter per client IP.

    Uses very high limits by default — intended as a safety net
    against runaway clients, not as a hard usage restriction.
    """

    def __init__(
        self,
        app,
        max_requests: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # ip -> list of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxies."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_entries(self, ip: str, now: float) -> None:
        """Remove request timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]
        # Prevent unbounded memory growth — drop IPs with no recent requests
        if not self._requests[ip]:
            del self._requests[ip]

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
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
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after_seconds": self.window_seconds,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Record this request
        self._requests[client_ip].append(now)

        return await call_next(request)
