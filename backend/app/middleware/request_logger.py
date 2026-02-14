"""
Structured request logging middleware for UAIE.

Logs every request with method, path, status code, and duration
for observability and debugging.
"""

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("uaie.middleware.request_logger")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Logs structured information about every HTTP request."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "%s %s -> %s (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        # Add timing header for downstream consumers
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response
