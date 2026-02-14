"""
Structured request logging middleware for UAIE.

Logs every request with method, path, status code, and duration
for observability and debugging.

Implemented as pure ASGI middleware (not BaseHTTPMiddleware) to avoid
the known Starlette issue with stacked BaseHTTPMiddleware corrupting
response bodies.
"""

import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("uaie.middleware.request_logger")


class RequestLoggerMiddleware:
    """Logs structured information about every HTTP request."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                # Add timing header
                duration_ms = round((time.time() - start_time) * 1000, 2)
                headers = list(message.get("headers", []))
                headers.append([b"x-response-time-ms", str(duration_ms).encode()])
                message = {**message, "headers": headers}
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                duration_ms = round((time.time() - start_time) * 1000, 2)
                method = scope.get("method", "?")
                path = scope.get("path", "?")
                logger.info("%s %s -> %s (%.2fms)", method, path, status_code, duration_ms)
            await send(message)

        await self.app(scope, receive, send_wrapper)
