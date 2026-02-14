"""
Unified error handling middleware for UAIE.

Catches unhandled exceptions and returns consistent JSON error responses
instead of letting them bubble up as raw 500 errors.

Implemented as pure ASGI middleware (not BaseHTTPMiddleware) to avoid
the known Starlette issue with stacked BaseHTTPMiddleware corrupting
response bodies.
"""

import logging
import traceback
import json

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("uaie.middleware.error_handler")


class ErrorHandlerMiddleware:
    """Catches unhandled exceptions and returns structured JSON error responses."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            path = scope.get("path", "unknown")
            method = scope.get("method", "unknown")
            logger.error("Unhandled exception on %s %s: %s", method, path, exc)
            logger.debug(traceback.format_exc())

            body = json.dumps({
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "path": path,
            }).encode("utf-8")

            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
