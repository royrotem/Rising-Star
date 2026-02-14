"""
Unified error handling middleware for UAIE.

Catches unhandled exceptions and returns consistent JSON error responses
instead of letting them bubble up as raw 500 errors.
"""

import logging
import traceback
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("uaie.middleware.error_handler")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns structured JSON error responses."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log the full traceback for debugging
            logger.error(
                "Unhandled exception on %s %s: %s",
                request.method,
                request.url.path,
                exc,
            )
            logger.debug(traceback.format_exc())

            # Return a clean JSON error (no internal details leaked)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred. Please try again.",
                    "path": request.url.path,
                },
            )
