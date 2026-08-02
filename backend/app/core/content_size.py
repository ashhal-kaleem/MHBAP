"""
content_size.py — ASGI middleware that rejects request bodies over a configurable limit.

Prevents large-payload DoS attacks.  Default: 10 MB.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_10_MB = 10 * 1024 * 1024


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds *max_bytes*."""

    def __init__(self, app, max_bytes: int = _10_MB) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        length_header = request.headers.get("content-length")
        if length_header:
            try:
                length = int(length_header)
            except ValueError:
                return Response("Bad Content-Length", status_code=400)
            if length > self._max:
                return Response(
                    f"Request body too large (max {self._max // 1024 // 1024} MB)",
                    status_code=413,
                )
        return await call_next(request)
