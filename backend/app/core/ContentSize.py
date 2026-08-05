"""
content_size.py — ASGI middleware that rejects request bodies over a configurable limit.

Prevents large-payload DoS attacks.  Default: 10 MB.
Handles both Content-Length header and chunked transfer encoding (no header).
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


_10_MB = 10 * 1024 * 1024


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds *max_bytes* — works for both
    Content-Length declared and chunked transfer-encoded requests."""

    def __init__(self, app: ASGIApp, max_bytes: int = _10_MB) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        length_header = request.headers.get("content-length")

        if length_header:
            # Fast path: trust declared Content-Length
            try:
                length = int(length_header)
            except ValueError:
                return Response("Bad Content-Length", status_code=400)
            if length > self._max:
                return Response(
                    f"Request body too large (max {self._max // 1024 // 1024} MB)",
                    status_code=413,
                )
        else:
            # Slow path: chunked or unknown — count bytes as they arrive
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > self._max:
                    return Response(
                        f"Request body too large (max {self._max // 1024 // 1024} MB)",
                        status_code=413,
                    )

        return await call_next(request)
