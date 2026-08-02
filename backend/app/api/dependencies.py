"""dependencies.py — FastAPI reusable auth and DB dependencies."""
from __future__ import annotations

from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status, Query, WebSocket, WebSocketException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token, is_token_blacklisted
from app.db.session import get_db  # re-export for convenience

_bearer = HTTPBearer(auto_error=False)


def get_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """Return the raw bearer token string. Raises 401 if the header is missing."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """Validate Bearer JWT, return user_id (sub claim). Raises 401 if invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    if await is_token_blacklisted(payload.get("jti")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    request.state.user_id = user_id
    return user_id


async def get_ws_current_user(
    websocket: WebSocket,
    access_token: Optional[str] = Query(None),
) -> str:
    """Validate Bearer JWT for WebSockets. Returns user_id or raises WebSocketException."""
    token = access_token
    if not token:
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token")

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))

    if await is_token_blacklisted(payload.get("jti")):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token has been revoked")

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload")

    return user_id


def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """Like get_current_user but returns None instead of 401 for anonymous."""
    if credentials is None:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id:
            request.state.user_id = user_id
        return user_id
    except ValueError:
        return None


# ── Role-based authorization ──────────────────────────────────────────────────

# Canonical roles understood by the project.
ROLE_PARTICIPANT = "participant"
ROLE_EDUCATOR = "educator"
ROLE_RESEARCHER = "researcher"

# Roles that may execute runner jobs.
RUNNER_ALLOWED_ROLES = {ROLE_PARTICIPANT, ROLE_EDUCATOR, ROLE_RESEARCHER}


async def get_current_user_with_role(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Tuple[str, str]:
    """
    Return (user_id, role) for the authenticated user.
    Fetches the user record from the DB to get the authoritative role.
    Raises 401 if the user record is missing or inactive.
    """
    import uuid as _uuid
    from app.services.user_service import get_user

    try:
        uid = _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier in token",
        )

    user = await get_user(db, uid)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or inactive",
        )
    return user_id, user.role


def require_roles(allowed: List[str]):
    """
    Dependency factory — raises HTTP 403 if the authenticated user's role
    is not in ``allowed``.

    Usage::

        @router.post("/...")
        async def endpoint(
            _auth = Depends(require_roles(["researcher", "educator"])),
        ):
            ...

    Returns the (user_id, role) tuple so callers can use it directly.
    """
    allowed_set = set(allowed)

    async def _check(
        user_id_role: Tuple[str, str] = Depends(get_current_user_with_role),
    ) -> Tuple[str, str]:
        user_id, role = user_id_role
        if role not in allowed_set:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' is not permitted to perform this action",
            )
        return user_id, role

    return _check
