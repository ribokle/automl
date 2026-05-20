"""Bearer-token auth dependency.

When ``API_AUTH_TOKEN`` is set in the environment, every request to a route
guarded by :func:`require_auth` must carry ``Authorization: Bearer <token>``.
When the env var is unset, the dependency is a no-op so the dev server keeps
working without configuration.

The token is read on every call rather than cached so test fixtures (and
``os.environ`` flips in CI) take effect immediately.
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from core.config import get_settings


def _expected_token() -> str | None:
    secret = get_settings().api_auth_token
    if not secret:
        return None
    token = secret.get_secret_value().strip()
    return token or None


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    expected = _expected_token()
    if not expected:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization[7:].strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
