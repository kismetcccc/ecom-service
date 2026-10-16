"""Trusted tool identity; model-supplied arguments never decide the user."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config.settings import settings
from app.server.auth import AuthService


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    session_id: str = "default"


@lru_cache(maxsize=1)
def _auth_service() -> AuthService:
    return AuthService(settings.auth_database_path, settings.auth_secret_key)


def issue_context_token(context: ToolContext) -> str:
    return _auth_service().issue_token(context.user_id)


def verify_context_token(token: str) -> ToolContext:
    return ToolContext(user_id=_auth_service().verify_token(token))
