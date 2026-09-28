"""按用户和会话隔离的 Agent 池。"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.config.settings import settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class AgentLike(Protocol):
    def chat(self, user_input: str): ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class InvalidSessionId(ValueError):
    pass


class AgentPoolFull(RuntimeError):
    pass


@dataclass
class PoolEntry:
    agent: AgentLike
    last_used: float


def validate_identifier(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise InvalidSessionId(
            f"{field} 只能包含字母、数字、点、下划线和短横线，长度为 1-64"
        )
    return value


def default_agent_factory(session_path: str, user_id: str) -> AgentLike:
    if settings.multi_agent_enabled:
        from app.multi_agent.orchestrator import MultiAgentOrchestrator

        return MultiAgentOrchestrator(session_path=session_path, user_id=user_id)

    from app.agent.chat import EcomAgent

    return EcomAgent(session_path=session_path, user_id=user_id)


class AgentPool:
    """复用会话 Agent；同一用户串行，不同用户可并行。"""

    def __init__(
        self,
        session_dir: str | Path,
        max_sessions: int = 1000,
        agent_factory: Callable[[str, str], AgentLike] = default_agent_factory,
    ):
        self.session_dir = Path(session_dir)
        self.max_sessions = max(1, max_sessions)
        self.agent_factory = agent_factory
        self._entries: dict[tuple[str, str], PoolEntry] = {}
        self._pending: set[tuple[str, str]] = set()
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    def session_path(self, user_id: str, session_id: str) -> Path:
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id")
        return self.session_dir / user_id / f"{session_id}.json"

    async def chat(self, user_id: str, session_id: str, message: str):
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            result = await asyncio.to_thread(entry.agent.chat, message)
            entry.last_used = time.monotonic()
            return result

    async def reset(self, user_id: str, session_id: str) -> bool:
        self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            key = (user_id, session_id)
            async with self._registry_lock:
                entry = self._entries.get(key)
            if entry is not None:
                await asyncio.to_thread(entry.agent.reset)
                entry.last_used = time.monotonic()
                return True

            path = self.session_path(user_id, session_id)
            if path.exists():
                await asyncio.to_thread(path.unlink)
                return True
            return False

    async def close_session(self, user_id: str, session_id: str) -> bool:
        self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            async with self._registry_lock:
                entry = self._entries.pop((user_id, session_id), None)
            if entry is None:
                return False
            await asyncio.to_thread(entry.agent.close)
            return True

    async def close_all(self) -> None:
        async with self._registry_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        if entries:
            await asyncio.gather(
                *(asyncio.to_thread(entry.agent.close) for entry in entries),
                return_exceptions=True,
            )

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        validate_identifier(user_id, "user_id")
        async with self._registry_lock:
            return self._user_locks.setdefault(user_id, asyncio.Lock())

    async def _get_or_create(
        self, user_id: str, session_id: str, path: Path
    ) -> PoolEntry:
        key = (user_id, session_id)
        async with self._registry_lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing
            if len(self._entries) + len(self._pending) >= self.max_sessions:
                raise AgentPoolFull("活跃会话数已达到上限")
            self._pending.add(key)

        try:
            agent = await asyncio.to_thread(
                self.agent_factory, str(path), user_id
            )
        except BaseException:
            async with self._registry_lock:
                self._pending.discard(key)
            raise
        entry = PoolEntry(agent=agent, last_used=time.monotonic())

        async with self._registry_lock:
            self._pending.discard(key)
            # 同一 user_id 已被用户锁保护，这里主要防御未来调用方式变化。
            existing = self._entries.setdefault(key, entry)
        if existing is not entry:
            await asyncio.to_thread(agent.close)
        return existing

    @property
    def active_sessions(self) -> int:
        return len(self._entries)
