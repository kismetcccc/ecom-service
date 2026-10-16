"""按用户和会话隔离的 Agent 池。"""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from app.config.settings import settings
from app.identifiers import valid_identifier
from app.server.execution import run_blocking

_T = TypeVar("_T")


class AgentLike(Protocol):
    def chat(self, user_input: str): ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
    def clear_short_term_memory(self) -> None: ...
    def delete_long_term_memory(self) -> None: ...
    def set_long_term_memory_consent(self, granted: bool) -> None: ...
    def rollback_summary(self, version_id: int, actor: str) -> dict: ...
    def delete_summary_version(self, version_id: int, actor: str) -> None: ...
    def mutate_memory_fact(
        self, scope: str, action: str, fact_id: str, actor: str,
        content: str | None = None, category: str | None = None,
    ) -> dict: ...


class InvalidSessionId(ValueError):
    pass


class AgentPoolFull(RuntimeError):
    pass


@dataclass
class PoolEntry:
    agent: AgentLike
    last_used: float


def validate_identifier(value: str, field: str) -> str:
    if not valid_identifier(value):
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
        # CR-12：仅活跃/等待请求持有强引用，空闲用户锁可自动回收。
        self._user_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
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
            result = await run_blocking(entry.agent.chat, message)
            entry.last_used = time.monotonic()
            return result

    async def reset(self, user_id: str, session_id: str) -> bool:
        self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            key = (user_id, os.path.normcase(session_id))
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

    async def clear_short_term(self, user_id: str, session_id: str) -> bool:
        """MR-04：清空指定会话短期记忆，并保留原始消息、摘要和长期记忆。"""
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            await asyncio.to_thread(entry.agent.clear_short_term_memory)
            entry.last_used = time.monotonic()
            return True

    async def delete_long_term(self, user_id: str) -> bool:
        """MR-04：清空该用户所有活跃 Agent 中的长期记忆缓存及持久化数据。"""
        validate_identifier(user_id, "user_id")
        lock = await self._get_user_lock(user_id)
        async with lock:
            async with self._registry_lock:
                entries = [
                    entry for (entry_user, _), entry in self._entries.items()
                    if entry_user == user_id
                ]
            if entries:
                for entry in entries:
                    await asyncio.to_thread(
                        entry.agent.delete_long_term_memory
                    )
                return True

            # 没有活跃 Agent 时直接清理该用户持久化文件和可靠检查点。
            from app.agent.memory.long_term import LongTermMemory

            memory = LongTermMemory(
                user_id=user_id,
                memory_dir=settings.memory_dir,
                max_facts=settings.max_ltm_facts,
                max_interaction_summaries=settings.max_ltm_interaction_summaries,
                interaction_retention_days=settings.ltm_interaction_retention_days,
                encryption_key=settings.memory_encryption_key,
            )
            existed = memory.exists()
            memory.reset()
            checkpoint_dir = Path(settings.memory_dir) / "checkpoints" / user_id
            for path in checkpoint_dir.glob("*.json") if checkpoint_dir.exists() else []:
                path.unlink(missing_ok=True)
                existed = True
            return existed

    async def set_long_term_consent(
        self, user_id: str, granted: bool
    ) -> None:
        """MR-12：同步所有活跃会话的授权缓存；撤回时删除持久化记忆。"""
        validate_identifier(user_id, "user_id")
        lock = await self._get_user_lock(user_id)
        async with lock:
            async with self._registry_lock:
                entries = [
                    entry for (entry_user, _), entry in self._entries.items()
                    if entry_user == user_id
                ]
            if entries:
                for entry in entries:
                    await asyncio.to_thread(
                        entry.agent.set_long_term_memory_consent, granted
                    )
                return

            from app.agent.memory.privacy import MemoryConsentStore

            store = MemoryConsentStore(settings.memory_consent_db_path)
            store.set_granted(user_id, granted)
            if not granted:
                # 已持有用户锁，直接清理，避免再次获取同一 asyncio.Lock。
                from app.agent.memory.long_term import LongTermMemory

                LongTermMemory(
                    user_id=user_id,
                    memory_dir=settings.memory_dir,
                    max_facts=settings.max_ltm_facts,
                    max_interaction_summaries=settings.max_ltm_interaction_summaries,
                    interaction_retention_days=settings.ltm_interaction_retention_days,
                    encryption_key=settings.memory_encryption_key,
                ).reset()
                checkpoint_dir = (
                    Path(settings.memory_dir) / "checkpoints" / user_id
                )
                paths = checkpoint_dir.glob("*.json") if checkpoint_dir.exists() else []
                for path in paths:
                    path.unlink(missing_ok=True)

    async def inspect(
        self,
        user_id: str,
        session_id: str,
        inspector: Callable[[AgentLike], _T],
    ) -> _T:
        """在用户锁内读取会话 Agent 状态，避免与聊天请求并发冲突。"""
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            result = await asyncio.to_thread(inspector, entry.agent)
            entry.last_used = time.monotonic()
            return result

    async def rollback_summary(
        self, user_id: str, session_id: str, version_id: int, actor: str
    ) -> dict:
        """MR-02：在用户锁内回滚摘要，避免与聊天压缩并发修改。"""
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            result = await asyncio.to_thread(
                entry.agent.rollback_summary, version_id, actor
            )
            entry.last_used = time.monotonic()
            return result

    async def delete_summary_version(
        self, user_id: str, session_id: str, version_id: int, actor: str
    ) -> None:
        """MR-02：在会话串行锁内删除摘要版本。"""
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            await asyncio.to_thread(
                entry.agent.delete_summary_version, version_id, actor
            )
            entry.last_used = time.monotonic()

    async def mutate_memory_fact(
        self,
        user_id: str,
        session_id: str,
        scope: str,
        action: str,
        fact_id: str,
        actor: str,
        content: str | None = None,
        category: str | None = None,
    ) -> dict:
        """MR-17：在用户串行锁内变更事实，避免与聊天/沉淀任务竞态。"""
        path = self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            entry = await self._get_or_create(user_id, session_id, path)
            result = await asyncio.to_thread(
                entry.agent.mutate_memory_fact,
                scope, action, fact_id, actor, content, category,
            )
            entry.last_used = time.monotonic()
            return result

    async def close_session(self, user_id: str, session_id: str) -> bool:
        self.session_path(user_id, session_id)
        lock = await self._get_user_lock(user_id)
        async with lock:
            async with self._registry_lock:
                entry = self._entries.pop((user_id, os.path.normcase(session_id)), None)
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

    async def reap_idle(self, idle_seconds: float) -> int:
        """CR-12：仅回收没有在途操作的空闲 Agent；已保存会话仍可恢复。"""
        cutoff = time.monotonic() - idle_seconds
        async with self._registry_lock:
            candidates = [(key, entry) for key, entry in self._entries.items() if entry.last_used < cutoff]
        removed = 0
        for key, expected in candidates:
            lock = await self._get_user_lock(key[0])
            if lock.locked():
                continue
            async with lock:
                async with self._registry_lock:
                    entry = self._entries.get(key)
                    if entry is not expected or entry.last_used >= cutoff:
                        continue
                    self._entries.pop(key)
                await run_blocking(entry.agent.close)
                removed += 1
        return removed

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        validate_identifier(user_id, "user_id")
        async with self._registry_lock:
            return self._user_locks.setdefault(user_id, asyncio.Lock())

    async def _get_or_create(
        self, user_id: str, session_id: str, path: Path
    ) -> PoolEntry:
        # CR-01：Windows 的大小写会话别名必须复用同一个 Agent，避免旧快照覆盖。
        key = (user_id, os.path.normcase(session_id))
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
