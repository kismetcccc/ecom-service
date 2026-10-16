"""单 Agent 与多 Agent 共用的 Memory/Session 生命周期服务。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Callable

from openai import APIConnectionError, RateLimitError

from app.server.middleware import request_id_var

from app.console import safe_print


@dataclass
class SessionTurnSnapshot:
    """MR-20：一轮请求开始前的统一事务快照。"""

    messages: list[dict]
    summary: str | None
    short_term: dict
    summary_versions: list[dict]
    summary_audit: list[dict]


class MemorySessionService:
    """MR-20：集中维护保存、回滚、重置、摘要和 Memory 更新语义。"""

    def __init__(self, owner):
        self.owner = owner

    def load(self) -> None:
        loaded = self.owner._memory_load_session()(self.owner.session_path)
        if not loaded:
            return
        self.owner.summary = loaded["summary"]
        self.owner.raw_messages = loaded["messages"]
        self.owner.summary_versions = loaded.get("summary_versions", [])
        self.owner.summary_audit = loaded.get("summary_audit", [])
        if loaded.get("short_term_memory"):
            self.owner.memory_manager.restore_stm(loaded["short_term_memory"])

    def snapshot(self) -> SessionTurnSnapshot:
        return SessionTurnSnapshot(
            messages=deepcopy(self.owner.raw_messages),
            summary=self.owner.summary,
            short_term=deepcopy(self.owner.memory_manager.stm_to_dict()),
            summary_versions=deepcopy(getattr(self.owner, "summary_versions", [])),
            summary_audit=deepcopy(getattr(self.owner, "summary_audit", [])),
        )

    def execute_turn(self, user_input: str):
        """执行模式特有主逻辑，并以完全一致的事务完成 Memory 后处理。"""
        snapshot = self.snapshot()
        self.owner.summary_versions = snapshot.summary_versions
        self.owner.summary_audit = snapshot.summary_audit
        staged_checkpoint = None
        try:
            self.owner.raw_messages.append({"role": "user", "content": user_input})
            result = self.owner._execute_mode_turn(user_input)
            self.owner.memory_manager.update_short_term(
                self.owner.raw_messages[-6:]
            )
            self.owner.raw_messages.append({
                "role": "assistant",
                "content": result.model_dump_json(),
            })
            if len(self.owner.raw_messages) > self.owner.history_threshold:
                try:
                    logging.getLogger(__name__).info(
                        "memory_auto_summary_started request_id=%s", request_id_var.get(),
                    )
                    self.compress_history()
                except (APIConnectionError, RateLimitError) as exc:
                    # Automatic housekeeping must not discard a completed reply.
                    # Keep full history; pre-request context limits remain enforced.
                    logging.getLogger(__name__).warning(
                        "memory_auto_summary_deferred request_id=%s type=%s",
                        request_id_var.get(), type(exc).__name__,
                    )
            staged_checkpoint = self.owner.memory_manager.stage_long_term_checkpoint(
                self.owner.raw_messages,
                self.owner.summary,
                self.owner.session_path,
            )
            self.save()
            self.owner.memory_manager.commit_long_term_checkpoint(staged_checkpoint)
            return result
        except Exception:
            # MR-20：两种模式共享同一个回滚入口，不再分别维护恢复字段。
            self.owner.raw_messages = snapshot.messages
            self.owner.summary = snapshot.summary
            self.owner.memory_manager.restore_stm(snapshot.short_term)
            self.owner.summary_versions = snapshot.summary_versions
            self.owner.summary_audit = snapshot.summary_audit
            self.owner.memory_manager.discard_staged_checkpoint(staged_checkpoint)
            raise

    def save(self) -> None:
        self.owner._memory_save_session()(
            self.owner.session_path,
            self.owner.raw_messages,
            self.owner.summary,
            short_term_memory=self.owner.memory_manager.stm_to_dict(),
            summary_versions=self.owner.summary_versions,
            summary_audit=self.owner.summary_audit,
        )

    def reset(self) -> None:
        self.owner.raw_messages = []
        self.owner.summary = None
        self.owner.summary_versions = []
        self.owner.summary_audit = []
        self.owner.memory_manager.reset_short_term()
        self.owner._memory_delete_session()(self.owner.session_path)

    def clear_short_term(self) -> None:
        self.owner.memory_manager.reset_short_term()
        self.owner.save()

    def mutate_fact(
        self, scope: str, action: str, fact_id: str, actor: str,
        content: str | None = None, category: str | None = None,
    ) -> dict:
        result = self.owner.memory_manager.mutate_fact(
            scope, action, fact_id, actor, content, category
        )
        if scope == "short-term":
            self.owner.save()
        return result

    def compress_history(self) -> None:
        old_messages, recent = self.owner.context_budget.split_history(
            self.owner.raw_messages,
            keep_recent_messages=self.owner.history_keep_recent,
        )
        if old_messages:
            self.summarize_history(old_messages, recent)

    def summarize_history(
        self, old_messages: list[dict], recent: list[dict],
    ) -> None:
        new_summary = self.owner._memory_summarizer()(
            client=self.owner.client,
            model=self.owner.model,
            old_messages=old_messages,
            prev_summary=self.owner.summary,
        )
        self.record_summary_version(new_summary, old_messages)
        self.owner.summary = new_summary
        self.owner.raw_messages = recent
        safe_print(
            f"\n💾 [已压缩 {len(old_messages)} 条老消息 → summary "
            f"({len(new_summary)} 字)]\n"
        )

    def record_summary_version(
        self, summary: str, source_messages: list[dict],
    ) -> None:
        versions = getattr(self.owner, "summary_versions", [])
        previous_id = versions[-1]["version_id"] if versions else None
        next_id = (previous_id or 0) + 1
        versions.append({
            "version_id": next_id,
            "summary": summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_message_count": len(source_messages),
            "source_range": {
                "start": 0,
                "end": max(0, len(source_messages) - 1),
            },
            "previous_version_id": previous_id,
        })
        limit = getattr(self.owner, "summary_version_limit", 10)
        self.owner.summary_versions = versions[-limit:]
        recorder = getattr(
            getattr(self.owner, "memory_manager", None),
            "record_summary_version",
            None,
        )
        if callable(recorder):
            recorder(next_id, len(source_messages))

    def rollback_summary(self, version_id: int, actor: str) -> dict:
        version = next((
            item for item in self.owner.summary_versions
            if item["version_id"] == version_id
        ), None)
        if version is None:
            raise ValueError("摘要版本不存在")
        self.owner.summary = version["summary"]
        self.owner.summary_audit.append({
            "action": "rollback",
            "version_id": version_id,
            "actor": actor,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self.owner.save()
        return version

    def delete_summary_version(self, version_id: int, actor: str) -> None:
        version = next((
            item for item in self.owner.summary_versions
            if item["version_id"] == version_id
        ), None)
        if version is None:
            raise ValueError("摘要版本不存在")
        if version["summary"] == self.owner.summary:
            raise ValueError("不能删除当前生效的摘要版本，请先回滚到其他版本")
        self.owner.summary_versions = [
            item for item in self.owner.summary_versions
            if item["version_id"] != version_id
        ]
        self.owner.summary_audit.append({
            "action": "delete",
            "version_id": version_id,
            "actor": actor,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self.owner.save()


class MemorySessionClientMixin:
    """为两种 Agent 暴露兼容的公共方法，核心实现全部委托给统一服务。"""

    def _memory_session(self) -> MemorySessionService:
        service = getattr(self, "_memory_session_service", None)
        if service is None:
            service = MemorySessionService(self)
            self._memory_session_service = service
        return service

    def chat(self, user_input: str):
        return self._memory_session().execute_turn(user_input)

    def reset(self) -> None:
        self._memory_session().reset()

    def clear_short_term_memory(self) -> None:
        self._memory_session().clear_short_term()

    def delete_long_term_memory(self) -> None:
        self.memory_manager.delete_long_term()

    def set_long_term_memory_consent(self, granted: bool) -> None:
        self.memory_manager.set_long_term_consent(granted)

    def mutate_memory_fact(
        self, scope: str, action: str, fact_id: str, actor: str,
        content: str | None = None, category: str | None = None,
    ) -> dict:
        return self._memory_session().mutate_fact(
            scope, action, fact_id, actor, content, category
        )

    def save(self) -> None:
        self._memory_session().save()

    def _compress_history(self) -> None:
        self._memory_session().compress_history()

    def _summarize_history(
        self, old_messages: list[dict], recent: list[dict],
    ) -> None:
        self._memory_session().summarize_history(old_messages, recent)

    def _record_summary_version(
        self, summary: str, source_messages: list[dict],
    ) -> None:
        self._memory_session().record_summary_version(summary, source_messages)

    def rollback_summary(self, version_id: int, actor: str) -> dict:
        return self._memory_session().rollback_summary(version_id, actor)

    def delete_summary_version(self, version_id: int, actor: str) -> None:
        self._memory_session().delete_summary_version(version_id, actor)
