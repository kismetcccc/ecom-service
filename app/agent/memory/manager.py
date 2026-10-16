"""记忆管理器：统一管理短期记忆和长期记忆。

EcomAgent 和 MultiAgentOrchestrator 通过此管理器与记忆系统交互。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import RLock
from typing import Optional
from dataclasses import asdict

from openai import OpenAI

from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.extraction import MemoryExtractionError
from app.agent.memory.short_term import ShortTermMemory
from app.agent.memory.privacy import MemoryConsentStore
from app.agent.memory.observability import memory_observability


# MR-03：长期记忆提取在线程池中执行；检查点先落盘，进程异常时可在下次启动重放。
_CHECKPOINT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="memory-checkpoint",
)


class MemoryManager:
    """记忆管理器：统一管理短期记忆和长期记忆。"""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        memory_enabled: bool = True,
        max_ltm_facts: int = 50,
        max_interaction_summaries: int = 100,
        interaction_retention_days: int = 180,
        require_consent: bool = True,
        consent_db_path: str = "app/sessions/memory_control.db",
        encryption_key: str = "",
        stm_ttl_hours: dict[str, int] | None = None,
        ltm_ttl_days: dict[str, int] | None = None,
        ltm_confirmation_days: int = 180,
        retrieval_top_k: int = 5,
        session_id: str = "default",
    ):
        self.client = client
        self.model = model
        self.memory_enabled = memory_enabled
        self.user_id = user_id
        self.session_id = session_id
        self.last_short_term_stats: dict = {}

        self.stm = ShortTermMemory(ttl_hours=stm_ttl_hours)
        self.ltm = LongTermMemory(
            user_id=user_id,
            memory_dir=memory_dir,
            max_facts=max_ltm_facts,
            max_interaction_summaries=max_interaction_summaries,
            interaction_retention_days=interaction_retention_days,
            encryption_key=encryption_key,
            ttl_days=ltm_ttl_days,
            confirmation_days=ltm_confirmation_days,
            retrieval_top_k=retrieval_top_k,
        )
        self.require_consent = require_consent
        self.consent_store = MemoryConsentStore(consent_db_path)
        self.long_term_consent = (
            not require_consent or self.consent_store.is_granted(user_id)
        )
        self._checkpoint_lock = RLock()
        self._checkpoint_futures: set[Future] = set()
        self._checkpoint_dir = Path(memory_dir) / "checkpoints" / user_id

        if self.memory_enabled and self.long_term_consent:
            self.ltm.load()
            self._resume_pending_checkpoints()

    @staticmethod
    def _request_id() -> str:
        """MR-18：HTTP 场景复用请求追踪 ID，CLI 场景安全返回空字符串。"""
        try:
            from app.server.middleware import request_id_var
            return request_id_var.get()
        except ImportError:
            return ""

    def _observe(self, event: str, **values) -> None:
        memory_observability.record(
            event,
            request_id=self._request_id(),
            user_id=self.user_id,
            session_id=self.session_id,
            storage_facts=len(self.ltm.facts) + len(self.stm.facts),
            **values,
        )

    @staticmethod
    def should_extract_short_term(recent_messages: list[dict]) -> tuple[bool, str]:
        """MR-15：规则预筛选明显没有记忆价值的寒暄，避免额外模型调用。"""
        user_text = next((
            str(message.get("content", "")).strip()
            for message in reversed(recent_messages)
            if message.get("role") == "user"
        ), "")
        if not user_text:
            return False, "no_user_message"
        normalized = re.sub(r"[\s，。！？!?,.]", "", user_text).casefold()
        greetings = {
            "你好", "您好", "嗨", "hello", "hi", "谢谢", "多谢",
            "好的", "好", "嗯", "再见", "拜拜", "在吗",
        }
        if normalized in greetings:
            return False, "chitchat"
        # 权威订单/物流工具的只读结果不属于用户偏好，也无需复制进记忆。
        # 原始会话仍保留本轮输入与工具轨迹，连续追问不会因此丢上下文。
        pure_read = re.search(
            r"(?:查|查询|查看|看看|看下).*(?:订单|物流|快递)|"
            r"(?:订单|物流|快递).*(?:状态|进度|到哪|发货)",
            user_text,
        )
        pure_policy_read = re.search(
            r"(?:查|查询|查看|了解|咨询|问).*(?:政策|规则|条件|时效)|"
            r"(?:政策|规则|条件|时效).*(?:是什么|怎么规定|如何规定)",
            user_text,
        )
        personal_fact = re.search(
            r"我叫|叫我|称呼|名字|喜欢|偏好|不喜欢|预算|尺码|尺寸|颜色|"
            r"会员|地址|电话|以后|记住|不要再",
            user_text,
        )
        if (pure_read or pure_policy_read) and not personal_fact:
            reason = "authoritative_policy_query" if pure_policy_read else "authoritative_read_query"
            return False, reason
        if any(message.get("role") == "tool" for message in recent_messages):
            return True, "tool_result"
        memory_markers = (
            "我叫", "叫我", "称呼", "名字", "喜欢", "偏好", "不喜欢",
            "预算", "尺码", "尺寸", "颜色", "会员", "地址", "电话",
            "订单", "物流", "快递", "退货", "退款", "换货", "投诉",
            "以后", "记住", "不要再", "正在找", "想买", "需要",
        )
        if any(marker in user_text for marker in memory_markers):
            return True, "memory_keyword"
        return False, "no_memory_signal"

    def update_short_term(self, recent_messages: list[dict]) -> dict:
        """MR-15：按需更新短期记忆；失败只记录状态，不阻断正式回复。"""
        started = time.perf_counter()
        if not self.memory_enabled:
            self.last_short_term_stats = {
                "triggered": False, "reason": "disabled", "latency_ms": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0,
            }
            self._observe("short_term_extract_skipped", reason="disabled")
            return self.last_short_term_stats
        should_extract, reason = self.should_extract_short_term(recent_messages)
        if not should_extract:
            self.last_short_term_stats = {
                "triggered": False, "reason": reason, "latency_ms": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "facts_created": 0,
            }
            self._observe("short_term_extract_skipped", reason=reason)
            return self.last_short_term_stats
        before_count = len(self.stm.facts)
        telemetry: dict = {}
        try:
            self.stm.update(
                self.client, self.model, recent_messages, telemetry
            )
            self.last_short_term_stats = {
                "triggered": True,
                "reason": reason,
                "success": True,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
                "prompt_tokens": telemetry.get("prompt_tokens", 0),
                "completion_tokens": telemetry.get("completion_tokens", 0),
                "attempts": telemetry.get("attempts", 1),
                "facts_created": max(0, len(self.stm.facts) - before_count),
                "conflicts": telemetry.get("conflicts", 0),
            }
            self._observe("short_term_extract", **self.last_short_term_stats)
        except Exception as error:
            # MR-13/MR-15：Memory 是旁路增强，模型、网络或格式异常不得回滚主回复。
            self.last_short_term_stats = {
                "triggered": True,
                "reason": reason,
                "success": False,
                "error_type": type(error).__name__,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
                "prompt_tokens": telemetry.get("prompt_tokens", 0),
                "completion_tokens": telemetry.get("completion_tokens", 0),
                "facts_created": 0,
            }
            self._observe("short_term_extract", **self.last_short_term_stats)
        return self.last_short_term_stats

    def build_memory_prompt_sections(self, current_query: str = "") -> list[dict]:
        """MR-10：按当前问题生成相关 Memory system prompt。"""
        started = time.perf_counter()
        if not self.memory_enabled:
            return []

        sections = []
        ltm_section = (
            self.ltm.build_prompt_section(current_query)
            if self.long_term_consent else None
        )
        if ltm_section:
            sections.append({"role": "system", "content": ltm_section})
        stm_section = self.stm.build_prompt_section()
        if stm_section:
            sections.append({"role": "system", "content": stm_section})
        self._observe(
            "memory_retrieval",
            latency_ms=(time.perf_counter() - started) * 1000,
            recalled_facts=getattr(self.ltm, "last_retrieval_count", 0),
            prompt_sections=len(sections),
        )
        return sections

    def consolidate_to_long_term(
        self, messages: list[dict], summary: Optional[str],
    ) -> None:
        """会话结束时，将本次对话的关键事实巩固到长期记忆。"""
        # MR-12：即使关闭会话，也不能绕过用户的长期记忆授权。
        if not self.memory_enabled or not self.long_term_consent:
            return
        started = time.perf_counter()
        try:
            stats = self.ltm.extract_and_save(
                self.client, self.model, messages, summary,
            )
            self._observe(
                "long_term_consolidation",
                latency_ms=(time.perf_counter() - started) * 1000,
                **(stats or {}),
            )
        except Exception as error:
            self._observe(
                "long_term_consolidation",
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_type=type(error).__name__,
            )
            raise

    def stage_long_term_checkpoint(
        self,
        messages: list[dict],
        summary: Optional[str],
        source_session: str,
    ) -> Path | None:
        """MR-03：先原子落盘待处理任务，主会话保存成功后再异步提交。"""
        if (
            not self.memory_enabled
            or not self.long_term_consent
            or (not messages and not summary)
        ):
            return None

        canonical = json.dumps(
            {"messages": messages, "summary": summary},
            ensure_ascii=False,
            sort_keys=True,
        )
        checkpoint_id = hashlib.sha256(
            f"{self.ltm.generation}\n{source_session}\n{canonical}".encode("utf-8")
        ).hexdigest()
        payload = {
            "version": 1,
            "checkpoint_id": checkpoint_id,
            "generation": self.ltm.generation,
            "source_session": source_session,
            "messages": messages,
            "summary": summary,
        }

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"{checkpoint_id}.json"
        if path.exists() or self.ltm.has_processed_checkpoint(checkpoint_id):
            return None
        tmp_path = path.with_suffix(".json.tmp")
        # MR-12：可靠检查点可能包含原始对话，必须和长期 Memory 一样加密落盘。
        tmp_path.write_bytes(self.ltm.cipher.encrypt_json(payload))
        os.replace(tmp_path, path)
        return path

    def commit_long_term_checkpoint(self, path: Path | None) -> None:
        """MR-03：会话已提交后异步提取；提交失败时保留磁盘任务供重启恢复。"""
        if path is None:
            return
        try:
            future = _CHECKPOINT_EXECUTOR.submit(
                self._process_checkpoint, path
            )
        except RuntimeError:
            return
        self._checkpoint_futures.add(future)
        future.add_done_callback(self._checkpoint_futures.discard)

    @staticmethod
    def discard_staged_checkpoint(path: Path | None) -> None:
        """MR-03：主会话提交失败时撤销尚未生效的检查点。"""
        if path is not None:
            path.unlink(missing_ok=True)

    def flush_checkpoints(self) -> None:
        """MR-03：关闭 Agent 时等待异步任务，并同步重试仍待处理的检查点。"""
        for future in list(self._checkpoint_futures):
            try:
                future.result()
            except Exception:
                pass
        for path in self._pending_checkpoint_paths():
            try:
                self._process_checkpoint(path)
            except Exception:
                # 保留任务文件，下一次启动继续重试。
                continue

    def _resume_pending_checkpoints(self) -> None:
        """MR-03：服务重启后自动重放未完成检查点。"""
        for path in self._pending_checkpoint_paths():
            self.commit_long_term_checkpoint(path)

    def _pending_checkpoint_paths(self) -> list[Path]:
        if not self._checkpoint_dir.exists():
            return []
        return sorted(self._checkpoint_dir.glob("*.json"))

    def _process_checkpoint(self, path: Path) -> None:
        # 同一个用户的检查点串行处理，避免同一进程中的长期记忆对象互相覆盖。
        with self._checkpoint_lock:
            started = time.perf_counter()
            try:
                payload = self.ltm.cipher.decrypt_json(path.read_bytes())
                # CR-05：恢复旧检查点时也校验删除代次和最新授权。
                if (payload.get("generation", 0) != self.ltm.store.generation(self.ltm.user_id)
                        or (self.require_consent and not self.consent_store.is_granted(self.ltm.user_id))):
                    path.unlink(missing_ok=True)
                    return
                checkpoint_id = payload["checkpoint_id"]
                if not self.ltm.has_processed_checkpoint(checkpoint_id):
                    stats = self.ltm.extract_and_save(
                        self.client,
                        self.model,
                        payload.get("messages", []),
                        payload.get("summary"),
                        checkpoint_id=checkpoint_id,
                        source_session=payload.get("source_session", ""),
                    )
                    self._observe(
                        "long_term_checkpoint",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        **(stats or {}),
                    )
                path.unlink(missing_ok=True)
            except Exception as error:
                self._observe(
                    "long_term_checkpoint",
                    success=False,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error_type=type(error).__name__,
                )
                raise

    def reset_short_term(self) -> None:
        """重置短期记忆（会话内重置时调用）。"""
        self.stm.reset()

    def delete_long_term(self) -> None:
        """MR-04：显式删除长期记忆及待处理检查点，防止删除后被异步任务恢复。"""
        if not self.memory_enabled:
            return
        with self._checkpoint_lock:
            for path in self._pending_checkpoint_paths():
                path.unlink(missing_ok=True)
            self.ltm.reset()

    def set_long_term_consent(self, granted: bool) -> None:
        """MR-12：持久化用户授权；撤回授权时同步删除长期数据与待办任务。"""
        self.consent_store.set_granted(self.ltm.user_id, granted)
        self.long_term_consent = granted
        if granted:
            self.ltm.load()
            self._resume_pending_checkpoints()
        else:
            self.delete_long_term()

    def reset_all(self) -> None:
        """重置所有记忆（短期+长期）。"""
        self.stm.reset()
        self.ltm.reset()

    def stm_to_dict(self) -> dict:
        return self.stm.to_dict()

    def restore_stm(self, data: dict) -> None:
        self.stm = ShortTermMemory.from_dict(data, ttl_hours=self.stm.ttl_hours)

    def mutate_fact(
        self,
        scope: str,
        action: str,
        fact_id: str,
        actor: str,
        content: str | None = None,
        category: str | None = None,
    ) -> dict:
        """MR-17：统一执行短期/长期事实的纠正、删除和确认。"""
        target = self.stm if scope == "short-term" else self.ltm
        if scope not in {"short-term", "long-term"}:
            raise ValueError("未知记忆范围")
        if scope == "long-term" and not self.long_term_consent:
            raise ValueError("长期记忆未授权")
        if action == "update":
            fact = target.update_fact(fact_id, actor, content, category)
        elif action == "delete":
            fact = target.delete_fact(fact_id, actor)
        elif action == "confirm":
            fact = target.confirm_fact(fact_id, actor)
        else:
            raise ValueError("未知记忆操作")
        self._observe(
            "memory_fact_mutation",
            action=action,
            scope=scope,
            fact_id=fact_id,
        )
        return asdict(fact)

    def metrics_snapshot(self) -> dict:
        """MR-18：返回当前用户聚合指标，不包含记忆正文。"""
        snapshot = memory_observability.snapshot(self.user_id)
        snapshot["storage"] = {
            "short_term_facts": len(self.stm.facts),
            "long_term_facts": len(self.ltm.facts),
            "interaction_summaries": len(self.ltm.interaction_summaries),
            "pending_checkpoints": len(self._pending_checkpoint_paths()),
        }
        return snapshot

    def record_summary_version(
        self, version_id: int, source_message_count: int,
    ) -> None:
        """MR-18：摘要正文不入日志，只记录版本和来源消息数量。"""
        self._observe(
            "summary_version_created",
            version_id=version_id,
            source_message_count=source_message_count,
        )
