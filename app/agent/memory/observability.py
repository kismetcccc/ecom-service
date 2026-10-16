"""Memory 可观测性：无敏感正文的结构化事件与聚合指标。"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import json
import logging
from threading import RLock
from typing import Any

logger = logging.getLogger("memory.lifecycle")


class MemoryObservability:
    """MR-18：进程内线程安全指标，适用于并发 HTTP 与后台检查点线程。"""

    def __init__(self, max_events: int = 1000):
        self._lock = RLock()
        self._counters: Counter[str] = Counter()
        self._latency_total_ms = 0.0
        self._latency_samples = 0
        self._events: deque[dict] = deque(maxlen=max_events)

    def record(
        self,
        event: str,
        *,
        request_id: str = "",
        user_id: str = "",
        session_id: str = "",
        success: bool = True,
        latency_ms: float = 0.0,
        **values: Any,
    ) -> None:
        # 仅允许标量指标进入日志，避免意外记录对话、事实正文或证据。
        safe_values = {
            key: value for key, value in values.items()
            if isinstance(value, (str, int, float, bool, type(None)))
            and key not in {"content", "evidence", "messages", "summary"}
        }
        item = {
            "event": event,
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "success": success,
            "latency_ms": round(float(latency_ms), 2),
            **safe_values,
        }
        with self._lock:
            self._counters["events_total"] += 1
            self._counters[f"event.{event}"] += 1
            self._counters["success_total" if success else "failure_total"] += 1
            self._counters["prompt_tokens"] += int(safe_values.get("prompt_tokens", 0) or 0)
            self._counters["completion_tokens"] += int(safe_values.get("completion_tokens", 0) or 0)
            self._counters["facts_created"] += int(safe_values.get("facts_created", 0) or 0)
            self._counters["conflicts_total"] += int(safe_values.get("conflicts", 0) or 0)
            self._latency_total_ms += float(latency_ms)
            self._latency_samples += 1
            self._events.append(item)
        logger.info("memory_event=%s", json.dumps(item, ensure_ascii=False))

    def snapshot(self, user_id: str | None = None) -> dict:
        with self._lock:
            events = [
                deepcopy(item) for item in self._events
                if user_id is None or item["user_id"] == user_id
            ]
            if user_id is None:
                counters = dict(self._counters)
                latency_total = self._latency_total_ms
                latency_samples = self._latency_samples
            else:
                counters = Counter()
                latency_total = 0.0
                latency_samples = 0
                for item in events:
                    counters["events_total"] += 1
                    counters[f"event.{item['event']}"] += 1
                    counters["success_total" if item["success"] else "failure_total"] += 1
                    for key in ("prompt_tokens", "completion_tokens", "facts_created", "conflicts"):
                        target = "conflicts_total" if key == "conflicts" else key
                        counters[target] += int(item.get(key, 0) or 0)
                    latency_total += float(item.get("latency_ms", 0))
                    latency_samples += 1
                counters = dict(counters)
        total = counters.get("events_total", 0)
        success = counters.get("success_total", 0)
        return {
            "counters": counters,
            "success_rate": round(success / total, 4) if total else 1.0,
            "average_latency_ms": round(latency_total / latency_samples, 2)
            if latency_samples else 0.0,
            "token_total": counters.get("prompt_tokens", 0)
            + counters.get("completion_tokens", 0),
            "conflict_rate": round(
                counters.get("conflicts_total", 0)
                / max(1, counters.get("facts_created", 0)), 4
            ),
            "recent_events": events[-100:],
        }

    def reset(self) -> None:
        """仅供隔离测试清空进程内指标。"""
        with self._lock:
            self._counters.clear()
            self._latency_total_ms = 0.0
            self._latency_samples = 0
            self._events.clear()


memory_observability = MemoryObservability()
