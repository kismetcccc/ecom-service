"""Provider-neutral context token budgeting and history boundaries."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


class ContextWindowExceeded(ValueError):
    """Raised when a request cannot fit after eligible history is compacted."""

    def __init__(self, estimated_tokens: int, input_budget: int):
        self.estimated_tokens = estimated_tokens
        self.input_budget = input_budget
        super().__init__(
            "上下文过长：预计输入 "
            f"{estimated_tokens} tokens，当前预算为 {input_budget} tokens。"
            "请缩短本次输入、减少附加内容或重置会话后重试。"
        )


def estimate_text_tokens(value: Any) -> int:
    """Conservatively estimate tokens for OpenAI-compatible providers.

    Exact tokenizers differ across providers. ASCII is counted one character per
    token, while non-ASCII text is estimated from UTF-8 bytes. This intentionally
    overestimates ordinary English and Chinese so budget checks fail safely.
    """
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)

    ascii_count = sum(1 for char in value if ord(char) < 128)
    non_ascii_bytes = sum(
        len(char.encode("utf-8")) for char in value if ord(char) >= 128
    )
    return ascii_count + math.ceil(non_ascii_bytes / 3)


def _message_groups(messages: Sequence[dict]) -> list[list[dict]]:
    """Group assistant tool calls with all immediately following tool results."""
    groups: list[list[dict]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


@dataclass(frozen=True)
class ContextBudgetManager:
    """Estimate model input size and choose history eligible for summarization."""

    window_tokens: int
    reserved_output_tokens: int
    keep_recent_messages: int

    @property
    def input_budget(self) -> int:
        return self.window_tokens - self.reserved_output_tokens

    def estimate(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
    ) -> int:
        # Chat protocols add role/separator metadata around every message.
        total = 3
        for message in messages:
            total += 4
            total += estimate_text_tokens(message.get("role", ""))
            total += estimate_text_tokens(message.get("content", ""))
            total += estimate_text_tokens(message.get("name", ""))
            total += estimate_text_tokens(message.get("tool_call_id", ""))
            total += estimate_text_tokens(message.get("tool_calls", []))
        if tools:
            total += 8 + estimate_text_tokens(list(tools))
        return total

    def ensure_within_budget(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
    ) -> int:
        estimated = self.estimate(messages, tools)
        if estimated > self.input_budget:
            raise ContextWindowExceeded(estimated, self.input_budget)
        return estimated

    def split_history(
        self,
        messages: Sequence[dict],
        keep_recent_messages: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Return (old, recent) without splitting a tool-call/result exchange."""
        keep = max(1, keep_recent_messages or self.keep_recent_messages)
        groups = _message_groups(messages)
        if len(groups) <= 1:
            return [], list(messages)

        recent_group_count = 0
        recent_message_count = 0
        for group in reversed(groups):
            # Always keep at least one older group available for compression.
            if recent_message_count >= keep and recent_group_count > 0:
                break
            recent_group_count += 1
            recent_message_count += len(group)

        if recent_group_count >= len(groups):
            return [], list(messages)

        old_groups = groups[:-recent_group_count]
        recent_groups = groups[-recent_group_count:]
        old = [message for group in old_groups for message in group]
        recent = [message for group in recent_groups for message in group]
        return old, recent
