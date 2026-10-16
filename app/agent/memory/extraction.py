"""LLM 事实提取：从对话中抽取短期/长期记忆事实。

模式与 app/agent/summarizer.py 一致：格式化对话 → 调用 LLM → 解析结果。
"""

from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime
from typing import TYPE_CHECKING, Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from app.commands import is_control_command
from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from app.agent.memory.long_term import MemoryFact


class MemoryExtractionError(RuntimeError):
    """MR-13：提取结果经过重试后仍不符合 Schema。"""


class ShortTermFactExtraction(BaseModel):
    """MR-06：模型必须为每条短期事实返回结构化元数据。"""

    content: str = Field(min_length=1, max_length=500)
    category: Literal["identity", "preference", "intent", "emotion", "order", "other"]
    source: Literal["user", "tool", "model"]
    confidence: float = Field(ge=0.0, le=1.0)
    memory_key: str = Field(default="", max_length=100)
    source_message_id: str = Field(default="", max_length=100)
    evidence: str = Field(default="", max_length=500)


class ShortTermExtraction(BaseModel):
    facts: list[ShortTermFactExtraction] = Field(default_factory=list, max_length=30)


class LongTermFactExtraction(BaseModel):
    content: str = Field(min_length=1, max_length=500)
    category: Literal["identity", "preference", "behavior", "issue", "other"]
    # MR-07：相同业务属性使用同一稳定键，供纠正和替换使用。
    memory_key: str = Field(default="", max_length=100)
    source: Literal["user", "tool", "model"] = "model"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_message_id: str = Field(default="", max_length=100)
    evidence: str = Field(default="", max_length=500)


class LongTermExtraction(BaseModel):
    facts: list[LongTermFactExtraction] = Field(default_factory=list, max_length=20)
    interaction_summary: str = Field(min_length=1, max_length=1000)


_ExtractionModel = TypeVar("_ExtractionModel", bound=BaseModel)


def _extract_json_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    schema: type[_ExtractionModel],
    max_attempts: int = 3,
) -> tuple[_ExtractionModel, dict]:
    """MR-13：严格解析 JSON；格式错误时附带错误信息重试，不保存失败文本。"""
    last_error: Exception | None = None
    retry_instruction = ""
    usage = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
    started = time.perf_counter()
    for _attempt in range(max_attempts):
        usage["attempts"] += 1
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt + retry_instruction},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        response_usage = getattr(response, "usage", None)
        usage["prompt_tokens"] += int(
            getattr(response_usage, "prompt_tokens", 0) or 0
        )
        usage["completion_tokens"] += int(
            getattr(response_usage, "completion_tokens", 0) or 0
        )
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            usage["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 2
            )
            return schema.model_validate_json(raw), usage
        except (ValidationError, json.JSONDecodeError) as error:
            last_error = error
            retry_instruction = (
                "\n\n上一次输出未通过 JSON Schema 校验。"
                "请只返回符合要求的合法 JSON，不要添加解释或代码块。"
            )
    raise MemoryExtractionError("记忆提取结果连续校验失败") from last_error


def _message_id(message: dict, index: int) -> str:
    """MR-11：为没有原生 ID 的历史消息生成稳定、不可逆的来源标识。"""
    explicit = message.get("id") or message.get("tool_call_id")
    if explicit:
        return str(explicit)
    raw = f"{index}:{message.get('role', '')}:{message.get('content', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _source_evidence(messages: list[dict], source: str) -> tuple[str, str]:
    role = {"user": "user", "tool": "tool", "model": "assistant"}.get(source)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == role and message.get("content"):
            return _message_id(message, index), str(message["content"])[:300]
    return "", ""


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。"""
    lines = []
    for index, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content") or ""
        message_id = _message_id(msg, index)

        if role == "user":
            # 兼容旧会话：历史客户端曾可能把 memory/reset 等本地命令
            # 错发给模型。控制命令不属于用户事实，不能进入记忆提取。
            if is_control_command(content):
                continue
            lines.append(f"用户[message_id={message_id}]：{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    lines.append(f"客服：[调用工具 {name}]")
            if content:
                lines.append(f"客服[message_id={message_id}]：{content}")
        elif role == "tool":
            display = content if len(content) <= 200 else content[:200] + "..."
            lines.append(f"[工具结果 message_id={message_id}] {display}")

    return "\n".join(lines)


def extract_short_term_facts(
    client: OpenAI,
    model: str,
    recent_messages: list[dict],
    existing_facts: list,
    telemetry: dict | None = None,
) -> list[dict]:
    """MR-06：从最近对话中提取结构化短期记忆事实增量。"""
    transcript = _build_transcript(recent_messages)
    if not transcript.strip():
        return []

    def describe(fact) -> str:
        if isinstance(fact, str):
            return fact
        if isinstance(fact, dict):
            return f"[{fact.get('category', 'other')}] {fact.get('content', '')}"
        return f"[{getattr(fact, 'category', 'other')}] {getattr(fact, 'content', '')}"

    existing_text = (
        "\n".join(f"- {describe(fact)}" for fact in existing_facts)
        if existing_facts else "（暂无）"
    )
    prompt = STM_EXTRACTION_PROMPT.format(existing_facts=existing_text)

    data, usage = _extract_json_with_retry(
        client,
        model,
        prompt,
        # MR-16：对话被序列化为不可信数据字段，避免与提取指令混在同一语义层。
        json.dumps({"untrusted_conversation": transcript}, ensure_ascii=False),
        ShortTermExtraction,
    )
    if telemetry is not None:
        telemetry.update(usage)
    results = []
    for fact in data.facts:
        item = fact.model_dump()
        fallback_id, fallback_evidence = _source_evidence(
            recent_messages, item["source"]
        )
        item["source_message_id"] = item["source_message_id"] or fallback_id
        item["evidence"] = item["evidence"] or fallback_evidence
        item["extraction_model"] = model
        # MR-11：不同证据来源使用不同可信上限，模型推断不得伪装成高置信事实。
        confidence_caps = {"tool": 0.98, "user": 0.90, "model": 0.50}
        item["confidence"] = min(item["confidence"], confidence_caps[item["source"]])
        results.append(item)
    return results


def extract_long_term_facts(
    client: OpenAI,
    model: str,
    messages: list[dict],
    summary: str | None,
    existing_facts: list,
    telemetry: dict | None = None,
) -> tuple[list[MemoryFact], str]:
    """从完整会话中提取长期记忆事实 + 交互摘要。"""
    from app.agent.memory.long_term import MemoryFact

    parts = []
    if summary:
        parts.append(f"【对话摘要】\n{summary}")

    transcript = _build_transcript(messages)
    if transcript:
        parts.append(f"【对话内容】\n{transcript}")

    if not parts:
        return [], ""

    existing_text = (
        "\n".join(f"- [{f.category}] {f.content}" for f in existing_facts)
        if existing_facts
        else "（暂无）"
    )
    prompt = LTM_EXTRACTION_PROMPT.format(existing_ltm=existing_text)

    data, usage = _extract_json_with_retry(
        client,
        model,
        prompt,
        # MR-16：摘要和原始对话仅作为 JSON 数据，不允许其中内容改变提取规则。
        json.dumps(
            {"untrusted_memory_source": "\n\n".join(parts)},
            ensure_ascii=False,
        ),
        LongTermExtraction,
    )
    if telemetry is not None:
        telemetry.update(usage)

    now = datetime.now().isoformat(timespec="seconds")
    new_facts = []
    for item in data.facts:
        content = item.content.strip()
        category = item.category
        if content:
            source_id, evidence = _source_evidence(messages, item.source)
            source_id = item.source_message_id or source_id
            evidence = item.evidence or evidence
            confidence = min(
                item.confidence,
                {"tool": 0.98, "user": 0.90, "model": 0.50}[item.source],
            )
            new_facts.append(MemoryFact(
                content=content,
                category=category,
                created_at=now,
                memory_key=item.memory_key.strip(),
                source=item.source,
                source_message_id=source_id,
                evidence=evidence,
                extraction_model=model,
                confidence=confidence,
                status=(
                    "active" if source_id and item.source != "model"
                    else "pending_confirmation"
                ),
                last_verified_at=now if source_id else "",
            ))

    return new_facts, data.interaction_summary.strip()
