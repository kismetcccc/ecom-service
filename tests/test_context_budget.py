"""Context budget unit tests; no model or network calls."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.agent.chat import EcomAgent
from app.agent.context_budget import (
    ContextBudgetManager,
    ContextWindowExceeded,
    estimate_text_tokens,
)
from app.config.settings import Settings
from app.multi_agent.agents import SubAgent
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.server import api


def manager(window: int = 200, reserved: int = 20, keep: int = 2):
    return ContextBudgetManager(window, reserved, keep)


def test_estimator_counts_messages_and_tool_schemas():
    budget = manager()
    messages = [{"role": "user", "content": "你好 hello"}]

    without_tools = budget.estimate(messages)
    with_tools = budget.estimate(
        messages,
        [{"type": "function", "function": {"name": "query_order"}}],
    )

    assert estimate_text_tokens("hello") == 5
    assert estimate_text_tokens("你好") == 2
    assert with_tools > without_tools > 0


def test_over_budget_error_reports_estimate_and_limit():
    budget = manager(window=40, reserved=10)
    messages = [{"role": "user", "content": "x" * 100}]

    with pytest.raises(ContextWindowExceeded) as exc_info:
        budget.ensure_within_budget(messages)

    assert exc_info.value.estimated_tokens > exc_info.value.input_budget
    assert exc_info.value.input_budget == 30


def test_history_split_never_separates_tool_call_from_results():
    budget = manager(keep=3)
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1"}],
    }
    messages = [
        {"role": "user", "content": "old"},
        tool_call,
        {"role": "tool", "tool_call_id": "call-1", "content": "result-1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "result-2"},
        {"role": "assistant", "content": "intermediate"},
        {"role": "user", "content": "current"},
    ]

    old, recent = budget.split_history(messages)

    assert old == [messages[0]]
    assert recent[0] is tool_call
    assert recent[1:3] == messages[2:4]


def test_invalid_context_settings_fail_at_startup():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            context_window_tokens=100,
            context_reserved_output_tokens=100,
        )

    with pytest.raises(ValidationError):
        Settings(_env_file=None, context_keep_recent_messages=0)


def test_single_agent_compacts_before_model_call(monkeypatch):
    agent = EcomAgent.__new__(EcomAgent)
    agent.raw_messages = [
        {"role": "user", "content": "旧消息" * 300},
        {"role": "assistant", "content": "旧回复" * 300},
        {"role": "user", "content": "当前问题"},
    ]
    agent.summary = None
    agent.context_budget = manager(window=3800, reserved=200, keep=1)
    agent.tool_manager = SimpleNamespace(tool_definitions=[])
    agent.skill_manager = None
    agent.memory_manager = SimpleNamespace(
        build_memory_prompt_sections=lambda: []
    )
    agent.client = object()
    agent.model = "fake-model"

    monkeypatch.setattr("app.agent.chat.summarize", lambda **_: "旧对话摘要")

    messages = agent._prepare_messages()

    assert agent.summary == "旧对话摘要"
    assert agent.raw_messages == [{"role": "user", "content": "当前问题"}]
    assert agent.context_budget.estimate(messages) <= agent.context_budget.input_budget


def test_multi_agent_uses_same_preflight_compaction(monkeypatch):
    orchestrator = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    orchestrator.raw_messages = [
        {"role": "user", "content": "旧消息" * 200},
        {"role": "assistant", "content": "旧回复" * 200},
        {"role": "user", "content": "当前问题"},
    ]
    orchestrator.summary = None
    orchestrator.context_budget = manager(window=1200, reserved=100, keep=1)
    orchestrator.skill_manager = None
    orchestrator.memory_manager = SimpleNamespace(
        build_memory_prompt_sections=lambda: []
    )
    orchestrator.client = object()
    orchestrator.model = "fake-model"
    subagent = SimpleNamespace(
        system_prompt="system",
        tool_manager=SimpleNamespace(tool_definitions=[]),
    )

    monkeypatch.setattr(
        "app.multi_agent.orchestrator.summarize",
        lambda **_: "旧对话摘要",
    )

    messages = orchestrator._prepare_messages(subagent)

    assert orchestrator.summary == "旧对话摘要"
    assert orchestrator.raw_messages == [
        {"role": "user", "content": "当前问题"}
    ]
    assert orchestrator.context_budget.estimate(messages) <= 1100


def test_rejected_turn_does_not_pollute_single_agent_history():
    agent = EcomAgent.__new__(EcomAgent)
    original = [{"role": "assistant", "content": "existing"}]
    agent.raw_messages = list(original)

    def reject():
        raise ContextWindowExceeded(101, 100)

    agent._react_loop = reject

    with pytest.raises(ContextWindowExceeded):
        agent.chat("too long")

    assert agent.raw_messages == original


def test_subagent_applies_reserved_output_limit():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="ok", tool_calls=None
                ))]
            )

    context_budget = manager(window=200, reserved=20)
    subagent = SubAgent(
        name="fake",
        system_prompt="system",
        tool_manager=SimpleNamespace(tool_definitions=[]),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
        model="fake-model",
        temperature=0.0,
        context_budget=context_budget,
    )

    reply, _ = subagent.handle([{"role": "user", "content": "hello"}])

    assert reply == "ok"
    assert captured["max_tokens"] == 20


def test_http_api_maps_context_overflow_to_413(monkeypatch):
    async def reject(*_):
        raise ContextWindowExceeded(101, 100)

    monkeypatch.setattr(api.pool, "chat", reject)
    payload = api.ChatRequest(user_id="user-a", message="hello")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.chat(payload))

    assert exc_info.value.status_code == 413
