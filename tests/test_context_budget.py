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


class FakeMemoryManager:
    """为事务回滚测试提供可观察的短期记忆状态。"""

    def __init__(self, facts=None):
        self.facts = list(facts or [])

    def stm_to_dict(self):
        return {"facts": list(self.facts)}

    def restore_stm(self, data):
        self.facts = list(data.get("facts", []))

    def update_short_term(self, _messages):
        self.facts.append("本轮临时事实")

    def stage_long_term_checkpoint(self, *_args):
        return None

    def commit_long_term_checkpoint(self, _path):
        return None

    def discard_staged_checkpoint(self, _path):
        return None


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
        build_memory_prompt_sections=lambda *_args: []
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
        build_memory_prompt_sections=lambda *_args: []
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
    agent.summary = "existing summary"
    agent.memory_manager = FakeMemoryManager(["existing fact"])

    def reject():
        raise ContextWindowExceeded(101, 100)

    agent._react_loop = reject

    with pytest.raises(ContextWindowExceeded):
        agent.chat("too long")

    assert agent.raw_messages == original
    assert agent.summary == "existing summary"
    assert agent.memory_manager.facts == ["existing fact"]


def test_single_agent_rolls_back_when_summary_generation_fails(monkeypatch):
    agent = EcomAgent.__new__(EcomAgent)
    original = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    agent.raw_messages = list(original)
    agent.summary = "旧摘要"
    agent.memory_manager = FakeMemoryManager(["旧事实"])
    agent.history_keep_recent = 1
    agent.context_budget = manager(keep=1)
    agent.client = object()
    agent.model = "fake-model"
    agent._react_loop = agent._compress_history

    def fail_summary(**_kwargs):
        raise RuntimeError("summary failed")

    monkeypatch.setattr("app.agent.chat.summarize", fail_summary)

    with pytest.raises(RuntimeError, match="summary failed"):
        agent.chat("本轮问题")

    assert agent.raw_messages == original
    assert agent.summary == "旧摘要"
    assert agent.memory_manager.facts == ["旧事实"]


def test_single_agent_rolls_back_when_save_fails(monkeypatch):
    agent = EcomAgent.__new__(EcomAgent)
    original = [{"role": "assistant", "content": "旧回答"}]
    agent.raw_messages = list(original)
    agent.summary = "旧摘要"
    agent.memory_manager = FakeMemoryManager(["旧事实"])
    agent.history_threshold = 999
    agent.session_path = "unused.json"
    agent._react_loop = lambda: "本轮回答"
    agent._extract_structured_response = lambda *_: SimpleNamespace(
        model_dump_json=lambda: '{"reply":"本轮回答"}'
    )

    def fail_save(*_args, **_kwargs):
        raise OSError("save failed")

    monkeypatch.setattr("app.agent.chat.save_session", fail_save)

    with pytest.raises(OSError, match="save failed"):
        agent.chat("本轮问题")

    assert agent.raw_messages == original
    assert agent.summary == "旧摘要"
    assert agent.memory_manager.facts == ["旧事实"]


def test_multi_agent_rolls_back_after_compaction_then_token_recheck_failure():
    orchestrator = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    original = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    orchestrator.raw_messages = list(original)
    orchestrator.summary = "旧摘要"
    orchestrator.memory_manager = FakeMemoryManager(["旧事实"])
    from app.multi_agent.router import RouteDecision
    orchestrator.router = SimpleNamespace(decide=lambda *_: RouteDecision("postsale", ("order",), 1.0, "test"))
    orchestrator.agents = {
        "postsale": SimpleNamespace(name="客服", tool_manager=SimpleNamespace(tool_definitions=[
            {"function": {"name": "query_order"}}, {"function": {"name": "list_user_orders"}},
        ]))
    }

    def fail_after_compaction(_agent):
        # 模拟 _prepare_messages 已完成滚动摘要，但 Token 复检仍然失败。
        orchestrator.raw_messages = [
            {"role": "user", "content": "本轮问题"}
        ]
        orchestrator.summary = "错误的新摘要"
        raise ContextWindowExceeded(101, 100)

    orchestrator._prepare_task_messages = fail_after_compaction

    with pytest.raises(ContextWindowExceeded):
        orchestrator.chat("本轮问题")

    assert orchestrator.raw_messages == original
    assert orchestrator.summary == "旧摘要"
    assert orchestrator.memory_manager.facts == ["旧事实"]


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
        asyncio.run(api.chat(payload, authenticated_user="user-a"))

    assert exc_info.value.status_code == 413
