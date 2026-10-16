"""高并发服务层测试，不调用真实模型 API。"""

import asyncio
import json
import subprocess
import sys
import threading
import time

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.agent.tools.manager import ToolManager
from app.schemas.response import CustomerServiceResponse, IntentType
from app.server.middleware import ConcurrencyLimitMiddleware
from app.server.api import ChatRequest, _memory_snapshot, _skill_snapshot
from app.server.pool import AgentPool, AgentPoolFull, InvalidSessionId


class FakeAgent:
    guard = threading.Lock()
    running = 0
    max_running = 0

    def __init__(self, session_path: str, user_id: str):
        self.session_path = session_path
        self.user_id = user_id
        self.messages = []
        self.short_term = ["temporary"]
        self.long_term = ["persistent"]

    def chat(self, message: str) -> CustomerServiceResponse:
        with self.guard:
            type(self).running += 1
            type(self).max_running = max(type(self).max_running, type(self).running)
        try:
            time.sleep(0.05)
            self.messages.append(message)
            return CustomerServiceResponse(
                intent=IntentType.OTHER,
                confidence=1.0,
                reply=f"{self.user_id}:{message}",
                requires_human=False,
            )
        finally:
            with self.guard:
                type(self).running -= 1

    def reset(self) -> None:
        self.messages.clear()
        self.short_term.clear()

    def clear_short_term_memory(self) -> None:
        self.short_term.clear()

    def delete_long_term_memory(self) -> None:
        self.long_term.clear()

    def set_long_term_memory_consent(self, granted: bool) -> None:
        self.long_term_consent = granted

    def close(self) -> None:
        pass


def make_pool():
    FakeAgent.running = 0
    FakeAgent.max_running = 0
    return AgentPool(
        session_dir="unused-test-sessions",
        agent_factory=lambda path, user: FakeAgent(path, user),
    )


def test_different_users_run_concurrently():
    async def scenario():
        pool = make_pool()
        results = await asyncio.gather(
            pool.chat("user-a", "main", "A"),
            pool.chat("user-b", "main", "B"),
        )
        assert {item.reply for item in results} == {"user-a:A", "user-b:B"}
        assert FakeAgent.max_running == 2

    asyncio.run(scenario())


def test_same_user_is_serialized_and_sessions_are_isolated():
    async def scenario():
        pool = make_pool()
        await asyncio.gather(
            pool.chat("user-a", "first", "A"),
            pool.chat("user-a", "second", "B"),
        )
        assert FakeAgent.max_running == 1
        assert pool.active_sessions == 2
        assert pool.session_path("user-a", "first") != pool.session_path(
            "user-a", "second"
        )

    asyncio.run(scenario())


def test_inspect_reads_the_isolated_session_agent():
    async def scenario():
        pool = make_pool()
        await pool.chat("user-a", "main", "A")
        messages = await pool.inspect(
            "user-a", "main", lambda agent: list(agent.messages)
        )
        assert messages == ["A"]

    asyncio.run(scenario())


def test_mr04_memory_clear_operations_have_separate_scopes():
    async def scenario():
        pool = make_pool()
        await pool.chat("user-a", "main", "A")

        await pool.clear_short_term("user-a", "main")
        state = await pool.inspect(
            "user-a",
            "main",
            lambda agent: (
                list(agent.messages),
                list(agent.short_term),
                list(agent.long_term),
            ),
        )
        assert state == (["A"], [], ["persistent"])

        await pool.delete_long_term("user-a")
        state = await pool.inspect(
            "user-a",
            "main",
            lambda agent: (
                list(agent.messages),
                list(agent.short_term),
                list(agent.long_term),
            ),
        )
        assert state == (["A"], [], [])

        await pool.reset("user-a", "main")
        state = await pool.inspect(
            "user-a",
            "main",
            lambda agent: (
                list(agent.messages),
                list(agent.short_term),
                list(agent.long_term),
            ),
        )
        assert state == ([], [], [])

    asyncio.run(scenario())


def test_session_capability_snapshots_are_json_ready():
    class Fact:
        id = "long-1"
        category = "preference"
        content = "喜欢蓝色"
        memory_key = "preference.color"
        status = "active"
        superseded_by = ""
        source = "user"
        source_session = "session-a"
        source_message_id = "message-1"
        evidence = "喜欢蓝色"
        extraction_model = "test-model"
        confidence = 0.9
        last_verified_at = "2026-10-02T00:00:00+00:00"

    class SkillManager:
        enabled = True

        @staticmethod
        def get_catalog():
            return [{"name": "recommend-products", "description": "商品推荐"}]

    class MemoryManager:
        memory_enabled = True
        long_term_consent = True
        stm = type("STM", (), {
            "facts": ["用户叫小明"],
            "to_dict": lambda self: {"facts": [{
                "id": "fact-1",
                "content": "用户叫小明",
                "category": "identity",
                "source": "user",
                "confidence": 0.9,
                "created_at": "2026-10-02T00:00:00+00:00",
                "updated_at": "2026-10-02T00:00:00+00:00",
                "expires_at": None,
                "status": "active",
                "memory_key": "identity.name",
            }]},
        })()
        ltm = type(
            "LTM",
            (),
            {
                "user_id": "user-a",
                "facts": [Fact()],
                "interaction_summaries": [{"summary": "咨询过耳机"}],
            },
        )()

    agent = type(
        "Agent",
        (),
        {"skill_manager": SkillManager(), "memory_manager": MemoryManager()},
    )()

    assert _skill_snapshot(agent)["skills"][0]["name"] == "recommend-products"
    memory = _memory_snapshot(agent)
    assert memory["short_term"][0]["content"] == "用户叫小明"
    assert memory["long_term"] == [
        {
            "id": "long-1",
            "category": "preference",
            "content": "喜欢蓝色",
            "memory_key": "preference.color",
            "status": "active",
            "superseded_by": "",
            "source": "user",
            "source_session": "session-a",
            "source_message_id": "message-1",
            "evidence": "喜欢蓝色",
            "extraction_model": "test-model",
            "confidence": 0.9,
            "last_verified_at": "2026-10-02T00:00:00+00:00",
        }
    ]


@pytest.mark.parametrize(
    "command",
    ["memory", " MEMORY ", "memory on", "memory off", "reset", "skills"],
)
def test_chat_api_rejects_client_control_commands(command):
    with pytest.raises(ValidationError, match="客户端控制命令"):
        ChatRequest(user_id="user-a", session_id="main", message=command)


def test_session_identifiers_reject_path_traversal():
    pool = make_pool()
    try:
        pool.session_path("../other", "main")
    except InvalidSessionId:
        pass
    else:
        raise AssertionError("必须拒绝目录穿越 user_id")


def test_session_capacity_is_atomic():
    async def scenario():
        pool = AgentPool(
            session_dir="unused-test-sessions",
            max_sessions=1,
            agent_factory=lambda path, user: FakeAgent(path, user),
        )
        results = await asyncio.gather(
            pool.chat("user-a", "main", "A"),
            pool.chat("user-b", "main", "B"),
            return_exceptions=True,
        )
        assert sum(isinstance(item, AgentPoolFull) for item in results) == 1
        assert pool.active_sessions == 1

    asyncio.run(scenario())


def test_concurrency_middleware_returns_503_when_queue_times_out():
    async def scenario():
        async def slow_app(scope, receive, send):
            await asyncio.sleep(0.05)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = ConcurrencyLimitMiddleware(
            slow_app, max_concurrency=1, queue_timeout=0.01
        )
        scope = {"type": "http", "path": "/v1/chat", "method": "POST"}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        first_messages = []
        second_messages = []

        async def send_first(message):
            first_messages.append(message)

        async def send_second(message):
            second_messages.append(message)

        first = asyncio.create_task(middleware(scope, receive, send_first))
        await asyncio.sleep(0.005)
        await middleware(scope, receive, send_second)
        await first

        assert first_messages[0]["status"] == 200
        assert second_messages[0]["status"] == 503

    asyncio.run(scenario())


def test_tool_context_is_isolated_per_agent():
    manager_a = ToolManager(
        local_tool_overrides={
            "recall_user_memory": lambda query="": {"user": "A"}
        }
    )
    manager_b = ToolManager(
        local_tool_overrides={
            "recall_user_memory": lambda query="": {"user": "B"}
        }
    )
    result_a = json.loads(manager_a.execute_tool("recall_user_memory", {}))
    result_b = json.loads(manager_b.execute_tool("recall_user_memory", {}))
    assert result_a["user"] == "A"
    assert result_b["user"] == "B"


def test_local_chat_import_does_not_require_mcp_dependency():
    """The API imports Agent on first chat; MCP-disabled mode must remain importable."""
    script = """
import builtins
original_import = builtins.__import__
def reject_mcp(name, *args, **kwargs):
    if name == 'mcp' or name.startswith('mcp.'):
        raise ModuleNotFoundError("No module named 'mcp'")
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_mcp
from app.agent.chat import EcomAgent
from app.agent.tools.manager import ToolManager
assert ToolManager(use_mcp=False).tool_definitions
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_chat_api_logs_unexpected_failure_without_exposing_details(monkeypatch):
    from app.server import api

    async def broken_chat(*_args):
        raise RuntimeError("private backend diagnostic")

    monkeypatch.setattr(api.pool, "chat", broken_chat)
    api.app.dependency_overrides[api.require_authenticated_user] = lambda: "user-a"
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/v1/chat",
                json={"user_id": "user-a", "session_id": "main", "message": "你好"},
            )
    finally:
        api.app.dependency_overrides.pop(api.require_authenticated_user, None)

    assert response.status_code == 500
    assert response.headers["x-request-id"] in response.json()["detail"]
    assert "private backend diagnostic" not in response.text
