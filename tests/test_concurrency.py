"""高并发服务层测试，不调用真实模型 API。"""

import asyncio
import json
import threading
import time

from app.agent.tools.manager import ToolManager
from app.schemas.response import CustomerServiceResponse, IntentType
from app.server.middleware import ConcurrencyLimitMiddleware
from app.server.pool import AgentPool, AgentPoolFull, InvalidSessionId


class FakeAgent:
    guard = threading.Lock()
    running = 0
    max_running = 0

    def __init__(self, session_path: str, user_id: str):
        self.session_path = session_path
        self.user_id = user_id
        self.messages = []

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
