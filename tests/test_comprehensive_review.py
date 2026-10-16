"""综合审查中的隔离、恢复及取消契约；全部使用临时数据和假 Agent。"""

import asyncio
import json
import threading

import pytest

from app.identifiers import valid_identifier
from app.server.auth import AuthService, AuthError
from app.server.pool import AgentPool
from app.agent.storage import load_session, save_session, SessionCorruptedError
from app.agent.memory.store import MemoryStateStore, StaleMemoryGeneration


def test_account_case_alias_cannot_register(tmp_path):
    auth = AuthService(str(tmp_path / "auth.db"), "x" * 32)
    auth.register("Alice", "example-password")
    with pytest.raises(AuthError):
        auth.register("alice", "another-password")
    assert auth.verify_token(auth.login("Alice", "example-password")) == "Alice"


@pytest.mark.parametrize("name", ["user.", "CON", "aux.txt", "LPT1", "..", "../user"])
def test_storage_identifiers_reject_windows_aliases(name):
    assert not valid_identifier(name)


def test_corrupt_session_is_preserved(tmp_path):
    path = tmp_path / "session.json"
    path.write_text('{"messages":', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(SessionCorruptedError):
        load_session(str(path))
    assert path.read_bytes() == original


def test_session_atomic_roundtrip(tmp_path):
    path = str(tmp_path / "session.json")
    save_session(path, [{"role": "user", "content": "你好"}], None)
    from pathlib import Path
    assert "你好".encode() not in Path(path).read_bytes()
    assert load_session(path)["messages"][0]["content"] == "你好"
    assert not list(tmp_path.glob("*.tmp"))


def test_old_memory_writer_cannot_resurrect_deleted_state(tmp_path):
    first = MemoryStateStore(tmp_path / "memory.db")
    second = MemoryStateStore(tmp_path / "memory.db")
    encode = lambda value: json.dumps(value).encode()
    merge = lambda old, new: new
    first.merge_and_save("user-a", {"facts": ["old"]}, json.loads, encode, merge)
    second.delete("user-a")
    with pytest.raises(StaleMemoryGeneration):
        first.merge_and_save("user-a", {"facts": ["stale"]}, json.loads, encode, merge)
    assert second.load("user-a") is None
    first.merge_and_save("user-a", {"facts": ["new"]}, json.loads, encode, merge,
                         expected_generation=second.generation("user-a"))
    assert json.loads(second.load("user-a")[0])["facts"] == ["new"]


def test_cancelled_chat_keeps_user_lock_until_thread_finishes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Agent:
        def chat(self, message):
            calls.append(message)
            if message == "first":
                entered.set()
                assert release.wait(5)
            return message

    async def scenario():
        pool = AgentPool(tmp_path, agent_factory=lambda *_: Agent())
        first = asyncio.create_task(pool.chat("user-a", "default", "first"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            second = asyncio.create_task(pool.chat("user-a", "default", "second"))
            await asyncio.sleep(0.05)
            assert calls == ["first"]
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == "second"

    asyncio.run(scenario())


def test_logout_revokes_only_the_selected_token(tmp_path):
    auth = AuthService(str(tmp_path / "auth.db"), "x" * 32)
    first = auth.register("alice", "example-password")
    second = auth.login("alice", "example-password")
    auth.revoke_token(first)
    with pytest.raises(AuthError):
        auth.verify_token(first)
    assert auth.verify_token(second) == "alice"


def test_legacy_session_is_encrypted_on_next_save(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "legacy-secret"}]}))
    data = load_session(str(path))
    save_session(str(path), data["messages"], data["summary"])
    assert b"legacy-secret" not in path.read_bytes()
    assert load_session(str(path))["messages"] == data["messages"]


def test_idle_pool_entry_closes_without_deleting_persisted_session(tmp_path):
    closed = []

    class Agent:
        def chat(self, message):
            return message

        def close(self):
            closed.append(True)

    async def scenario():
        pool = AgentPool(tmp_path, agent_factory=lambda *_: Agent())
        await pool.chat("user-a", "default", "first")
        entry = next(iter(pool._entries.values()))
        entry.last_used = 0
        assert await pool.reap_idle(1) == 1
        assert pool.active_sessions == 0
        assert closed == [True]

    asyncio.run(scenario())


def test_authentication_attempts_are_rate_limited():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.server.middleware import AuthenticationLimitMiddleware
    app = FastAPI()
    app.add_middleware(AuthenticationLimitMiddleware, attempts_per_minute=2)

    @app.post("/v1/auth/login")
    def login():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post("/v1/auth/login").status_code == 200
        assert client.post("/v1/auth/login").status_code == 200
        response = client.post("/v1/auth/login")
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"


def test_binary_signing_key_is_not_stripped(tmp_path):
    from app.security_keys import load_or_create_key
    key = b" " + b"x" * 30 + b"\n"
    path = tmp_path / "auth.secret"
    assert load_or_create_key(path, lambda: key, 32) == key
    assert load_or_create_key(path, lambda: b"z" * 32, 32) == key


def test_auth_http_identity_and_logout(tmp_path, monkeypatch):
    import app.server.api as api
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "auth_service", AuthService(str(tmp_path / "auth.db"), "x" * 32))
    token = api.auth_service.register("alice", "example-password")
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(api.app) as client:
        assert client.get("/v1/auth/me", headers=headers).json() == {"user_id": "alice"}
        assert client.post("/v1/auth/logout", headers=headers).status_code == 204
        assert client.get("/v1/auth/me", headers=headers).status_code == 401


def test_upstream_auth_failure_does_not_return_user_auth_error(monkeypatch):
    import app.server.api as api
    import httpx
    from openai import AuthenticationError
    from fastapi.testclient import TestClient

    async def fail(*_args):
        raise AuthenticationError("invalid upstream key", response=httpx.Response(
            401, request=httpx.Request("POST", "https://example.invalid")), body=None)

    monkeypatch.setattr(api.pool, "chat", fail)
    monkeypatch.setattr(api.app, "dependency_overrides", {api.require_authenticated_user: lambda: "alice"})
    with TestClient(api.app) as client:
        response = client.post("/v1/chat", json={"user_id": "alice", "message": "hello"})
    assert response.status_code == 502
    assert "OPENAI_API_KEY" not in response.text
