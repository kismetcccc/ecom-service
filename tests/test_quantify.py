"""Measurement semantics: failed requests, denominators, and unknown values."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.scripts.quantify import http_benchmark, main


def test_live_requires_opt_in():
    with pytest.raises(SystemExit):
        main(["retrieval"])
    with pytest.raises(SystemExit):
        main(["http", "--user-id", "bench", "--live", "--requests", "0"])


def test_http_errors_are_not_successful_throughput(monkeypatch):
    calls = []
    def handler(request):
        if request.url.path == "/v1/auth/me":
            return httpx.Response(200, json={"user_id": "bench"})
        calls.append(request)
        return httpx.Response(429, json={"detail": "busy"})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("TEST_QUANT_TOKEN", "fixture-token")
    args = SimpleNamespace(token_env="TEST_QUANT_TOKEN", timeout=2, url="http://test",
                           user_id="bench", message="hello", warmup=1, requests=3, concurrency=2)
    metrics, rows, metadata = asyncio.run(http_benchmark(args))
    values = {m["key"]: m["value"] for m in metrics}
    assert len(calls) == 4 and len(rows) == 3
    assert values["rps"] > 0 and values["successful_rps"] == 0
    assert values["chat_p95"] is None and values["ttft"] is None
    assert values["request_errors"] == 1
    assert metadata["success_rate"]["denominator"] == 3


def test_http_wrong_identity_stops_before_chat(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"user_id": "different"})), **kw))
    monkeypatch.setenv("TEST_QUANT_TOKEN", "fixture-token")
    args = SimpleNamespace(token_env="TEST_QUANT_TOKEN", timeout=2, url="http://test", user_id="bench")
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(http_benchmark(args))
