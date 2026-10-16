"""离线验证限流、冷却和元数据降级，不消耗真实模型额度。"""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI, RateLimitError

from app.config.llm_transport import GuardedModelClient, ModelGate
from app.config.settings import settings
from app.multi_agent.orchestrator import MultiAgentOrchestrator


def test_gate_bounds_concurrency_and_frequency(monkeypatch):
    monkeypatch.setattr(settings, 'llm_max_inflight', 1)
    monkeypatch.setattr(settings, 'llm_requests_per_minute', 1)
    gate = ModelGate()
    assert gate.enter()
    assert not gate.enter()
    gate.leave()
    assert not gate.enter()
    assert gate.active == 0


def test_upstream_rate_limit_cools_all_clients_without_retry():
    handler = Mock(return_value=httpx.Response(429, json={'error': {'message': 'busy'}}))
    gate = ModelGate()
    with GuardedModelClient(transport=httpx.MockTransport(handler), gate=gate) as http:
        client = OpenAI(api_key='test', http_client=http, max_retries=0)
        for _ in range(2):
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model='test', messages=[])
    assert handler.call_count == 1
    assert gate.active == 0


def test_transport_failure_releases_slot():
    handler = Mock(side_effect=httpx.ReadTimeout('test'))
    gate = ModelGate()
    with GuardedModelClient(transport=httpx.MockTransport(handler), gate=gate) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get('https://example.invalid')
    assert gate.active == 0


def test_successful_sdk_response_is_preserved():
    body = {'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'test',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': 'ok'}}]}
    gate = ModelGate()
    with GuardedModelClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=body)), gate=gate) as http:
        client = OpenAI(api_key='test', http_client=http, max_retries=0)
        response = client.chat.completions.create(model='test', messages=[])
        assert response.choices[0].message.content == 'ok'
    assert gate.active == 0


def test_multi_agent_metadata_needs_no_model_client():
    from app.multi_agent.router import RouteDecision
    result = MultiAgentOrchestrator._response_from_decision(
        RouteDecision('postsale', ('order',), 0.9, 'model'), '订单已发货')
    assert result.reply == '订单已发货'
    assert result.confidence == 0.9


def test_read_failure_has_visible_reply():
    agent = object.__new__(MultiAgentOrchestrator)
    future = Mock()
    future.result.side_effect = RuntimeError('failed')
    task = SimpleNamespace(read_only=True, key='service')
    result = agent._collect_task(task, future)
    assert result.blocked
    assert result.text.strip()
