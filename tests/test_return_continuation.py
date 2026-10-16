import json
from types import SimpleNamespace

import pytest

from app.config.settings import settings
from app.multi_agent.router import Router, resume_return_request
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.workflow import run_return_workflow
from tests.test_return_workflow import agents_for, ORDER


def router():
    def unavailable(*args, **kwargs):
        raise RuntimeError("classification unavailable")
    return Router(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable))), "test")


def waiting_history(text="处理退换货"):
    decision = router().decide(text)
    result, _ = run_return_workflow(text, decision, {})
    return [{"role":"user", "content":text}, {"role":"assistant", "content":result.model_dump_json()}]


def test_request_then_bare_order_reaches_confirmation_without_model(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_workflows_enabled", True)
    history = waiting_history()
    assert json.loads(history[-1]["content"])["workflow_status"] == "needs_input"
    history.append({"role":"user", "content":ORDER})
    orchestrator = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    orchestrator.router = router()
    orchestrator.raw_messages = history
    orchestrator.agents, calls = agents_for()
    result = orchestrator._execute_business_turn(ORDER)
    assert result.workflow_status == "waiting_confirmation"
    assert set(calls) == {"query_order", "search_knowledge", "get_return_eligibility"}
    assert "尚未提交" in result.reply


@pytest.mark.parametrize("message", ["不退了", "查物流", f"查订单 {ORDER}", f"{ORDER} ORD-OTHER", "确认", "我要买鞋"])
def test_new_intent_or_multiple_orders_never_resume(message):
    assert resume_return_request(message, waiting_history()) is None


def test_no_pending_state_or_intervening_turn_never_resumes():
    assert resume_return_request(ORDER, []) is None
    history = waiting_history()
    history.extend([{"role":"user", "content":"不退了"}, {"role":"assistant", "content":"好的"}])
    assert resume_return_request(ORDER, history) is None
    assert resume_return_request(ORDER, [{"role":"user", "content":waiting_history()[-1]["content"]}]) is None


def test_exchange_intent_survives_order_followup():
    history = waiting_history("我要申请换货")
    resumed = resume_return_request(ORDER, history)
    assert resumed == f"申请换货 {ORDER}"
    agents, _ = agents_for()
    result, _ = run_return_workflow(resumed, router().decide(ORDER, history), agents)
    assert "换货诉求请联系人工" in result.reply


def test_fast_rule_does_not_route_write_without_deterministic_workflow(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_workflows_enabled", False)
    assert router().decide("处理退换货").agent is None


def test_pending_state_survives_json_session_roundtrip():
    history = json.loads(json.dumps(waiting_history()))
    assert router().decide("ORD-20261012-4C26CE75539D7E36", history).source == "workflow_continuation"
