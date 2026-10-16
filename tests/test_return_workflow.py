import json
import threading
from types import SimpleNamespace

import pytest

from app.multi_agent.workflow import run_return_workflow
from app.multi_agent.router import RouteDecision
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.config.settings import settings


ORDER = "ORD-20240115-001"
DECISION = RouteDecision("postsale", ("return",), .95, "model")


def agents_for(overrides=None, barrier=None):
    calls = []
    values = {
        "query_order": {"success": True, "order": {"order_id": ORDER}},
        "search_knowledge": {"success": True, "results": [{
            "chunk_id": "policy-1", "text": "演示退货规则", "doc": "退货政策", "status": "demo",
        }]},
        "get_return_eligibility": {"success": True, "eligible": True, "order_id": ORDER},
    }
    values.update(overrides or {})

    class Manager:
        def __init__(self, names):
            self.names = names

        def available_tool_names(self):
            return self.names

        def execute_tool(self, name, arguments):
            assert name in self.names
            calls.append(name)
            if barrier and name in {"query_order", "search_knowledge"}:
                barrier.wait(timeout=3)
            value = values[name]
            if isinstance(value, Exception):
                raise value
            return json.dumps(value)

    return {
        "postsale": SimpleNamespace(tool_manager=Manager({"query_order", "get_return_eligibility"})),
        "presale": SimpleNamespace(tool_manager=Manager({"search_knowledge"})),
    }, calls


def test_workflow_parallel_prerequisites_and_confirmation_boundary():
    agents, calls = agents_for(barrier=threading.Barrier(2))
    result, messages = run_return_workflow(f"申请退货 {ORDER}", DECISION, agents, parallel=True)
    assert result.workflow_status == "waiting_confirmation"
    assert calls[-1] == "get_return_eligibility"
    assert set(calls) == {"query_order", "search_knowledge", "get_return_eligibility"}
    assert [step.status for step in result.task_outcomes] == ["verified", "verified", "verified", "waiting_confirmation"]
    assert result.sources[0].chunk_id == "policy-1"
    assert result.knowledge_status == "VERIFIED_DEMO"
    assert "尚未提交" in result.reply
    assert len(messages) == 6
    for index in range(0, 6, 2):
        assert messages[index]["tool_calls"][0]["id"] == messages[index + 1]["tool_call_id"]


@pytest.mark.parametrize("failure", [
    {"success": False, "error_code": "UNAVAILABLE"},
    {"success": True, "results": []},
    {"success": True, "results": "malformed"},
    {"success": True, "results": None},
    {"success": "true"}, [], TimeoutError(),
])
def test_policy_failure_blocks_dependent_steps(failure):
    agents, calls = agents_for({"search_knowledge": failure})
    result, _ = run_return_workflow(f"申请退货 {ORDER}", DECISION, agents)
    assert result.workflow_status == "blocked"
    assert "get_return_eligibility" not in calls
    assert result.task_outcomes[2].status == "skipped"


@pytest.mark.parametrize("value", [False, "true", None])
def test_ineligible_or_malformed_eligibility_never_advances(value):
    agents, _ = agents_for({"get_return_eligibility": {
        "success": True, "order_id": ORDER, "eligible": value,
    }})
    result, _ = run_return_workflow(f"申请退款 {ORDER}", DECISION, agents)
    assert result.workflow_status == "blocked"
    assert result.task_outcomes[-1].status == "skipped"


@pytest.mark.parametrize("message", ["我要退货", f"退货 {ORDER} 和 ORD-20240120-002"])
def test_ambiguous_order_has_no_tool_calls(message):
    agents, calls = agents_for()
    result, messages = run_return_workflow(message, DECISION, agents)
    assert result.workflow_status == "needs_input"
    assert not calls and not messages


def test_foreign_order_stops_eligibility():
    agents, calls = agents_for({"query_order": {"success": False, "error_code": "NOT_FOUND"}})
    result, _ = run_return_workflow(f"申请退货 {ORDER}", DECISION, agents)
    assert result.workflow_status == "blocked"
    assert "get_return_eligibility" not in calls


def test_orchestrator_uses_workflow_and_persists_evidence(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_workflows_enabled", True)
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.router = SimpleNamespace(decide=lambda *_: DECISION)
    orchestrator.agents, _ = agents_for()
    orchestrator.raw_messages = []
    result = orchestrator._execute_mode_turn(f"退货 {ORDER}")
    assert result.workflow_id
    assert len(orchestrator.raw_messages) == 6
