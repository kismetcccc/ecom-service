"""Graph routing, per-invocation state and side-effect error boundaries."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.multi_agent.graph import build_multi_agent_graph
from app.multi_agent.planner import TaskPlan
from app.multi_agent.router import RouteDecision
from app.schemas.response import CustomerServiceResponse, IntentType
from app.config.settings import settings


def owner_for(decision):
    response = CustomerServiceResponse(intent=IntentType.OTHER, confidence=1, reply="verified")
    return SimpleNamespace(
        raw_messages=[], router=SimpleNamespace(decide=Mock(return_value=decision)),
        _clarify_route=Mock(return_value=response), _run_return=Mock(return_value=response),
        _plan_tasks=Mock(return_value=TaskPlan((), False)),
        _execute_plan=Mock(return_value=response),
        _review_result=Mock(side_effect=lambda user, result, previous: result),
    )


def initial():
    return {"user_input": "request", "previous_message_ids": set(), "review_enabled": True}


@pytest.mark.parametrize("agent,intents,path", [
    (None, ("order",), ["router", "clarify", "review"]),
    ("postsale", ("greeting",), ["router", "greeting", "review"]),
    ("postsale", ("return",), ["router", "return_workflow", "review"]),
    ("postsale", ("order",), ["router", "planner", "specialists", "review"]),
])
def test_branch_runs_expected_nodes_once(monkeypatch, agent, intents, path):
    monkeypatch.setattr(settings, "multi_agent_workflows_enabled", True)
    owner = owner_for(RouteDecision(agent, intents, 1, "test"))
    graph = build_multi_agent_graph(owner)
    updates = list(graph.stream(initial(), stream_mode="updates"))
    assert [name for update in updates for name in update] == path
    owner.router.decide.assert_called_once()
    owner._review_result.assert_called_once()
    assert owner._execute_plan.call_count == ("specialists" in path)
    assert owner._run_return.call_count == ("return_workflow" in path)


def test_plan_failure_and_previous_result_do_not_leak_to_next_turn():
    owner = owner_for(RouteDecision("postsale", ("order",), 1, "test"))
    blocked = CustomerServiceResponse(intent=IntentType.OTHER, confidence=1, reply="blocked", task_status="blocked")
    owner._plan_tasks.side_effect = [blocked, TaskPlan((), False)]
    graph = build_multi_agent_graph(owner)
    assert graph.invoke(initial())["result"] is blocked
    owner._execute_plan.assert_not_called()
    assert graph.invoke(initial())["result"].reply == "verified"
    owner._execute_plan.assert_called_once()
    assert graph.checkpointer is None


def test_write_failure_propagates_without_retry_or_review(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_workflows_enabled", True)
    owner = owner_for(RouteDecision("postsale", ("return",), 1, "test"))
    owner._run_return.side_effect = RuntimeError("write outcome unknown")
    with pytest.raises(RuntimeError, match="write outcome unknown"):
        build_multi_agent_graph(owner).invoke(initial())
    owner._run_return.assert_called_once()
    owner._review_result.assert_not_called()


def test_business_only_entry_does_not_double_review():
    owner = owner_for(RouteDecision("postsale", ("order",), 1, "test"))
    result = build_multi_agent_graph(owner).invoke({**initial(), "review_enabled": False})
    assert result["result"].reply == "verified"
    owner._review_result.assert_not_called()
