import json
from types import SimpleNamespace

import pytest

from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, READ_ONLY_TOOLS, SubAgent
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.planner import build_plan
from app.multi_agent.router import Router, RouteDecision
from app.schemas.response import IntentType
from tests.test_agent_quality import trace, reviewer_with, APPROVE, REVISE


@pytest.mark.parametrize("result", ['broken', '[]', 'null', '{}', '{"success":"false"}', '{"success":1}'])
def test_malformed_result_blocks_following_write_without_retry(result):
    calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments="{}"))
             for i, name in enumerate(["query_order", "apply_refund"])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=calls))]))))
    executed = []
    manager = SimpleNamespace(tool_definitions=[], execute_tool=lambda name, args: executed.append(name) or result)
    budget = SimpleNamespace(reserved_output_tokens=100, ensure_within_budget=lambda *_: None)
    _, messages = SubAgent("test", "test", manager, client, "test", 0, budget).handle([], max_steps=1)
    assert executed == ["query_order"]
    results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    assert [item["error_code"] for item in results] == ["INVALID_TOOL_RESULT", "DEPENDENCY_BLOCKED"]


@pytest.mark.parametrize("arguments", ['broken', '[]', 'null'])
def test_invalid_arguments_never_reach_tool(arguments):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="apply_refund", arguments=arguments))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]))))
    executed = []
    manager = SimpleNamespace(tool_definitions=[], execute_tool=lambda *args: executed.append(args))
    budget = SimpleNamespace(reserved_output_tokens=100, ensure_within_budget=lambda *_: None)
    _, messages = SubAgent("test", "test", manager, client, "test", 0, budget).handle([], max_steps=1)
    assert not executed
    assert json.loads(messages[-1]["content"])["error_code"] == "INVALID_TOOL_ARGUMENTS"


@pytest.mark.parametrize("query", ["快递三天没更新", "显示签收但我没收到", "订单和物流状态不一致", "物流一直不动"])
def test_delivery_anomalies_use_dedicated_readonly_agent(query):
    decision = Router(None, "unused").decide(query)
    assert decision.agent == "fulfillment"
    assert "delivery_exception" in decision.intents
    task = build_plan(decision).tasks[0]
    assert task.read_only and task.agent == "fulfillment"
    assert {"query_order", "query_logistics"} <= task.required_tools
    assert AGENT_CONFIGS["fulfillment"]["tools"] <= READ_ONLY_TOOLS


def test_simple_logistics_stays_postsale_and_mixed_refund_keeps_write_scope():
    assert Router(None, "unused").decide("查物流到哪了").agent == "postsale"
    task = build_plan(RouteDecision("postsale", ("delivery_exception", "return"), .9, "model")).tasks[0]
    assert task.agent == "postsale"
    assert not task.read_only
    assert {"query_order", "query_logistics", "search_knowledge"} <= task.required_tools


def test_composite_diagnosis_and_product_keeps_both_tasks():
    plan = build_plan(RouteDecision("fulfillment", ("delivery_exception", "product"), .9, "model"))
    assert [task.agent for task in plan.tasks] == ["fulfillment", "presale"]
    assert plan.parallel


def fake_orchestrator(messages):
    orch = object.__new__(MultiAgentOrchestrator)
    decision = RouteDecision("fulfillment", ("delivery_exception",), .99, "rule:v2")
    orch.router = SimpleNamespace(decide=lambda *_: decision)
    tools = AGENT_CONFIGS["fulfillment"]["tools"]
    orch.agents = {"fulfillment": SimpleNamespace(
        name="履约诊断", tool_manager=SimpleNamespace(available_tool_names=lambda: tools),
        handle=lambda *args, **kwargs: ("物流记录已签收；您报告未收到，需要核实签收凭证。", messages),
    )}
    orch.raw_messages = []
    orch.max_react_steps = 3
    orch._prepare_task_messages = lambda _: []
    return orch


def test_diagnosis_requires_matched_order_and_logistics_records():
    messages = trace() + trace("query_logistics", {"success": True, "logistics": {"status": "delivered"}})
    orch = fake_orchestrator(messages)
    result = orch._execute_mode_turn("订单 ORD-001 显示签收没收到")
    assert result.task_status == "single"
    assert result.intent == IntentType.DELIVERY_EXCEPTION
    assert result.task_outcomes[0].agent == "fulfillment"


@pytest.mark.parametrize("messages", [[], trace(), trace() + trace("query_logistics", {"success": True, "logistics": {"status": "delivered"}}, order="ORD-002")])
def test_missing_or_mismatched_evidence_blocks_diagnosis(messages):
    result = fake_orchestrator(messages)._execute_mode_turn("诊断异常")
    assert result.task_status == "blocked"
    assert "不能判断" in result.reply


def test_reviewer_receives_only_current_turn_evidence(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    messages = trace() + trace("query_logistics", {"success": True, "logistics": {"status": "delivered"}})
    orch = fake_orchestrator(messages)
    orch.raw_messages = trace(order="ORD-PRIVATE-OLD")
    orch.quality_reviewer, calls = reviewer_with([APPROVE])
    result = orch._execute_mode_turn("签收没收到")
    assert result.quality_review.status == "approved"
    assert "ORD-PRIVATE-OLD" not in json.dumps(calls)


def test_rejected_draft_is_not_reused_as_future_expert_history(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    monkeypatch.setattr(settings, "multi_agent_review_max_repairs", 0)
    messages = trace() + trace("query_logistics", {"success": True, "logistics": {"status": "delivered"}})
    messages.append({"role": "assistant", "content": "明天肯定到达"})
    orch = fake_orchestrator(messages)
    orch.quality_reviewer, _ = reviewer_with([REVISE])
    result = orch._execute_mode_turn("签收没收到")
    assert result.task_status == "blocked"
    assert "明天肯定到达" not in json.dumps(orch.raw_messages, ensure_ascii=False)


def test_facts_fallback_scrubs_unreviewed_draft(monkeypatch):
    from tests.test_agent_quality import fulfillment_case
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    _, messages = fulfillment_case()
    messages.append({"role": "assistant", "content": "明天肯定到达"})
    orch = fake_orchestrator(messages)
    orch.quality_reviewer, _ = reviewer_with([TimeoutError()])
    result = orch._execute_mode_turn("订单 ORD-001 物流一直不动")
    assert result.quality_review.facts_only
    assert "明天肯定到达" not in json.dumps(orch.raw_messages, ensure_ascii=False)


@pytest.mark.parametrize("policy", [
    {"success": False, "error_code": "UNAVAILABLE", "results": []},
    {"success": True, "results": []},
    {"success": False, "error_code": "NO_MATCH", "results": []},
])
@pytest.mark.parametrize("complete", [True, False])
def test_policy_outage_preserves_only_verified_fulfillment_records(monkeypatch, policy, complete):
    from tests.test_agent_quality import fulfillment_case
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    _, messages = fulfillment_case()
    if not complete:
        messages = messages[:2]
    messages += trace("search_knowledge", policy)
    messages.append({"role": "assistant", "content": "明天肯定到达，赔偿一万元"})
    orch = fake_orchestrator(messages)
    orch.quality_reviewer, calls = reviewer_with([])
    result = orch._execute_mode_turn("订单 ORD-001 的物流一直不动")
    if complete:
        assert result.task_status == "partial"
        assert "运输中" in result.reply and "政策检索" in result.reply
        assert result.quality_review.status == "skipped"
        assert result.quality_review.facts_only
        assert "赔偿一万元" not in json.dumps(orch.raw_messages, ensure_ascii=False)
    else:
        assert "运输中" not in result.reply
        assert not result.quality_review.facts_only
    assert "明天肯定" not in result.reply
    assert not calls


def test_failed_batch_emits_skipped_results_without_executing_siblings():
    tools = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments="{}"))
             for i, name in enumerate(["query_order", "apply_refund"])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tools))],
    ))))
    executed = []
    def execute(name, args):
        executed.append(name)
        return '{"success":false,"error_code":"UNAVAILABLE"}'
    manager = SimpleNamespace(tool_definitions=[], execute_tool=execute)
    budget = SimpleNamespace(reserved_output_tokens=100, ensure_within_budget=lambda *_: None)
    agent = SubAgent("test", "test", manager, client, "test", 0, budget)
    _, messages = agent.handle([], max_steps=1)
    assert executed == ["query_order"]
    results = [item for item in messages if item["role"] == "tool"]
    assert len(results) == 2
    assert json.loads(results[1]["content"])["error_code"] == "DEPENDENCY_BLOCKED"


def test_real_orchestrator_initializes_five_roles_and_dispatches_authenticated_tools(tmp_path, monkeypatch):
    """Exercise real initialization, routing, ToolManager and review; only IO/model are faked."""
    monkeypatch.setattr(settings, "mcp_enabled", False)
    monkeypatch.setattr(settings, "skills_enabled", False)
    monkeypatch.setattr(settings, "memory_enabled", False)
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    monkeypatch.setattr(settings, "memory_consent_db_path", str(tmp_path / "consent.db"))
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    executed = []
    class Store:
        def query_order(self, user_id, order_id):
            executed.append(("order", user_id, order_id))
            return {"success": True, "order": {"order_id": order_id, "status": "delivered"}}
        def query_logistics(self, user_id, order_id):
            executed.append(("logistics", user_id, order_id))
            return {"success": True, "logistics": {"status": "delivered"}}
    store = Store()
    monkeypatch.setattr("app.agent.tools.order.get_order_store", lambda: store)
    monkeypatch.setattr("app.agent.tools.logistics.get_order_store", lambda: store)
    tool_calls = [SimpleNamespace(id=name, function=SimpleNamespace(
        name=name, arguments='{"order_id":"ORD-001"}',
    )) for name in ("query_order", "query_logistics")]
    replies = iter([
        SimpleNamespace(content=None, tool_calls=tool_calls),
        SimpleNamespace(content="物流记录已签收，但您反馈未收到，需要核实签收凭证。", tool_calls=[]),
        SimpleNamespace(content=json.dumps(APPROVE), tool_calls=[]),
    ])
    model_calls = []
    def create(**kwargs):
        model_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=next(replies))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("app.multi_agent.orchestrator.create_openai_client", lambda: client)
    orch = MultiAgentOrchestrator(session_path=str(tmp_path / "session.json"), user_id="test-owner")
    try:
        assert set(orch.agents) == {"presale", "postsale", "complaint", "fulfillment"}
        assert orch.quality_reviewer.name == "小飞-回复质检"
        assert not hasattr(orch.quality_reviewer, "tool_manager")
        query = "订单 ORD-001 显示签收但没收到"
        orch.raw_messages.append({"role": "user", "content": query})
        result = orch._execute_mode_turn(query)
        assert result.intent == IntentType.DELIVERY_EXCEPTION
        assert result.quality_review.status == "approved"
        assert executed == [("order", "test-owner", "ORD-001"), ("logistics", "test-owner", "ORD-001")]
        assert len(model_calls) == 3
        assert "tools" not in model_calls[-1]
    finally:
        orch.close()
