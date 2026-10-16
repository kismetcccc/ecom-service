"""MA requirements: routing safety, capability plans and read-only execution."""

import json
import threading
from types import SimpleNamespace

import pytest

from app.multi_agent.agents import READ_ONLY_TOOLS, SubAgent
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.planner import PlanningError, build_plan
from app.multi_agent.router import RouteDecision, Router
from app.config.settings import settings
from app.schemas.response import IntentType


def _router_with_reply(reply):
    class Completions:
        def create(self, **_kwargs):
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    return Router(SimpleNamespace(chat=SimpleNamespace(completions=Completions())), "test-model")


def test_rule_fast_path_and_unsafe_parse_fallback():
    router = _router_with_reply(RuntimeError("model unavailable"))
    assert router.decide("你好").source.startswith("rule:")
    assert router.decide("查订单 ORD-20240115-001").agent == "postsale"
    assert router.decide("取消订单 ORD-20240115-001").agent is None
    # 明确退款现在进入确定性只读核验，不等于执行退款写操作。
    assert router.decide("我要退款").agent == "postsale"
    assert router.decide("我要转人工").agent is None
    assert router.decide("银行卡被盗刷了").risk_level == "high"


@pytest.mark.parametrize("query", [
    "我想了解退换货政策", "我要咨询退货政策", "帮我查看退款规则",
    "我想了解退换货政策，不申请退货", "只想看政策",
    "七天无理由退货的政策是什么？我只咨询政策，不申请退货。",
])
def test_policy_questions_never_start_return_workflow(query):
    decision = _router_with_reply(RuntimeError("must not call model")).decide(query)
    assert decision.agent == "presale"
    assert decision.intents == ("policy",)
    assert decision.source.startswith("rule:")


@pytest.mark.parametrize("query", [
    "我想退货，顺便了解退货政策", "帮我申请退款并查退款政策",
    "了解退货政策后，帮我办理退货", "我要退款",
])
def test_real_return_actions_are_not_erased_by_policy_words(query):
    from app.multi_agent.router import infer_intents
    assert "return" in infer_intents(query)


def test_greeting_finishes_without_expert_or_tool_call():
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.router = SimpleNamespace(decide=lambda *_args: RouteDecision(
        "postsale", ("greeting",), 1.0, "rule:v1",
    ))
    orchestrator.raw_messages = [{"role": "user", "content": "你好"}]
    result = orchestrator._execute_mode_turn("你好")
    assert "小飞" in result.reply
    assert result.task_status == "single"


def test_invalid_or_low_confidence_output_never_silently_selects_writable_agent(monkeypatch):
    # 明确退货现有核验快路；此测试专门验证未命中快路时的模型输出校验。
    monkeypatch.setattr(settings, "multi_agent_rules_enabled", False)
    assert _router_with_reply("postsale").decide("我要退款").agent is None
    low = json.dumps({
        "agent": "postsale", "intents": ["return"], "confidence": 0.3,
        "escalation": False, "risk_level": "medium",
    })
    decision = _router_with_reply(low).decide("我要退款")
    assert decision.agent is None
    assert decision.fallback_reason == "LOW_CONFIDENCE"
    invented = json.dumps({
        "agent": "postsale", "intents": ["return"], "confidence": 0.99,
        "escalation": False, "risk_level": "low",
    })
    assert _router_with_reply(invented).decide("这款鞋怎么样").agent is None


def test_mixed_complaint_and_refund_keeps_business_capability_and_escalation():
    raw = json.dumps({
        "agent": "complaint", "intents": ["complaint", "return"],
        "confidence": 0.96, "escalation": True, "risk_level": "medium",
    })
    decision = _router_with_reply(raw).decide("我要投诉并退款")
    assert decision.agent == "postsale"
    assert decision.escalation
    plan = build_plan(decision)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].agent == "postsale"
    assert not plan.tasks[0].read_only
    complain_and_buy = build_plan(RouteDecision(
        "presale", ("complaint", "product"), 0.9, "model", escalation=True,
    ))
    assert [task.agent for task in complain_and_buy.tasks] == ["complaint", "presale"]
    assert not complain_and_buy.parallel


def test_only_distinct_independent_read_tasks_can_parallelize():
    read_plan = build_plan(RouteDecision("postsale", ("product", "order"), 0.9, "model"))
    assert {task.agent for task in read_plan.tasks} == {"presale", "postsale"}
    assert read_plan.parallel
    write_plan = build_plan(RouteDecision("postsale", ("product", "return"), 0.9, "model"))
    assert not write_plan.parallel
    with pytest.raises(PlanningError):
        build_plan(
            RouteDecision("postsale", ("order",), 0.9, "model"),
            available_tools={"postsale": set()},
        )


def test_missing_runtime_tool_returns_unavailable_without_running_agent():
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.router = SimpleNamespace(decide=lambda *_args: RouteDecision("postsale", ("order",), 0.9, "model"))
    orchestrator.agents = {"postsale": SimpleNamespace(tool_manager=SimpleNamespace(tool_definitions=[]))}
    orchestrator.raw_messages = [{"role": "user", "content": "查订单"}]
    result = orchestrator._execute_mode_turn("查订单")
    assert result.task_status == "blocked"
    assert "工具暂不可用" in result.reply


def test_single_read_only_expert_failure_is_blocked_not_http_500():
    class FailingAgent:
        tool_manager = SimpleNamespace(tool_definitions=[
            {"function": {"name": "query_product"}},
        ])
        name = "商品专家"

        def handle(self, *_args, **_kwargs):
            raise TimeoutError("upstream timeout")

    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.router = SimpleNamespace(decide=lambda *_args: RouteDecision(
        "presale", ("product",), 0.9, "model",
    ))
    orchestrator.agents = {"presale": FailingAgent()}
    orchestrator.raw_messages = [{"role": "user", "content": "推荐商品"}]
    orchestrator.max_react_steps = 1
    orchestrator._prepare_task_messages = lambda _task: [{"role": "system", "content": "x"}]
    result = orchestrator._execute_mode_turn("推荐商品")
    assert result.task_status == "blocked"
    assert result.requires_human is True
    assert "已核实" not in result.reply


def test_read_only_task_rejects_hidden_write_tool_call():
    calls = []

    class Completions:
        def create(self, **_kwargs):
            calls.append("model")
            if len(calls) == 1:
                function = SimpleNamespace(name="apply_refund", arguments='{"preview_token":"x","idempotency_key":"y"}')
                tool_call = SimpleNamespace(id="call-1", function=function)
                message = SimpleNamespace(content=None, tool_calls=[tool_call])
            else:
                message = SimpleNamespace(content="无法执行该操作。", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class ToolManager:
        tool_definitions = [
            {"function": {"name": "apply_refund"}},
            {"function": {"name": "query_order"}},
        ]

        def execute_tool(self, name, _args):
            raise AssertionError(f"forbidden tool executed: {name}")

    budget = SimpleNamespace(reserved_output_tokens=100, ensure_within_budget=lambda *_args: None)
    agent = SubAgent("test", "prompt", ToolManager(),
                     SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
                     "test-model", 0.0, budget)
    _, messages = agent.handle([{"role": "system", "content": "x"}], max_steps=2,
                               allowed_tools=READ_ONLY_TOOLS)
    tool_result = json.loads(next(item["content"] for item in messages if item["role"] == "tool"))
    assert tool_result["error_code"] == "UNAUTHORIZED"


def test_failed_tool_stops_react_loop_without_another_model_call():
    calls = []

    class Completions:
        def create(self, **_kwargs):
            calls.append("model")
            function = SimpleNamespace(
                name="search_knowledge",
                arguments='{"query":"退货政策","top_k":3}',
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(id="call-1", function=function)],
            ))])

    class ToolManagerStub:
        tool_definitions = [{"function": {
            "name": "search_knowledge",
            "parameters": {"type": "object", "properties": {}},
        }}]

        @staticmethod
        def execute_tool(_name, _args):
            return json.dumps({
                "success": False, "error_code": "UNAVAILABLE", "error": "Qdrant 未启动",
            }, ensure_ascii=False)

    budget = SimpleNamespace(
        reserved_output_tokens=100,
        ensure_within_budget=lambda *_args: None,
    )
    agent = SubAgent(
        "test", "prompt", ToolManagerStub(),
        SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        "test-model", 0.0, budget,
    )
    text, messages = agent.handle(
        [{"role": "system", "content": "x"}], max_steps=5,
        allowed_tools=READ_ONLY_TOOLS,
    )
    assert len(calls) == 1
    assert "无法核实" in text
    assert any(item["role"] == "tool" for item in messages)


def test_orchestrator_merges_parallel_results_in_plan_order():
    barrier = threading.Barrier(2)

    class FakeAgent:
        def __init__(self, reply, tools):
            self.reply = reply
            self.tool_manager = SimpleNamespace(tool_definitions=[{"function": {"name": name}} for name in tools])

        def handle(self, _messages, max_steps, allowed_tools):
            assert max_steps == 1
            assert allowed_tools == READ_ONLY_TOOLS
            barrier.wait(timeout=2)
            return self.reply, [{"role": "assistant", "content": self.reply}]

    orchestrator = object.__new__(MultiAgentOrchestrator)
    decision = RouteDecision("postsale", ("product", "order"), 0.9, "model")
    orchestrator.router = SimpleNamespace(decide=lambda *_args: decision)
    orchestrator.agents = {
        "presale": FakeAgent("商品信息已回复", {"query_product"}),
        "postsale": FakeAgent("订单信息已回复", {"query_order", "list_user_orders"}),
    }
    orchestrator.raw_messages = [{"role": "user", "content": "查订单并推荐商品"}]
    orchestrator.max_react_steps = 1
    orchestrator._prepare_task_messages = lambda task: [{"role": "system", "content": task.label}]

    result = orchestrator._execute_mode_turn("查订单并推荐商品")
    assert result.task_status == "answered"
    assert [item.label for item in result.task_outcomes] == ["订单、物流或售后处理", "商品咨询"]
    assert result.reply.index("订单信息") < result.reply.index("商品信息")


def test_parallel_policy_failure_keeps_other_answer_but_not_unverified_claim():
    class FakeAgent:
        def __init__(self, kind, tools):
            self.kind = kind
            self.tool_manager = SimpleNamespace(tool_definitions=[{"function": {"name": name}} for name in tools])

        def handle(self, _messages, **_kwargs):
            if self.kind == "policy":
                call = {"id": "search-1", "function": {"name": "search_knowledge", "arguments": "{}"}}
                return "平台保证七天退款", [
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "search-1", "content": json.dumps({
                        "success": False, "error_code": "UNAVAILABLE", "results": [],
                    })},
                ]
            return "商品信息已核实", [{"role": "assistant", "content": "商品信息已核实"}]

    orchestrator = object.__new__(MultiAgentOrchestrator)
    decision = RouteDecision("postsale", ("product", "order", "policy"), 0.9, "model")
    orchestrator.router = SimpleNamespace(decide=lambda *_args: decision)
    orchestrator.agents = {
        "postsale": FakeAgent("policy", {"query_order", "list_user_orders", "search_knowledge"}),
        "presale": FakeAgent("product", {"query_product"}),
    }
    orchestrator.raw_messages = [{"role": "user", "content": "查订单并推荐商品，说明政策"}]
    orchestrator.max_react_steps = 1
    orchestrator._prepare_task_messages = lambda task: [{"role": "system", "content": task.label}]

    result = orchestrator._execute_mode_turn("查订单并推荐商品，说明政策")
    assert result.task_status == "partial"
    assert result.knowledge_status == "UNAVAILABLE"
    assert "商品信息已核实" in result.reply
    assert "平台保证七天退款" not in result.reply


def test_write_capable_service_step_stays_serial_and_precedes_product_reply():
    calls = []

    class FakeAgent:
        def __init__(self, key, tools):
            self.key = key
            self.tool_manager = SimpleNamespace(tool_definitions=[{"function": {"name": name}} for name in tools])

        def handle(self, _messages, **kwargs):
            calls.append((self.key, kwargs["allowed_tools"]))
            return f"{self.key}已回复", [{"role": "assistant", "content": f"{self.key}已回复"}]

    orchestrator = object.__new__(MultiAgentOrchestrator)
    decision = RouteDecision("postsale", ("return", "product"), 0.9, "model")
    orchestrator.router = SimpleNamespace(decide=lambda *_args: decision)
    orchestrator.agents = {
        "postsale": FakeAgent("service", {"query_order", "list_user_orders", "search_knowledge", "apply_refund"}),
        "presale": FakeAgent("product", {"query_product"}),
    }
    orchestrator.raw_messages = [{"role": "user", "content": "退款并推荐商品"}]
    orchestrator.max_react_steps = 1
    orchestrator._prepare_task_messages = lambda task: [{"role": "system", "content": task.label}]

    result = orchestrator._execute_mode_turn("退款并推荐商品")
    assert [key for key, _ in calls] == ["service", "product"]
    assert calls[0][1] is None
    assert calls[1][1] == READ_ONLY_TOOLS
    assert result.task_status == "answered"


def test_expert_context_omits_raw_tool_traces_but_keeps_recent_dialogue():
    history = [
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "敏感的原始订单数据"},
        {"role": "assistant", "content": "订单状态已回复"},
        {"role": "user", "content": "再推荐商品"},
    ]
    recent = MultiAgentOrchestrator._minimal_history(history, 2)
    assert recent == history[-2:]
    assert all(item["role"] != "tool" and not item.get("tool_calls") for item in recent)


def test_structured_response_reuses_validated_route_without_model_call():
    decision = RouteDecision("postsale", ("order",), 0.92, "model")
    result = MultiAgentOrchestrator._response_from_decision(
        decision, "已核实订单状态",
    )
    assert result.reply == "已核实订单状态"
    assert result.intent == IntentType.ORDER_QUERY
    assert result.confidence == 0.92
    assert result.requires_human is False


def test_list_orders_uses_rule_router_without_model_call():
    router = _router_with_reply(AssertionError("router model must not run"))
    decision = router.decide("查看所有订单")
    assert decision.agent == "postsale"
    assert decision.source.startswith("rule:")


def test_return_policy_question_is_read_only_rule_route():
    router = _router_with_reply(AssertionError("router model must not run"))
    decision = router.decide("查退货政策")
    assert decision.agent == "presale"
    assert decision.intents == ("policy",)
    assert decision.source.startswith("rule:")


def test_explicit_return_action_is_not_downgraded_to_policy_only():
    decision = Router._fast_rule(
        "我想申请退货，请说明退货政策",
        ("return", "policy"),
    )
    assert decision is None
