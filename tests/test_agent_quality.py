import json
from types import SimpleNamespace

import pytest

from app.agent.context_budget import ContextBudgetManager
from app.config.settings import settings
from app.multi_agent.quality import QualityReviewAgent, ReviewDecision, collect_evidence
from app.schemas.response import CustomerServiceResponse, IntentType, TaskOutcome


def trace(name="query_order", data=None, order="ORD-001"):
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call-{name}", "function": {"name": name, "arguments": json.dumps({"order_id": order})},
        }]},
        {"role": "tool", "tool_call_id": f"call-{name}", "content": json.dumps(data or {
            "success": True, "order": {"order_id": order, "status": "shipped", "address": "PRIVATE", "phone": "SECRET"},
        })},
    ]


def response(reply="包裹明天一定到达"):
    return CustomerServiceResponse(
        intent=IntentType.ORDER_QUERY, confidence=.9, reply=reply,
        task_outcomes=[TaskOutcome(label="订单查询", status="answered")],
    )


APPROVE = {"verdict": "approve", "issues": []}
REVISE = {"verdict": "revise", "issues": [{"code": "UNSAFE_COMMITMENT", "instruction": "删除无依据的确定送达承诺", "evidence_ids": ["e1"]}]}


def reviewer_with(replies):
    calls = []
    outputs = iter(replies)

    def create(**kwargs):
        calls.append(kwargs)
        value = next(outputs)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])

    agent = QualityReviewAgent(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "review-test", ContextBudgetManager(28000, 4000, 6),
    )
    return agent, calls


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_mode", "enforce")
    monkeypatch.setattr(settings, "multi_agent_review_max_repairs", 1)


def test_handoff_is_tool_bound_and_removes_private_fields():
    evidence = collect_evidence(trace())
    assert evidence[0].evidence_id == "e1"
    encoded = evidence[0].model_dump_json()
    assert "PRIVATE" not in encoded and "SECRET" not in encoded
    assert evidence[0].arguments["order_id"] == "ORD-001"
    with pytest.raises(ValueError):
        collect_evidence(trace()[1:])
    with pytest.raises(ValueError):
        collect_evidence(trace() + trace())
    with pytest.raises(ValueError):
        collect_evidence(trace()[:1])


def test_approved_answer_keeps_business_metadata():
    agent, calls = reviewer_with([APPROVE])
    result = response("订单已发货")
    agent.evaluate("订单怎么样", result, trace())
    assert result.reply == "订单已发货"
    assert result.quality_review.status == "approved"
    assert result.quality_review.evidence_count == 1
    assert result.task_status == "single"
    assert len(calls) == 1
    assert all("tools" not in call for call in calls)
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("content,expected", [
    ('```json\n{"verdict":"approve","issues":[]}\n```', "approved"),
    ('```json\n{"verdict":"approve","issues":[{"code":"OTHER","instruction":"问题"}]}\n```', "unavailable"),
    ('说明：{"verdict":"approve","issues":[]}', "unavailable"),
    ('{"verdict":"approve","issues":[],"reason":"extra"}', "unavailable"),
])
def test_review_format_wrapper_keeps_strict_semantics(content, expected):
    agent, _ = reviewer_with([])
    agent.client.chat.completions.create = lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    result = agent.evaluate("查订单", response(), trace())
    assert result.quality_review.status == expected


def test_repair_is_rechecked_and_never_reexecutes_tools():
    agent, calls = reviewer_with([REVISE, {"reply": "订单记录显示已发货，暂不能保证具体到达时间。"}, APPROVE])
    result = agent.evaluate("什么时候到", response(), trace())
    assert result.quality_review.status == "repaired"
    assert result.quality_review.repair_count == 1
    assert result.quality_review.model_calls == 3
    assert "暂不能保证" in result.reply
    assert all("tools" not in call for call in calls)


def test_second_rejection_terminates_instead_of_looping():
    agent, calls = reviewer_with([REVISE, {"reply": "明天保证到"}, REVISE])
    result = agent.evaluate("什么时候到", response(), trace())
    assert len(calls) == 3
    assert result.task_status == "blocked"
    assert result.requires_human
    assert "明天" not in result.reply
    assert result.task_outcomes[0].status == "blocked"


@pytest.mark.parametrize("failure", [TimeoutError(), {"verdict": "approve", "issues": [REVISE["issues"][0]]},
    {"verdict": "revise", "issues": [{**REVISE["issues"][0], "evidence_ids": ["invented"]}]}])
def test_failed_or_fabricated_review_never_approves(failure):
    agent, calls = reviewer_with([failure])
    result = agent.evaluate("什么时候到", response(), trace())
    assert result.quality_review.status == "unavailable"
    assert result.task_status == "blocked"
    assert len(calls) == 1


def test_shadow_observes_without_repair_or_mutation(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_mode", "shadow")
    agent, calls = reviewer_with([REVISE])
    original = response()
    result = agent.evaluate("什么时候到", original, trace())
    assert result.reply == "包裹明天一定到达"
    assert result.task_status == "single"
    assert result.quality_review.status == "blocked"
    assert len(calls) == 1


def test_off_and_deterministic_workflow_do_not_call_model(monkeypatch):
    agent, calls = reviewer_with([])
    result = response()
    result.workflow_id = "deterministic-flow"
    assert agent.evaluate("退货", result, []).quality_review.status == "skipped"
    monkeypatch.setattr(settings, "multi_agent_review_mode", "off")
    assert agent.evaluate("订单", response(), trace()).quality_review.status == "disabled"
    assert not calls


def test_zero_repair_budget_rejects_immediately(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_max_repairs", 0)
    agent, calls = reviewer_with([REVISE])
    assert agent.evaluate("什么时候到", response(), trace()).task_status == "blocked"
    assert len(calls) == 1


def test_oversize_evidence_fails_before_model_request():
    agent, calls = reviewer_with([])
    messages = trace(data={"success": True, "order": {"description": "a" * 19000}})
    assert agent.evaluate("订单", response(), messages).quality_review.status == "unavailable"
    assert not calls


def test_review_contract_rejects_unknown_fields():
    with pytest.raises(ValueError):
        ReviewDecision.model_validate({**APPROVE, "execute_refund": True})


def test_deadline_expiration_never_publishes_late_approval(monkeypatch):
    import app.multi_agent.quality as module
    ticks = iter([0, 0, 100, 100])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    agent, calls = reviewer_with([APPROVE])
    result = agent.evaluate("订单", response(), trace())
    assert result.quality_review.status == "unavailable"
    assert result.task_status == "blocked"
    assert len(calls) == 1


def test_shadow_timeout_preserves_original_answer(monkeypatch):
    monkeypatch.setattr(settings, "multi_agent_review_mode", "shadow")
    agent, _ = reviewer_with([TimeoutError()])
    result = agent.evaluate("订单", response(), trace())
    assert result.reply == "包裹明天一定到达"
    assert result.quality_review.status == "unavailable"


def test_benchmark_counts_unavailability_separately_from_safe_rejection():
    from app.evaluation.review_quality import summarize
    report = summarize([
        {"expected": "approve", "status": "blocked", "model_calls": 1},
        {"expected": "reject", "status": "approved", "model_calls": 1},
        {"expected": "reject", "status": "unavailable", "model_calls": 1},
    ])
    assert report["false_accepts"] == report["false_rejects"] == report["unavailable"] == 1
    assert report["matched"] == 0 and not report["passed"]


def test_labeled_benchmark_runs_one_review_per_case_and_restores_config():
    from pathlib import Path
    from app.evaluation.review_quality import run_cases
    cases = json.loads(Path("app/evaluation/review_cases.json").read_text(encoding="utf-8"))
    decisions = [APPROVE if case["expected"] == "approve" else REVISE for case in cases]
    agent, calls = reviewer_with(decisions)
    report = run_cases(agent, cases)
    assert report["passed"]
    assert len(calls) == len(cases)
    assert settings.multi_agent_review_max_repairs == 1


def fulfillment_case():
    result = response()
    result.intent = IntentType.DELIVERY_EXCEPTION
    result.task_outcomes[0].agent = "fulfillment"
    messages = trace(data={"success": True, "source": "demo_seed", "order": {
        "order_id": "ORD-001", "status": "shipped", "tracking_number": "SF1"}})
    messages += trace("query_logistics", {"success": True, "logistics": {
        "status": "in_transit", "tracking_number": "SF1"}})
    return result, messages


@pytest.mark.parametrize("host,model,enabled,expected", [
    ("dashscope.aliyuncs.com", "deepseek-v4.1-flash", True, True),
    ("dashscope-intl.aliyuncs.com", "deepseek-v3.2", True, True),
    ("dashscope.aliyuncs.com", "deepseek-v4.1-flash", False, False),
    ("api.openai.com", "deepseek-v4.1-flash", True, False),
    ("dashscope.aliyuncs.com.attacker.test", "deepseek-v4.1-flash", True, False),
    ("dashscope.aliyuncs.com", "deepseek-r1", True, False),
    ("dashscope.aliyuncs.com", "other-model", True, False),
])
def test_fast_review_is_provider_and_model_scoped(monkeypatch, host, model, enabled, expected):
    monkeypatch.setattr(settings, "openai_base_url", f"https://{host}/compatible-mode/v1")
    monkeypatch.setattr(settings, "multi_agent_review_fast_mode", enabled)
    agent, calls = reviewer_with([APPROVE])
    agent.model = model
    assert agent.evaluate("查询订单", response(), trace()).quality_review.status == "approved"
    assert calls[0].get("extra_body") == ({"enable_thinking": False} if expected else None)


def test_fulfillment_timeout_shows_only_bound_facts():
    result, messages = fulfillment_case()
    agent, calls = reviewer_with([TimeoutError()])
    agent.evaluate("物流一直不动", result, messages)
    assert result.quality_review.reason_code == "REVIEW_TIMEOUT"
    assert result.quality_review.status == "unavailable"
    assert result.quality_review.facts_only
    assert result.task_status == "partial"
    assert result.task_outcomes[0].status == "verified"
    assert "运输中" in result.reply and "演示数据" in result.reply
    assert "明天" not in result.reply and result.requires_human
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["missing", "mismatch", "mixed", "write", "unknown_write", "malformed", "rejected", "invalid_review"])
def test_fulfillment_fallback_does_not_bypass_evidence_or_review(change, monkeypatch):
    result, messages = fulfillment_case()
    failure = TimeoutError()
    if change == "missing":
        messages = messages[:2]
    elif change == "mismatch":
        messages[-1]["content"] = json.dumps({"success": True, "logistics": {"status": "in_transit", "tracking_number": "OTHER"}})
    elif change == "mixed":
        result.task_outcomes.append(TaskOutcome(label="退款", status="answered", agent="aftersales"))
    elif change in {"write", "unknown_write"}:
        messages += trace("apply_refund" if change == "write" else "custom_write", {"success": True})
    elif change == "malformed":
        messages[-1]["content"] = "invalid json"
    elif change == "rejected":
        monkeypatch.setattr(settings, "multi_agent_review_max_repairs", 0)
        failure = REVISE
    else:
        failure = {"verdict": "bad"}
    agent, _ = reviewer_with([failure])
    agent.evaluate("物流一直不动", result, messages)
    assert not result.quality_review.facts_only
    assert result.task_status == "blocked"
    assert "运输中" not in result.reply
