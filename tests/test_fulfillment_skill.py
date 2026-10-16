import json

import pytest

from app.agent.skills import SkillManager
from app.agent.skills.fulfillment_audit import audit_fulfillment, SKILL_NAME
from tests.test_agent_quality import trace, reviewer_with, APPROVE
from tests.test_fulfillment_agent import fake_orchestrator


def records(order_status="shipped", logistics_status="in_transit", source="demo_seed", tracking="SF1"):
    return trace(data={"success": True, "source": source, "order": {
        "order_id": "ORD-001", "status": order_status, "tracking_number": "SF1",
        "address": "PRIVATE", "phone": "SECRET"}}) + trace("query_logistics", {
        "success": True, "source": source, "logistics": {
            "status": logistics_status, "tracking_number": tracking}})


@pytest.mark.parametrize("order,logistics,code", [
    ("shipped", "in_transit", "NO_KNOWN_DIVERGENCE"),
    ("refund_processing", "in_transit", "REFUND_IN_TRANSIT"),
    ("delivered", "in_transit", "STATUS_DIVERGENCE"),
    ("shipped", "delivered", "STATUS_DIVERGENCE"),
    ("pending", "in_transit", "STATUS_DIVERGENCE"),
    ("delivered", "delivered", "NO_KNOWN_DIVERGENCE"),
    ("invented", "in_transit", "UNKNOWN_STATUS"),
    ("shipped", {}, "UNKNOWN_STATUS"),
])
def test_rule_matrix(order, logistics, code):
    report = audit_fulfillment(records(order, logistics), "test-version")
    assert code in {check.code for check in report.checks}
    assert report.version == "test-version" and report.rule_version == "1"
    assert report.checks[0].evidence_ids == ["e1", "e2"]
    encoded = report.model_dump_json()
    assert "PRIVATE" not in encoded and "SECRET" not in encoded
    if code != "UNKNOWN_STATUS":
        assert "DEMO_RECORDS" in encoded and "LIVE_STATUS_UNCONFIRMED" in encoded


@pytest.mark.parametrize("case", ["missing", "mismatch", "duplicate", "failed", "orphan"])
def test_untrusted_or_incomplete_evidence_never_passes(case):
    messages = records(tracking="OTHER" if case == "mismatch" else "SF1")
    if case == "missing":
        messages = messages[:2]
    elif case == "duplicate":
        messages += trace()
    elif case == "failed":
        messages[-1]["content"] = '{"success":false,"error_code":"UNAVAILABLE"}'
    elif case == "orphan":
        messages = messages[1:]
    assert audit_fulfillment(messages, "v").status == "incomplete"


def test_orchestrator_activates_skill_and_hands_audit_to_reviewer():
    orch = fake_orchestrator(records("refund_processing"))
    orch.skill_manager = SkillManager()
    captured = []
    orch.agents["fulfillment"].handle = lambda messages, **kwargs: (captured.extend(messages) or "仅核实记录", records("refund_processing"))
    orch.quality_reviewer, calls = reviewer_with([APPROVE])
    result = orch._execute_mode_turn("订单 ORD-001 的物流一直不动")
    assert result.skill_audit.skill == SKILL_NAME
    assert len(result.skill_audit.version) == 64
    assert "履约证据核对 SOP" in captured[0]["content"]
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["skill_audit"]["checks"][1]["code"] == "REFUND_IN_TRANSIT"
    assert len(calls) == 1


def test_skill_disabled_keeps_existing_path():
    orch = fake_orchestrator(records())
    orch.skill_manager = SkillManager(enabled=False)
    assert orch._execute_mode_turn("物流不动").skill_audit is None


def test_wrong_tracking_blocks_draft_before_review():
    orch = fake_orchestrator(records(tracking="OTHER"))
    orch.skill_manager = SkillManager()
    orch.quality_reviewer, calls = reviewer_with([])
    result = orch._execute_mode_turn("物流不动")
    assert result.task_status == "blocked"
    assert result.skill_audit.status == "incomplete"
    assert not calls
