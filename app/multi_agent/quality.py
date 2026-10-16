"""Bounded evaluator/optimizer with current-turn, tool-bound evidence handoff.

The reviewer has no ToolManager and the repair call has no tools. A critique
cannot re-run a write or alter business metadata. No shared mutable turn state.
"""

import json
import logging
import re
import time
from urllib.parse import urlsplit
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, field_validator
from openai import APITimeoutError, APIConnectionError, APIStatusError

from app.config.settings import settings
from app.prompts.agent_quality import REVIEW_PROMPT, REPAIR_PROMPT
from app.schemas.response import IntentType, QualityReview
from app.server.middleware import request_id_var


_logger = logging.getLogger(__name__)
_FIELDS = {
    "query_order": {"order"}, "list_user_orders": {"orders", "count"},
    "query_logistics": {"logistics"},
    "query_product": {"products", "count"}, "search_products": {"products", "count"},
    "search_knowledge": {"results", "index_version", "wiki_pages", "provider", "wiki_fallback_reason"},
    "get_return_eligibility": {"order_id", "eligible", "reason_code", "policy_note"},
    "preview_refund": {"order_id", "amount", "reason", "expires_at", "message"},
    "apply_refund": {"application_id", "order_id", "status", "message"},
    "get_refund_status": {"application_id", "order_id", "status", "message", "created_at"},
}
_PRIVATE = ("token", "secret", "password", "phone", "address", "email", "recipient", "user_id", "owner_id", "idempotency")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    tool: str
    arguments: dict
    data: dict


class ReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal["UNSUPPORTED_CLAIM", "MISSED_INTENT", "UNSAFE_COMMITMENT", "CONFLICT", "PRIVACY", "OTHER"]
    instruction: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["approve", "revise", "escalate"]
    issues: list[ReviewIssue] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent(self):
        if (self.verdict == "approve") != (not self.issues):
            raise ValueError("verdict and issues disagree")
        return self


class RepairedReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reply: str = Field(min_length=1, max_length=6000)

    @field_validator("reply")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("empty repaired reply")
        return value.strip()


def _clean(value, depth=0):
    if depth > 12:
        raise ValueError("evidence nesting limit")
    if isinstance(value, dict):
        return {key: _clean(item, depth + 1) for key, item in value.items()
                if not any(part in key.lower() for part in _PRIVATE) and key not in {"user", "收货人", "地址", "电话"}}
    if isinstance(value, list):
        return [_clean(item, depth + 1) for item in value]
    return value


def collect_evidence(messages: list[dict]) -> list[Evidence]:
    """Bind results to earlier calls in this handoff, never history or prose."""
    pending = {}
    seen = set()
    evidence = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not isinstance(call_id, str) or call_id in seen:
                    raise ValueError("ambiguous tool call id")
                seen.add(call_id)
                pending[call_id] = call.get("function", {})
        elif message.get("role") == "tool":
            function = pending.pop(message.get("tool_call_id"), None)
            if function is None:
                raise ValueError("orphan tool result")
            name = function.get("name")
            if name not in _FIELDS:
                continue
            raw = message.get("content", "")
            if not isinstance(raw, str) or len(raw) > 128000:
                raise ValueError("oversized tool result")
            data = json.loads(raw)
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(data, dict) or not isinstance(data.get("success"), bool) or not isinstance(arguments, dict):
                raise ValueError("invalid tool result")
            fields = _FIELDS[name] | {"success", "error_code", "source", "as_of"}
            evidence.append(Evidence(
                evidence_id=f"e{len(evidence) + 1}", tool=name,
                arguments={key: arguments[key] for key in ("order_id", "application_id") if key in arguments},
                data=_clean({key: value for key, value in data.items() if key in fields}),
            ))
    if pending:
        raise ValueError("incomplete tool exchange")
    if len(evidence) > 20 or len(json.dumps([item.model_dump() for item in evidence], ensure_ascii=False)) > 18000:
        raise ValueError("evidence budget exceeded")
    return evidence


class QualityReviewAgent:
    """Independent LLM reviewer: strict contract, no tools, one repair maximum."""

    name = "小飞-回复质检"

    def __init__(self, client, model, context_budget):
        self.client = client
        self.model = model
        self.context_budget = context_budget

    def _call(self, prompt, payload, schema, deadline, stats):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("review time budget exhausted")
        messages = [
            {"role": "system", "content": prompt + "\nJSON schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        self.context_budget.ensure_within_budget(messages)
        options = {}
        host = urlsplit(settings.openai_base_url).hostname or ""
        supported_models = {"deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-0731", "deepseek-v3.2", "deepseek-v3.2-exp", "deepseek-v3.1"}
        if (settings.multi_agent_review_fast_mode and self.model in supported_models
                and host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}):
            options["extra_body"] = {"enable_thinking": False}
        stats.model_calls += 1
        result = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
            max_tokens=self.context_budget.reserved_output_tokens,
            response_format={"type": "json_object"},
            timeout=min(remaining, settings.multi_agent_review_timeout_seconds),
            **options,
        )
        if time.monotonic() > deadline:
            raise TimeoutError("review time budget exhausted")
        if getattr(result.choices[0], "finish_reason", None) == "length":
            raise ValueError("review output truncated")
        content = result.choices[0].message.content or ""
        # Accept only a complete JSON fence, never extract an arbitrary object
        # from prose or change a verdict/issue to make invalid review pass.
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", content, re.S)
        if fenced:
            content = fenced.group(1)
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            _logger.warning("agent_review_schema_error request_id=%s schema=%s errors=%s",
                            request_id_var.get(), schema.__name__,
                            [(error["loc"], error["type"]) for error in exc.errors(include_input=False)])
            raise

    def evaluate(self, request, result, messages):
        started = time.monotonic()
        mode = settings.multi_agent_review_mode
        stats = QualityReview(status="skipped", mode=mode)
        result.quality_review = stats
        if mode == "off":
            stats.status = "disabled"
            return result
        # Deterministic workflow/failed-query replies have no generated draft to repair.
        if (result.workflow_id or result.intent == IntentType.GREETING or result.task_status == "blocked"
                or (len(result.task_outcomes) == 1 and result.task_outcomes[0].reason_code == "FACTS_ONLY_POLICY_UNAVAILABLE")):
            stats.reason_code = "DETERMINISTIC_OR_BLOCKED"
            stats.facts_only = any(item.reason_code == "FACTS_ONLY_POLICY_UNAVAILABLE" for item in result.task_outcomes)
            return result
        deadline = started + settings.multi_agent_review_budget_seconds
        evidence = []
        try:
            evidence = collect_evidence(messages)
            stats.evidence_count = len(evidence)
            payload = {
                "user_request": request, "draft": result.reply,
                "follow_up_question": result.follow_up_question,
                "evidence": [item.model_dump() for item in evidence],
                "task_outcomes": [item.model_dump() for item in result.task_outcomes],
                "skill_audit": result.skill_audit.model_dump() if result.skill_audit else None,
            }
            valid_ids = {item.evidence_id for item in evidence}
            for attempt in range(settings.multi_agent_review_max_repairs + 1):
                decision = self._call(REVIEW_PROMPT, payload, ReviewDecision, deadline, stats)
                if any(ref not in valid_ids for issue in decision.issues for ref in issue.evidence_ids):
                    raise ValueError("reviewer invented evidence")
                if decision.verdict == "approve":
                    stats.status = "repaired" if attempt else "approved"
                    if mode == "enforce":
                        result.reply = payload["draft"]
                    return result
                if (decision.verdict == "escalate" or mode == "shadow"
                        or attempt == settings.multi_agent_review_max_repairs):
                    stats.status = "blocked"
                    stats.reason_code = "REVIEW_ESCALATION" if decision.verdict == "escalate" else "REVIEW_REJECTED"
                    break
                repair = self._call(
                    REPAIR_PROMPT, {**payload, "feedback": decision.model_dump()},
                    RepairedReply, deadline, stats,
                )
                stats.repair_count += 1
                payload["draft"] = repair.reply
        except Exception as exc:
            stats.status = "unavailable"
            if isinstance(exc, (TimeoutError, APITimeoutError)):
                stats.reason_code = "REVIEW_TIMEOUT"
            elif isinstance(exc, APIStatusError) and exc.status_code == 429:
                stats.reason_code = "REVIEW_RATE_LIMITED"
            elif isinstance(exc, (APIConnectionError, APIStatusError)):
                stats.reason_code = "REVIEW_UPSTREAM_ERROR"
            else:
                stats.reason_code = "REVIEW_INVALID_RESPONSE" if stats.model_calls else "REVIEW_INVALID_EVIDENCE"
            _logger.warning("agent_review_failed request_id=%s type=%s", request_id_var.get(), type(exc).__name__)
        finally:
            stats.elapsed_ms = int((time.monotonic() - started) * 1000)
            _logger.info(
                "agent_review request_id=%s status=%s mode=%s calls=%d repairs=%d evidence=%d elapsed_ms=%d",
                request_id_var.get(), stats.status, mode, stats.model_calls,
                stats.repair_count, stats.evidence_count, stats.elapsed_ms,
            )
        if mode == "enforce":
            if (stats.status == "unavailable"
                    and stats.reason_code in {"REVIEW_TIMEOUT", "REVIEW_RATE_LIMITED", "REVIEW_UPSTREAM_ERROR"}
                    and all(call.get("function", {}).get("name") in {"query_order", "query_logistics", "search_knowledge"}
                            for message in messages for call in message.get("tool_calls", []))
                    and self._facts_fallback(result, evidence)):
                stats.facts_only = True
                return result
            result.reply = "本轮答复未能完成可靠性核验，暂不能给出确定结论。请联系人工核实；如涉及已提交的申请，请先查询处理进度，勿重复提交。"
            result.follow_up_question = None
            result.requires_human = True
            result.task_status = "blocked"
            # No re-execution and no modification of verified sources/business state.
            for outcome in result.task_outcomes:
                if outcome.status == "answered":
                    outcome.status = "blocked"
                    outcome.reason_code = stats.reason_code
        return result

    @staticmethod
    def _facts_fallback(result, evidence, *, policy_unavailable=False):
        """Publish only enum-backed records for a single read-only diagnosis."""
        if (result.intent != IntentType.DELIVERY_EXCEPTION or len(result.task_outcomes) != 1
                or result.task_outcomes[0].agent != "fulfillment"
                or result.task_outcomes[0].status not in ({"answered", "blocked"} if policy_unavailable else {"answered"})
                or any(item.tool not in {"query_order", "query_logistics", "search_knowledge"} for item in evidence)):
            return False
        orders = [item for item in evidence if item.tool == "query_order"]
        shipments = [item for item in evidence if item.tool == "query_logistics"]
        if len(orders) != 1 or len(shipments) != 1:
            return False
        order, shipment = orders[0], shipments[0]
        order_id = order.arguments.get("order_id")
        record, logistics = order.data.get("order"), shipment.data.get("logistics")
        if (not isinstance(order_id, str) or not re.fullmatch(r"ORD-[A-Za-z0-9-]{1,60}", order_id)
                or order.data.get("success") is not True or shipment.data.get("success") is not True
                or not isinstance(record, dict) or not isinstance(logistics, dict)
                or record.get("order_id") != order_id or shipment.arguments.get("order_id") != order_id
                or not record.get("tracking_number")
                or record.get("tracking_number") != logistics.get("tracking_number")):
            return False
        order_labels = {"pending": "待发货", "purchased": "已购买", "refunded": "已退款", "pending_payment": "待付款", "cancelled": "已取消", "shipped": "已发货", "delivered": "已送达", "refund_processing": "退款处理中"}
        logistics_labels = {"in_transit": "运输中", "delivered": "已送达", "pending": "待揽收"}
        order_status = record.get("status")
        logistics_status = logistics.get("status")
        if not isinstance(order_status, str) or not isinstance(logistics_status, str):
            return False
        if order_status not in order_labels or logistics_status not in logistics_labels:
            return False
        demo = any(item.data.get("source") in {"demo", "demo_seed"} for item in (order, shipment))
        result.reply = (
            f"已查询到订单 {order_id}：订单记录为“{order_labels[order_status]}”，物流记录为“{logistics_labels[logistics_status]}”。"
            + ("这些是演示数据，不能代表当前真实物流。" if demo else "以上为系统查询记录，不代表承运商实时状态。")
            + ("本轮政策检索未取得有效依据，仅展示已查询的记录，不能据此确认延误责任、赔偿规则或送达时间。建议联系人工客服或承运商核实最新轨迹。"
               if policy_unavailable else "本轮自动质检服务暂不可用，仅展示已查询的记录，暂不能确认物流停滞原因或承诺送达时间。建议联系人工客服或承运商核实最新轨迹。")
        )
        result.follow_up_question = None
        result.requires_human = True
        result.task_status = "partial"
        result.task_outcomes[0].status = "verified"
        result.task_outcomes[0].reason_code = "FACTS_ONLY_POLICY_UNAVAILABLE" if policy_unavailable else "FACTS_ONLY_DIAGNOSIS_UNAVAILABLE"
        return True
