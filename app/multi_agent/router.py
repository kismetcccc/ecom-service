"""结构化路由：规则快路、模型分类与显式安全降级。"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from app.config.settings import settings
from app.prompts.agents import ROUTER_PROMPT
from app.server.middleware import request_id_var

VALID_AGENTS = frozenset({"presale", "postsale", "complaint", "fulfillment"})
VALID_INTENTS = frozenset({
    "product", "order", "logistics", "return", "policy", "complaint",
    "greeting", "account", "other", "delivery_exception",
})
ROUTER_RULE_VERSION = "v4"
_DELIVERY_EXCEPTION = re.compile(
    r"物流.{0,12}(?:异常|停滞|不动|没更新|未更新)|"
    r"快递.{0,12}(?:异常|停滞|不动|没更新|没收到)|"
    r"签收.{0,12}(?:没收到|未收到|未收货)|"
    r"(?:订单|物流).{0,12}(?:状态冲突|状态不一致)|丢件|配送异常"
)
_READ_QUERY = re.compile(r"查|查询|查看|看下|看看|状态|进度|到哪|发货|物流|快递")
_WRITE_REQUEST = re.compile(r"取消|修改|删除|关闭|退货|退款|换货|退掉|退了|申请|赔偿")
_RETURN_POLICY_QUERY = re.compile(
    r"(?:查|查询|查看|了解|咨询|问|说说).{0,8}(?:退货|退款|换货).{0,4}(?:政策|规则|条件|时效|流程)|"
    r"(?:退货|退款|换货).{0,4}(?:政策|规则|条件|时效).{0,8}(?:是什么|怎么规定|如何规定)",
)
_RETURN_ACTION_CUE = re.compile(
    r"(?:我要|我想|想要|帮我|给我|需要)\s*(?:退货|退款|换货|退换货)|"
    r"(?:申请|办理|发起|执行)\s*(?:退货|退款|换货|退换货)|退掉|退了|不想要了",
)
_logger = logging.getLogger(__name__)


def resume_return_request(user_input: str, history: Optional[List[dict]]) -> str | None:
    """仅承接紧邻的系统补单号请求；不从旧历史猜订单，也不把补单号当退款确认。"""
    if not re.fullmatch(r"\s*(?:订单号[：:]?\s*)?ORD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\s*", user_input, re.I):
        return None
    previous = list(history or [])
    if previous and previous[-1].get("role") == "user" and previous[-1].get("content") == user_input:
        previous.pop()
    if not previous or previous[-1].get("role") != "assistant":
        return None
    try:
        response = json.loads(previous[-1].get("content", ""))
    except (TypeError, ValueError):
        return None
    if (not isinstance(response, dict) or response.get("intent") != "return_request"
            or response.get("workflow_status") != "needs_input" or not response.get("workflow_id")):
        return None
    # 只读取紧邻回复对应的上一条用户请求，用于保留“换货”诉求，不继承旧订单号。
    prior_input = next((m.get("content", "") for m in reversed(previous[:-1]) if m.get("role") == "user"), "")
    action = "申请换货" if isinstance(prior_input, str) and "换货" in prior_input and "退换货" not in prior_input else "申请退货"
    return f"{action} {user_input.strip()}"


@dataclass(frozen=True)
class RouteDecision:
    """路由结果不等于业务授权；agent=None 要求澄清或人工建议。"""

    agent: str | None
    intents: tuple[str, ...]
    confidence: float
    source: str
    escalation: bool = False
    risk_level: str = "low"
    fallback_reason: str = ""


def infer_intents(text: str) -> tuple[str, ...]:
    """只提取显式信号；不凭关键词推断订单归属或操作已获确认。"""
    found: list[str] = []
    patterns = (
        ("complaint", r"投诉|举报|消协|监管|曝光|起诉|追责|赔偿"),
        ("return", r"退货|退款|换货|退掉|退了|不想要了|售后申请"),
        ("logistics", r"物流|快递|派送|到哪了|发货|运单"),
        ("order", r"订单|下单|已购|购买记录"),
        ("product", r"推荐|想买|帮我选|有什么商品|款式|库存|价格|规格|优惠"),
        ("policy", r"政策|规则|时效|七天无理由|会员权益|包邮"),
        ("account", r"登录|账号|账户|密码|验证码|支付|盗刷|银行卡"),
    )
    for intent, pattern in patterns:
        if re.search(pattern, text, re.I):
            found.append(intent)
    if _DELIVERY_EXCEPTION.search(text):
        found.append("delivery_exception")
    # “查退货政策”中的“退货”是政策主题，不代表用户授权执行售后写操作。
    if ("return" in found and "policy" in found
            and _RETURN_POLICY_QUERY.search(text)
            and not _RETURN_ACTION_CUE.search(re.sub(
                r"(?:不|不要|不用|无需|不需要)(?:申请|办理|发起|执行)(?:退换货|退货|退款|换货)", "", text))):
        found.remove("return")
    if not found and re.fullmatch(r"\s*(你好|您好|嗨|hi|hello|在吗)[！!。\s]*", text, re.I):
        found.append("greeting")
    return tuple(found)


class Router:
    """先运行高精度快路，再向模型请求严格 JSON；失败不默认售后。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(self, user_input: str, history: Optional[List[dict]] = None) -> str | None:
        """兼容旧调用方；新编排器应使用包含降级原因的 decide。"""
        return self.decide(user_input, history).agent

    def decide(self, user_input: str, history: Optional[List[dict]] = None) -> RouteDecision:
        started = time.perf_counter()
        resumed = resume_return_request(user_input, history)
        if resumed and settings.multi_agent_workflows_enabled:
            decision = RouteDecision("postsale", ("return",), 1.0, "workflow_continuation")
            self._record(decision, started)
            return decision
        explicit = infer_intents(user_input)
        rule = self._fast_rule(user_input, explicit) if settings.multi_agent_rules_enabled else None
        if rule:
            self._record(rule, started)
            return rule

        try:
            raw = self._call_model(user_input, history)
            decision = self._parse(raw, explicit, user_input)
        except Exception as exc:
            _logger.warning("route_model_failed request_id=%s type=%s", request_id_var.get(), type(exc).__name__)
            decision = self._safe_fallback(explicit, "MODEL_UNAVAILABLE", user_input)
        self._record(decision, started)
        return decision

    @staticmethod
    def _fast_rule(user_input: str, explicit: tuple[str, ...]) -> RouteDecision | None:
        if re.search(r"验证码|支付密码|盗刷|银行卡|账户被盗", user_input):
            return RouteDecision(None, explicit or ("account",), 1.0,
                                 f"rule:{ROUTER_RULE_VERSION}", escalation=True, risk_level="high",
                                 fallback_reason="ACCOUNT_SECURITY")
        if re.search(r"(?:转|找|要)(?:接)?人工|人工客服", user_input):
            return RouteDecision(None, explicit or ("other",), 1.0,
                                 f"rule:{ROUTER_RULE_VERSION}", escalation=True,
                                 fallback_reason="HUMAN_REQUEST")
        if explicit == ("greeting",):
            return RouteDecision("postsale", explicit, 1.0, f"rule:{ROUTER_RULE_VERSION}")
        if explicit == ("complaint",):
            return RouteDecision("complaint", explicit, 1.0, f"rule:{ROUTER_RULE_VERSION}", escalation=True)
        if explicit == ("policy",):
            return RouteDecision("presale", explicit, 0.99, f"rule:{ROUTER_RULE_VERSION}")
        # 明确的退换货申请直接进入只读核验工作流，避免分类模型故障反复追问。
        # 这不是写操作授权；退款依然必须经独立预览和用户确认。
        if (settings.multi_agent_workflows_enabled and explicit == ("return",)
                and re.search(r"申请|办理|处理|我要|我想|帮我|退掉|退了", user_input)
                and not re.search(r"不想|不要|不用|不需要|不申请|取消|别|无需", user_input)):
            return RouteDecision("postsale", explicit, 0.99, f"rule:{ROUTER_RULE_VERSION}")
        if ("delivery_exception" in explicit and not _WRITE_REQUEST.search(user_input)
                and set(explicit) <= {"delivery_exception", "order", "logistics"}):
            return RouteDecision("fulfillment", explicit, 0.99, f"rule:{ROUTER_RULE_VERSION}")
        # 订单/物流只读查询（包括“查看所有订单”）能力归属明确，无需额外模型分类。
        if (Router._safe_read_query(user_input)
                and set(explicit) in ({"order"}, {"logistics"}, {"order", "logistics"})):
            return RouteDecision("postsale", explicit, 0.99, f"rule:{ROUTER_RULE_VERSION}")
        return None

    @staticmethod
    def _safe_read_query(user_input: str) -> bool:
        return bool(_READ_QUERY.search(user_input)) and not bool(_WRITE_REQUEST.search(user_input))

    def _call_model(self, user_input: str, history: Optional[List[dict]]) -> str:
        recent_context = ""
        if history:
            context_messages = history
            if history[-1].get("role") == "user" and history[-1].get("content") == user_input:
                context_messages = history[:-1]
            recent = [m for m in context_messages[-4:] if m.get("role") in ("user", "assistant")]
            lines = []
            for message in recent:
                content = message.get("content", "")
                if isinstance(content, str) and 0 < len(content) < 200:
                    role = "用户" if message["role"] == "user" else "客服"
                    lines.append(f"{role}: {content}")
            if lines:
                recent_context = "【最近对话】\n" + "\n".join(lines) + "\n\n"
        routing_data = f"{recent_context}【当前用户消息】\n{user_input}\n\n请输出分类结果。"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": routing_data}],
            temperature=0.0,
            max_tokens=160,
            timeout=settings.multi_agent_model_timeout_seconds,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse(raw: str, explicit: tuple[str, ...], user_input: str = "") -> RouteDecision:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return Router._safe_fallback(explicit, "INVALID_OUTPUT", user_input)
        if not isinstance(data, dict):
            return Router._safe_fallback(explicit, "INVALID_OUTPUT", user_input)
        agent = data.get("agent")
        intents = data.get("intents")
        confidence = data.get("confidence")
        escalation = data.get("escalation")
        risk = data.get("risk_level")
        if (
            not isinstance(agent, str) or agent not in VALID_AGENTS
            or not isinstance(intents, list) or not intents
            or any(not isinstance(item, str) or item not in VALID_INTENTS for item in intents)
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1
            or not isinstance(escalation, bool)
            or risk not in {"low", "medium", "high"}
        ):
            return Router._safe_fallback(explicit, "INVALID_OUTPUT", user_input)
        ordered = tuple(dict.fromkeys([*intents, *explicit]))
        escalation = escalation or "complaint" in explicit
        if "return" in ordered and "return" not in explicit:
            return RouteDecision(None, ordered, float(confidence), "clarify", escalation, risk, "UNCONFIRMED_WRITE_INTENT")
        if confidence < settings.multi_agent_route_min_confidence:
            return RouteDecision(None, ordered, confidence, "clarify", escalation, risk, "LOW_CONFIDENCE")
        if "account" in ordered and risk == "high":
            return RouteDecision(None, ordered, confidence, "clarify", True, risk, "HIGH_RISK_ACCOUNT")
        # Business actions must be assigned to an agent with the required tools.
        if "delivery_exception" in ordered and "return" not in ordered:
            agent = "fulfillment"
            source = "capability_override"
        elif any(item in ordered for item in ("return", "order", "logistics")) and agent != "postsale":
            agent = "postsale"
            source = "capability_override"
        elif "product" in ordered and agent == "complaint" and not escalation:
            agent = "presale"
            source = "capability_override"
        else:
            source = "model"
        return RouteDecision(agent, ordered, float(confidence), source, escalation, risk)

    @staticmethod
    def _safe_fallback(explicit: tuple[str, ...], reason: str, user_input: str = "") -> RouteDecision:
        # Only unambiguous, read-only requests may bypass a failed model response.
        if ("delivery_exception" in explicit and not _WRITE_REQUEST.search(user_input)
                and set(explicit) <= {"delivery_exception", "order", "logistics"}):
            return RouteDecision("fulfillment", explicit, 0.8, f"rule_fallback:{ROUTER_RULE_VERSION}", fallback_reason=reason)
        if (explicit in (("order",), ("logistics",), ("order", "logistics"))
                and Router._safe_read_query(user_input)):
            return RouteDecision("postsale", explicit, 0.8, f"rule_fallback:{ROUTER_RULE_VERSION}", fallback_reason=reason)
        if explicit in (("product",), ("policy",)):
            return RouteDecision("presale", explicit, 0.8, f"rule_fallback:{ROUTER_RULE_VERSION}", fallback_reason=reason)
        return RouteDecision(None, explicit, 0.0, "clarify", fallback_reason=reason)

    @staticmethod
    def _record(decision: RouteDecision, started: float) -> None:
        _logger.info(
            "multi_agent_route request_id=%s source=%s agent=%s intents=%s confidence=%.2f fallback=%s elapsed_ms=%d",
            request_id_var.get(), decision.source, decision.agent or "none",
            ",".join(decision.intents), decision.confidence, decision.fallback_reason,
            int((time.perf_counter() - started) * 1000),
        )
