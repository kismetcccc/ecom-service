"""Deterministic return workflow over specialist-scoped, authenticated tools.

No generated prose can advance a dependency or authorize a refund. Every run
rechecks live facts; the existing preview/confirmation API owns the write.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from uuid import uuid4

from app.agent.rag.evidence import attach_knowledge_evidence
from app.schemas.response import CustomerServiceResponse, IntentType, TaskOutcome
from app.schemas.response import SkillExecution, RefundAction
from app.agent.skills.return_runtime import load_return_skill, execution_snapshot


def run_return_workflow(user_input, decision, agents, *, parallel=False, skill_manager=None):
    response, traces = _run_return_workflow(user_input, decision, agents, parallel=parallel, skill_manager=skill_manager)
    if response.skill_execution:
        response.skill_execution.state = response.workflow_status
        if response.workflow_status == "blocked":
            response.skill_execution.reason_code = next((step.reason_code for step in response.task_outcomes if step.status == "blocked"), "PRECHECK_BLOCKED")
    if response.workflow_status == "waiting_confirmation":
        # The action only pre-fills the order. Preview rechecks live ownership/state.
        order_id = next(json.loads(m["tool_calls"][0]["function"]["arguments"])["order_id"]
                        for m in traces if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "query_order")
        response.refund_action = RefundAction(order_id=order_id, workflow_id=response.workflow_id)
    return response, traces


def _run_return_workflow(user_input, decision, agents, *, parallel=False, skill_manager=None):
    """Return (response, protocol-valid audit messages), without mutating history."""
    workflow_id = uuid4().hex
    # Never select a remembered order or silently pick one of several orders.
    order_ids = {value.upper() for value in re.findall(
        r"(?<![A-Za-z0-9_-])ORD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9_-])", user_input, re.I,
    )}
    response = CustomerServiceResponse(
        intent=IntentType.RETURN_REQUEST, confidence=decision.confidence,
        reply="", workflow_id=workflow_id, requires_human=decision.escalation,
    )
    loaded = load_return_skill(skill_manager)
    if loaded.get("success"):
        response.skill_execution = SkillExecution(**execution_snapshot(
            loaded, "needs_input", run_id=workflow_id, policy_scope="rag_wiki_current_turn"))
    elif loaded.get("error_code") != "SKILLS_DISABLED":
        response.skill_execution = SkillExecution(**execution_snapshot(loaded, "blocked", run_id=workflow_id))
        response.task_outcomes = [TaskOutcome(key="skill", label="处理规范加载", status="blocked",
                                             reason_code=loaded.get("error_code", "SKILL_UNAVAILABLE"))]
        response.workflow_status = "blocked"
        response.task_status = "blocked"
        response.requires_human = True
        response.reply = "退款处理规范暂不可用，本轮未执行申请，请稍后重试或联系人工。"
        return response, []
    if len(order_ids) == 1 and any(word in user_input.replace("退换货", "退货") for word in ("换货", "部分退款", "部分退货", "退其中", "只退一")):
        response.workflow_status = "blocked"
        response.task_status = "blocked"
        response.requires_human = True
        response.reply = "当前自助入口仅支持整单退款申请。换货诉求请联系人工；部分退款也需人工核实，本轮未创建退款申请。"
        response.task_outcomes = [TaskOutcome(key="scope", label="办理范围检查", status="blocked", reason_code="UNSUPPORTED_SCOPE")]
        if response.skill_execution:
            resource = load_return_skill(skill_manager, resource="references/exceptions.md")
            if resource.get("success"):
                response.skill_execution.resources.append(resource["resource"])
        return response, []
    if len(order_ids) != 1:
        response.reply = "请提供本次要处理的一个订单号，我会核验订单、退换货政策和售后资格。"
        response.follow_up_question = "请直接回复本次要处理的一个订单号，也可以在「我的订单」中查看并申请退款。"
        response.workflow_status = "needs_input"
        response.task_status = "blocked"
        return response, []
    order_id = next(iter(order_ids)).upper()
    traces = []
    outcomes = []

    def call(agent_key, tool, arguments):
        try:
            manager = agents[agent_key].tool_manager
            refresh = getattr(manager, "available_tool_names", None)
            names = refresh() if callable(refresh) else {
                item["function"]["name"] for item in manager.tool_definitions
            }
            if tool not in names:
                data = {"success": False, "error_code": "CAPABILITY_UNAVAILABLE"}
            else:
                data = json.loads(manager.execute_tool(tool, arguments))
            if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
                data = {"success": False, "error_code": "INVALID_TOOL_RESULT"}
        except Exception:
            data = {"success": False, "error_code": "UNAVAILABLE"}
        call_id = "wf-" + uuid4().hex
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": tool, "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }]},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps(data, ensure_ascii=False)},
        ]
        return data, messages

    jobs = [
        ("order", "订单核验", "postsale", "query_order", {"order_id": order_id}),
        ("policy", "退换货政策核验", "presale", "search_knowledge",
         {"query": "退换货退款申请条件、限制和办理流程", "top_k": 3}),
    ]
    if parallel:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="return-check") as executor:
            futures = [executor.submit(copy_context().run, call, *job[2:]) for job in jobs]
            checks = [future.result() for future in futures]
    else:
        checks = [call(*job[2:]) for job in jobs]
    for job, (data, messages) in zip(jobs, checks):
        valid = data.get("success") is True
        if job[0] == "order":
            valid = valid and isinstance(data.get("order"), dict) and data["order"].get("order_id") == order_id
        else:
            valid = valid and isinstance(data.get("results"), list) and bool(data["results"]) and all(
                isinstance(hit, dict) and hit.get("chunk_id") and hit.get("text")
                for hit in data.get("results", [])
            )
        outcomes.append(TaskOutcome(
            key=job[0], label=job[1], status="verified" if valid else "blocked",
            reason_code="" if valid else str(data.get("error_code", "INVALID_TOOL_RESULT")),
        ))
        if not valid:
            # Invalid successful payloads must not become citations in the common
            # evidence collector, nor look successful to later conversation turns.
            messages[-1]["content"] = json.dumps({
                "success": False, "error_code": outcomes[-1].reason_code, "results": [],
            })
        traces.extend(messages)

    prerequisites_ok = all(outcome.status == "verified" for outcome in outcomes)
    eligible = False
    if prerequisites_ok:
        data, messages = call("postsale", "get_return_eligibility", {"order_id": order_id})
        traces.extend(messages)
        valid = (data.get("success") is True and data.get("order_id") == order_id
                 and isinstance(data.get("eligible"), bool))
        eligible = valid and data["eligible"]
        outcomes.append(TaskOutcome(
            key="eligibility", label="售后资格初审", depends_on=["order", "policy"],
            status="verified" if eligible else "blocked",
            reason_code="" if eligible else str(data.get("error_code", data.get("reason_code", "INVALID_TOOL_RESULT"))),
        ))
    else:
        outcomes.append(TaskOutcome(
            key="eligibility", label="售后资格初审", depends_on=["order", "policy"],
            status="skipped", reason_code="DEPENDENCY_BLOCKED",
        ))
    outcomes.append(TaskOutcome(
        key="confirmation", label="退款预览与用户确认", depends_on=["eligibility"],
        status="waiting_confirmation" if eligible else "skipped",
        reason_code="CONFIRMATION_REQUIRED" if eligible else "DEPENDENCY_BLOCKED",
    ))
    response.task_outcomes = outcomes
    response.task_status = "answered" if eligible else "blocked"
    response.workflow_status = "waiting_confirmation" if eligible else "blocked"
    response.requires_human = decision.escalation or not eligible
    response.reply = (
        f"订单 {order_id} 已完成订单与政策检索核验，符合当前系统的初步自助申请条件，最终资格以售后审核为准。"
        "请在退款申请面板填写原因、核对预览并明确确认。本轮尚未提交退款申请。"
        if eligible else
        f"订单 {order_id} 的自动售后核验未通过，后续申请步骤已停止。请核对订单号，稍后重试或联系人工核实。"
    )
    if "换货" in user_input.replace("退换货", "退货"):
        response.reply += " 当前自助入口办理退款申请；换货诉求请联系人工确认，本轮未创建换货申请。"
        response.requires_human = True
    return attach_knowledge_evidence(response, traces, preserve_reply_on_failure=True), traces
