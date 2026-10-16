"""有界客服任务计划；只允许经过能力校验的单专家或双专家执行。"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS
from app.multi_agent.router import RouteDecision


class PlanningError(ValueError):
    """计划超出已批准的 Agent 能力或复杂度。"""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    agent: str
    intents: tuple[str, ...]
    read_only: bool
    required_tools: frozenset[str]

    @property
    def scope_instruction(self) -> str:
        return (
            f"当前编排仅委托你处理「{self.label}」。只回答这一子任务；"
            "其他诉求由编排器单独处理。不得声称其他子任务或人工转接已完成。"
        )


@dataclass(frozen=True)
class TaskPlan:
    tasks: tuple[TaskSpec, ...]
    parallel: bool


def build_plan(
    decision: RouteDecision,
    available_tools: dict[str, set[str]] | None = None,
) -> TaskPlan:
    """按业务能力合并同域意图；写风险任务绝不进入并行路径。"""
    if decision.agent is None:
        raise PlanningError("路由尚未确定", "NO_ROUTE")
    intents = set(decision.intents)
    tasks: list[TaskSpec] = []
    service = tuple(item for item in ("order", "logistics", "return", "delivery_exception") if item in intents)
    product = "product" in intents
    policy = "policy" in intents

    if service:
        service_intents = (*service, *(("policy",) if policy else ()), *(("complaint",) if "complaint" in intents else ()))
        required: set[str] = set()
        if "order" in service or "return" in service:
            required.update({"query_order", "list_user_orders"})
        if "logistics" in service:
            required.add("query_logistics")
        if policy or "return" in service:
            required.add("search_knowledge")
        diagnostic = "delivery_exception" in service and "return" not in service
        if "delivery_exception" in service:
            required.update({"query_order", "query_logistics", "list_user_orders"})
        tasks.append(TaskSpec(
            "service", "履约异常诊断" if diagnostic else "售后处理与投诉诉求" if "complaint" in intents else "订单、物流或售后处理",
            "fulfillment" if diagnostic else "postsale", service_intents,
            "return" not in service, frozenset(required),
        ))
    if product:
        tasks.append(TaskSpec(
            "product", "商品咨询", "presale",
            ("product", *(("policy",) if policy and not service else ())),
            True, frozenset({"query_product"}),
        ))
    elif policy and not service:
        tasks.append(TaskSpec(
            "policy", "政策咨询", "presale", ("policy",),
            True, frozenset({"search_knowledge"}),
        ))
    if "complaint" in intents and not service:
        tasks.insert(0, TaskSpec("complaint", "投诉诉求", "complaint", ("complaint",), True, frozenset()))
    if not tasks:
        tasks.append(TaskSpec(
            "general", "当前咨询", decision.agent, decision.intents,
            True, frozenset(),
        ))
    if len(tasks) > settings.multi_agent_max_tasks:
        raise PlanningError("诉求较多，需要分步确认", "TOO_MANY_TASKS")
    for task in tasks:
        configured = AGENT_CONFIGS[task.agent]["tools"]
        available = available_tools.get(task.agent, set()) if available_tools is not None else configured
        # MA-04: 运行时目录和静态专家白名单都必须通过，不能靠动态发现越权。
        if not task.required_tools.issubset(configured) or not task.required_tools.issubset(available):
            raise PlanningError("当前专家不具备处理该诉求所需的工具", "CAPABILITY_UNAVAILABLE")
    # Complaint is a separate escalation signal when a business task exists.
    # The business expert retains the verified workflow; the response can still advise human help.
    parallel = (
        settings.multi_agent_parallel_reads and not decision.escalation and len(tasks) == 2
        and all(task.read_only for task in tasks)
        and len({task.agent for task in tasks}) == 2
    )
    return TaskPlan(tuple(tasks), parallel)
