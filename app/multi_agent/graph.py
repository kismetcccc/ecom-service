"""LangGraph 多 Agent 控制流；会话事务与存储仍由 MemorySessionService 管理。

不配置 checkpointer 或节点重试，避免重放已经生效的退货/退款操作。
图状态仅存活于一次调用；专家内部 ReAct、只读并行额度与业务实现沿用原逻辑。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.multi_agent.planner import TaskPlan
from app.multi_agent.router import RouteDecision, resume_return_request
from app.schemas.response import CustomerServiceResponse, IntentType


class MultiAgentState(TypedDict, total=False):
    user_input: str
    previous_message_ids: set[int]
    review_enabled: bool
    decision: RouteDecision
    plan: TaskPlan
    result: CustomerServiceResponse


def build_multi_agent_graph(owner):
    """每个会话独立编译，节点调用该会话现有组件，不增加模型调用。"""
    def route(state: MultiAgentState):
        return {"decision": owner.router.decide(state["user_input"], owner.raw_messages)}

    def choose_route(state: MultiAgentState):
        decision = state["decision"]
        if decision.agent is None:
            return "clarify"
        if decision.intents == ("greeting",):
            return "greeting"
        if (settings.multi_agent_workflows_enabled and "return" in decision.intents
                and set(decision.intents) <= {"return", "order", "policy", "complaint"}):
            return "return_workflow"
        return "planner"

    def clarify(state: MultiAgentState):
        return {"result": owner._clarify_route(state["decision"])}

    def greeting(state: MultiAgentState):
        return {"result": CustomerServiceResponse(
            intent=IntentType.GREETING, confidence=state["decision"].confidence,
            reply="您好，我是小飞。可以帮您查订单物流、了解商品或处理售后，请告诉我您的需求。",
        )}

    def return_workflow(state: MultiAgentState):
        request = resume_return_request(state["user_input"], owner.raw_messages) or state["user_input"]
        return {"result": owner._run_return(request, state["decision"])}

    def planner(state: MultiAgentState):
        planned = owner._plan_tasks(state["decision"])
        return {"plan": planned} if isinstance(planned, TaskPlan) else {"result": planned}

    def specialists(state: MultiAgentState):
        return {"result": owner._execute_plan(state["decision"], state["plan"])}

    def review(state: MultiAgentState):
        if not state["review_enabled"]:
            return {}
        return {"result": owner._review_result(
            state["user_input"], state["result"], state["previous_message_ids"],
        )}

    graph = StateGraph(MultiAgentState)
    for name, node in (
        ("router", route), ("clarify", clarify), ("greeting", greeting),
        ("return_workflow", return_workflow), ("planner", planner),
        ("specialists", specialists), ("review", review),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", choose_route, {
        name: name for name in ("clarify", "greeting", "return_workflow", "planner")
    })
    graph.add_conditional_edges("planner", lambda state: "review" if "result" in state else "specialists",
                                {"review": "review", "specialists": "specialists"})
    for name in ("clarify", "greeting", "return_workflow", "specialists"):
        graph.add_edge(name, "review")
    graph.add_edge("review", END)
    return graph.compile(name="customer_service_multi_agent")
