"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
"""

import json

from openai import OpenAI

from app.agent.context_budget import ContextBudgetManager
from app.console import safe_print
from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT
from app.prompts.agent_quality import FULFILLMENT_PROMPT
from app.agent.tools.manager import ToolManager
from app.agent.tools.trace import action_summary, result_summary
from app.config.settings import settings


# MA-07/11：并行和路由故障回退只允许这些已知只读工具。
READ_ONLY_TOOLS = frozenset({
    "query_product", "search_products", "search_knowledge", "list_user_orders",
    "query_order", "query_logistics", "get_return_eligibility",
    "get_refund_status", "recall_user_memory", "load_skill",
})


AGENT_CONFIGS = {
    "fulfillment": {
        "name": "小飞-履约诊断",
        "prompt": FULFILLMENT_PROMPT,
        "tools": {"query_order", "query_logistics", "list_user_orders", "search_knowledge"},
    },
    "presale": {
        "name": "小飞-售前",
        "prompt": PRESALE_PROMPT,
        "tools": {"query_product", "search_products", "search_knowledge", "list_user_orders", "load_skill"},
    },
    "postsale": {
        "name": "小飞-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {
            "query_order", "query_logistics", "apply_refund",
            "get_return_eligibility", "preview_refund", "get_refund_status",
            "list_user_orders", "search_knowledge", "load_skill",
        },
    },
    "complaint": {
        "name": "小飞-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "search_knowledge", "load_skill"},
    },
}


class SubAgent:
    """专业子 Agent：拥有独立的 prompt 和工具子集，执行 ReAct 循环。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_manager: ToolManager,
        client: OpenAI,
        model: str,
        temperature: float,
        context_budget: ContextBudgetManager,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature
        self.context_budget = context_budget

    def handle(
        self, messages: list[dict], max_steps: int = 5,
        allowed_tools: frozenset[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """执行 ReAct 循环，返回 (最终文本, 新增消息列表)。"""
        new_messages: list[dict] = []
        working = list(messages)
        # 既从模型可见 schema 移除，也在执行前再次校验，不能只靠 Prompt 限权。
        tool_definitions = [
            item for item in self.tool_manager.tool_definitions
            if allowed_tools is None or item["function"]["name"] in allowed_tools
        ]

        for _ in range(max_steps):
            self.context_budget.ensure_within_budget(
                working, tool_definitions
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working,
                temperature=self.temperature,
                tools=tool_definitions,
                max_tokens=self.context_budget.reserved_output_tokens,
                timeout=settings.multi_agent_model_timeout_seconds,
            )
            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                msg = {"role": "assistant", "content": content}
                new_messages.append(msg)
                return content, new_messages

            msg_dict: dict = {"role": "assistant", "content": assistant_msg.content}
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
            new_messages.append(msg_dict)
            working.append(msg_dict)

            for call_index, tc in enumerate(assistant_msg.tool_calls):
                func_name = tc.function.name
                invalid_arguments = False
                try:
                    func_args = json.loads(tc.function.arguments)
                except (TypeError, ValueError):
                    func_args = {}
                    invalid_arguments = True
                if not isinstance(func_args, dict):
                    func_args = {}
                    invalid_arguments = True

                self._print_action(func_name, func_args)
                if invalid_arguments:
                    result_str = json.dumps({
                        "success": False, "error_code": "INVALID_TOOL_ARGUMENTS",
                        "error": "工具参数必须是 JSON 对象",
                    }, ensure_ascii=False)
                elif allowed_tools is not None and func_name not in allowed_tools:
                    result_str = json.dumps({
                        "success": False, "error_code": "UNAUTHORIZED",
                        "error": "当前子任务无权调用此工具",
                    }, ensure_ascii=False)
                else:
                    result_str = self.tool_manager.execute_tool(func_name, func_args)
                try:
                    tool_result = json.loads(result_str)
                except (TypeError, ValueError):
                    tool_result = None
                if not isinstance(tool_result, dict) or type(tool_result.get("success")) is not bool:
                    # Unknown outcome must never authorize a dependent action.
                    # Do not retry: a write may already have taken effect.
                    tool_result = {
                        "success": False, "error_code": "INVALID_TOOL_RESULT",
                        "error": "工具结果不符合契约；执行结果未知，请核验后再操作，不要自动重试",
                    }
                    result_str = json.dumps(tool_result, ensure_ascii=False)
                self._print_observation(result_str)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
                new_messages.append(tool_msg)
                working.append(tool_msg)

                if isinstance(tool_result, dict) and tool_result.get("success") is False:
                    # Complete the protocol for every declared call without
                    # executing siblings after a failure (especially writes).
                    for skipped in assistant_msg.tool_calls[call_index + 1:]:
                        new_messages.append({
                            "role": "tool", "tool_call_id": skipped.id,
                            "content": json.dumps({"success": False, "error_code": "DEPENDENCY_BLOCKED"}),
                        })
                    # 编排器对任何失败工具结果都会标记 blocked；继续让模型重试或润色
                    # 不会改变结果，只会增加模型费用并可能重复访问故障依赖。
                    return (
                        "相关业务信息或政策暂无法核实，请稍后重试或联系人工客服。",
                        new_messages,
                    )

        self.context_budget.ensure_within_budget(working)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=working,
            temperature=self.temperature,
            max_tokens=self.context_budget.reserved_output_tokens,
            timeout=settings.multi_agent_model_timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        new_messages.append({"role": "assistant", "content": content})
        return content, new_messages

    def _print_thought(self, text: str) -> None:
        if settings.log_model_content:
            safe_print(f"\n  💭 [{self.name}·思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        safe_print(f"  🔧 [{self.name}·调用工具] {action_summary(func_name, func_args)}")

    def _print_observation(self, result: str) -> None:
        safe_print(f"  📋 [{self.name}·工具结果] {result_summary(result)}")
