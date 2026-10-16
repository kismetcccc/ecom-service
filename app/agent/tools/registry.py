"""工具注册表：OpenAI function calling schema + 分发执行"""

import json
from typing import Callable

from app.agent.tools.order import query_order
from app.agent.tools.product import query_product, search_products
from app.agent.tools.logistics import query_logistics
from app.agent.tools.refund import apply_refund, get_refund_status, get_return_eligibility, preview_refund
from app.agent.tools.context import ToolContext
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.user_orders import list_user_orders
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.skill_tool import load_skill

_TOOL_MAP: dict[str, Callable] = {
    "query_order": query_order,
    "query_product": query_product,
    "search_products": search_products,
    "query_logistics": query_logistics,
    "apply_refund": apply_refund,
    "preview_refund": preview_refund,
    "get_refund_status": get_refund_status,
    "get_return_eligibility": get_return_eligibility,
    "search_knowledge": search_knowledge,
    "list_user_orders": list_user_orders,
    "recall_user_memory": recall_user_memory,
    "load_skill": load_skill,
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 ORD-20240115-001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_product",
            "description": "根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "商品名称关键词或商品ID，例如「耳机」「运动鞋」「SHOE-270-BK-42」",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 ORD-20240115-001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、常见问题 FAQ）。"
                "当顾客询问规则、流程、时效、是否支持等政策类问题时使用，"
                "比如「能退货吗」「多久到账」「钻石会员有什么权益」「偏远地区包邮吗」。"
                "返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用顾客的原问题或一句简洁中文描述要查的政策点",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，默认 3，最大 5",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_orders",
            "description": (
                "查询当前用户的所有订单概要列表（订单号、状态、商品、金额、下单时间）。"
                "当用户想查订单但未提供订单号，或提供的订单号查不到时，"
                "调用此工具列出订单供用户确认"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": "提交已在页面明确确认的退款预览；缺少已确认凭据时不可调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "preview_token": {
                        "type": "string",
                        "description": "已确认的退款预览令牌",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "本次提交的唯一幂等标识",
                    },
                },
                "required": ["preview_token", "idempotency_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_memory",
            "description": (
                "查询当前用户的记忆信息，包括本次对话提取的短期记忆和跨会话的长期记忆。"
                "当需要回顾用户的偏好、历史问题、会员信息等时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选的查询关键词，用于过滤记忆内容",
                        "default": "",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "加载指定技能的完整操作指令。"
                "当用户问题匹配某个可用技能时，调用此工具获取该技能的详细处理流程，"
                "然后按流程指引使用已有工具完成用户请求。"
                "可用技能会在系统提示中列出。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要加载的技能名称，如 process-return、track-order、product-recommend",
                    },
                    "resource": {"type": "string", "description": "可选：技能返回清单中的资源路径；不传时加载主指令"}
                },
                "required": ["skill_name"],
            },
        },
    },
]

# MCP 与本地模式共享业务契约；敏感工具始终要求服务端身份。
for _name, _description, _properties, _required in [
    ("get_return_eligibility", "查询本人订单的售后资格，不代替最终人工审核", {"order_id": {"type": "string"}}, ["order_id"]),
    ("preview_refund", "预览本人订单的退款申请；用户须在页面明确确认", {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id", "reason"]),
    ("get_refund_status", "查询本人退款申请状态", {"application_id": {"type": "string"}, "idempotency_key": {"type": "string"}}, []),
    ("search_products", "按类目和价格区间查询商品；库存与价格以返回更新时间为准", {"category": {"type": "string"}, "min_price": {"type": "number"}, "max_price": {"type": "number"}, "limit": {"type": "integer"}}, []),
]:
    TOOL_DEFINITIONS.append({"type": "function", "function": {"name": _name, "description": _description, "parameters": {"type": "object", "properties": _properties, "required": _required}}})

PROTECTED_TOOLS = {
    "query_order", "query_logistics", "list_user_orders", "apply_refund",
    "preview_refund", "get_refund_status", "get_return_eligibility",
}


def execute_tool(name: str, arguments: dict, *, context: ToolContext | None = None) -> str:
    """根据工具名称分发执行，返回 JSON 字符串结果。"""
    func = _TOOL_MAP.get(name)
    if not func:
        return json.dumps({"success": False, "error_code": "UNKNOWN_TOOL", "error": "未知工具"}, ensure_ascii=False)
    try:
        if name in PROTECTED_TOOLS:
            if context is None:
                result = {"success": False, "error_code": "UNAUTHORIZED", "error": "请先登录"}
            else:
                result = func(**arguments, context=context)
        else:
            result = func(**arguments)
    except Exception:
        result = {"success": False, "error_code": "TOOL_ERROR", "error": "工具暂时不可用"}
    return json.dumps(result, ensure_ascii=False)
