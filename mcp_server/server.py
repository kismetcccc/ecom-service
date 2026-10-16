"""MCP Server：通过 Streamable HTTP 暴露电商工具服务。

启动方式：python mcp_server/server.py
默认监听：http://127.0.0.1:9123/mcp
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from mcp.server.mcpserver import MCPServer
    _MCP_V2 = True
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
    _MCP_V2 = False

from app.agent.tools.order import query_order as _query_order
from app.agent.tools.user_orders import list_user_orders as _list_user_orders
from app.agent.tools.product import query_product as _query_product
from app.agent.tools.product import search_products as _search_products
from app.agent.tools.logistics import query_logistics as _query_logistics
from app.agent.tools.refund import (
    apply_refund as _apply_refund,
    get_refund_status as _get_refund_status,
    get_return_eligibility as _get_return_eligibility,
    preview_refund as _preview_refund,
)
from app.agent.tools.knowledge import search_knowledge as _search_knowledge
from app.agent.tools.context import verify_context_token
from app.server.auth import AuthError


def _authorized(func, auth_token: str, **kwargs) -> str:
    """Check identity at the MCP boundary, including direct MCP callers."""
    try:
        context = verify_context_token(auth_token)
    except AuthError:
        result = {"success": False, "error_code": "UNAUTHORIZED", "error": "请先登录"}
    else:
        result = func(context=context, **kwargs)
    return json.dumps(result, ensure_ascii=False)

mcp = (
    MCPServer("ecom-tools") if _MCP_V2
    else MCPServer("ecom-tools", host="127.0.0.1", port=9123)
)


@mcp.tool()
def query_order(order_id: str, auth_token: str = "") -> str:
    """根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等"""
    return _authorized(_query_order, auth_token, order_id=order_id)


@mcp.tool()
def list_user_orders(auth_token: str = "") -> str:
    """查询当前已登录用户的订单列表"""
    return _authorized(_list_user_orders, auth_token)


@mcp.tool()
def query_product(keyword: str) -> str:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索"""
    result = _query_product(keyword)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def search_products(category: str = "", min_price: float = 0, max_price: float = 0, limit: int = 10) -> str:
    """按类目、价格和数量筛选 SQLite 商品；返回库存和更新时间"""
    return json.dumps(_search_products(category, min_price, max_price, limit), ensure_ascii=False)


@mcp.tool()
def query_logistics(order_id: str, auth_token: str = "") -> str:
    """根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件"""
    return _authorized(_query_logistics, auth_token, order_id=order_id)


@mcp.tool()
def get_return_eligibility(order_id: str, auth_token: str = "") -> str:
    """查询本人订单是否可以发起退款申请"""
    return _authorized(_get_return_eligibility, auth_token, order_id=order_id)


@mcp.tool()
def preview_refund(order_id: str, reason: str, auth_token: str = "") -> str:
    """预览退款；用户还必须通过确认页面明确确认"""
    return _authorized(_preview_refund, auth_token, order_id=order_id, reason=reason)


@mcp.tool()
def apply_refund(preview_token: str, idempotency_key: str, auth_token: str = "") -> str:
    """提交已确认的退款申请；重复提交幂等"""
    return _authorized(_apply_refund, auth_token, preview_token=preview_token, idempotency_key=idempotency_key)


@mcp.tool()
def get_refund_status(application_id: str = "", idempotency_key: str = "", auth_token: str = "") -> str:
    """查询当前用户退款申请的处理状态"""
    return _authorized(_get_refund_status, auth_token, application_id=application_id, idempotency_key=idempotency_key)


@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、FAQ）。
    当顾客询问规则、流程、时效、是否支持等政策类问题时使用。
    返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策。
    """
    result = _search_knowledge(query, top_k=top_k)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    if _MCP_V2:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=9123)
    else:
        mcp.run(transport="streamable-http")
