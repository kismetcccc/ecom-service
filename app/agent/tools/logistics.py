from app.agent.tools.context import ToolContext
from app.agent.tools.order_store import get_order_store, failure


def query_logistics(order_id: str, *, context: ToolContext | None = None) -> dict:
    """根据订单号查询物流轨迹信息。"""
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().query_logistics(context.user_id, order_id)
