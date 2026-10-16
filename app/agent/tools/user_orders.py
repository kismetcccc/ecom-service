from app.agent.tools.context import ToolContext
from app.agent.tools.order_store import get_order_store, failure

STATUS_LABELS = {
    "pending": "待发货",
    "purchased": "已购买",
    "refunded": "已退款",
    "pending_payment": "待付款",
    "cancelled": "已取消",
    "shipped": "已发货",
    "delivered": "已签收",
    "refund_processing": "退款中",
}


def list_user_orders(*, context: ToolContext | None = None) -> dict:
    """查询当前用户的所有订单概要列表。"""
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().list_orders(context.user_id)
