from app.agent.tools.context import ToolContext
from app.agent.tools.order_store import get_order_store, failure


def get_return_eligibility(order_id: str, *, context: ToolContext | None = None) -> dict:
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().eligibility(context.user_id, order_id)


def preview_refund(order_id: str, reason: str, *, context: ToolContext | None = None) -> dict:
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().preview_refund(context.user_id, order_id, reason)


def apply_refund(
    preview_token: str, idempotency_key: str, *, context: ToolContext | None = None,
) -> dict:
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().apply_refund(context.user_id, preview_token, idempotency_key)


def get_refund_status(
    application_id: str = "", idempotency_key: str = "", *, context: ToolContext | None = None,
) -> dict:
    if context is None:
        return failure("UNAUTHORIZED", "请先登录")
    return get_order_store().refund_status(context.user_id, application_id, idempotency_key)
