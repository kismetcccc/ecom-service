"""Authenticated shopping endpoints; checkout is explicit and never model-triggered."""

from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.order_store import get_order_store
from app.database import get_product_repository


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=64)
    quantity: int = Field(strict=True, ge=1, le=99)
    expected_price: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


def shopping_router(authenticate):
    router = APIRouter(prefix="/v1", tags=["shopping"])

    @router.get("/products")
    def products(q: Annotated[str, Query(max_length=100)] = "", user: str = Depends(authenticate)):
        repository = get_product_repository()
        items = repository.search(q) if q.strip() else repository.search_catalog()
        return {"products": get_order_store().available_products(items)}

    @router.post("/orders")
    def checkout(payload: CheckoutRequest, user: str = Depends(authenticate)):
        result = get_order_store().create_order(user, payload, get_product_repository())
        if not result["success"]:
            raise HTTPException(status_code=409, detail=result["error"])
        return result

    @router.get("/orders")
    def orders(user: str = Depends(authenticate)):
        return get_order_store().list_orders(user)

    @router.get("/orders/{order_id}")
    def order(order_id: str, user: str = Depends(authenticate)):
        result = get_order_store().query_order(user, order_id)
        if not result["success"]:
            raise HTTPException(status_code=404, detail="未找到该订单")
        return result

    @router.post("/orders/{order_id}/cancel")
    def cancel(order_id: str, user: str = Depends(authenticate)):
        result = get_order_store().cancel_purchase(user, order_id)
        if not result["success"]:
            raise HTTPException(status_code=409, detail=result["error"])
        return result

    return router
