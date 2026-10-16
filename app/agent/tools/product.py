from sqlalchemy.exc import SQLAlchemyError

from app.database import get_product_repository
from app.agent.tools.order_store import get_order_store


def query_product(keyword: str) -> dict:
    """根据商品名称关键词或商品 ID 查询 SQLite 商品库。"""
    try:
        results = get_order_store().available_products(get_product_repository().search(keyword))
    except (SQLAlchemyError, OSError):
        return {
            "success": False,
            "error": "商品数据库暂时不可用，请稍后重试",
            "products": [],
        }
    return {
        "success": True,
        "count": len(results),
        "products": results,
        "source": "sqlite",
    }


def search_products(
    category: str = "", min_price: float = 0,
    max_price: float = 0, limit: int = 10,
) -> dict:
    """Filter products by category and price for grounded recommendations."""
    if min_price < 0 or max_price < 0 or (max_price and max_price < min_price):
        return {"success": False, "error_code": "INVALID_ARGUMENT", "error": "价格区间无效"}
    if not 1 <= limit <= 20 or len(category) > 100:
        return {"success": False, "error_code": "INVALID_ARGUMENT", "error": "搜索条件超出范围"}
    try:
        products = get_product_repository().search_catalog(
            category, min_price, max_price or None, limit
        )
        products = get_order_store().available_products(products)
    except (SQLAlchemyError, OSError):
        return {"success": False, "error_code": "UNAVAILABLE", "error": "商品数据库暂时不可用"}
    return {"success": True, "products": products, "count": len(products), "source": "sqlite"}
