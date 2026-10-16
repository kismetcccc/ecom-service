"""业务数据库访问层。"""

from app.database.products import ProductRepository, get_product_repository

__all__ = ["ProductRepository", "get_product_repository"]
