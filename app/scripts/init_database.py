"""初始化 SQLite 商品库并幂等导入内置的6条商品。"""

from app.agent.tools.mock_data import PRODUCTS
from app.config.settings import settings
from app.database import ProductRepository


def main() -> None:
    repository = ProductRepository(settings.database_url)
    repository.create_schema()
    imported = repository.seed(PRODUCTS.values())
    print(
        f"SQLite 商品库初始化完成：导入/更新 {imported} 条，"
        f"当前共 {repository.count()} 条。"
    )


if __name__ == "__main__":
    main()
