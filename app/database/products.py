"""SQLite 商品表及查询仓储。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable

from sqlalchemy import JSON, DateTime, Integer, Numeric, String, Text, cast, event, func, or_
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.agent.tools.mock_data import PRODUCTS
from app.config.settings import settings


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    specs: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": float(self.price),
            "stock": self.stock,
            "description": self.description,
            "specs": self.specs or {},
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProductRepository:
    """线程安全连接池之上的商品读写接口。"""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL 不能为空")
        url = make_url(database_url)
        if url.get_backend_name() != "sqlite":
            raise ValueError("商品数据库仅支持 SQLite")

        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)

        engine_options: dict = {
            "pool_pre_ping": True,
            "connect_args": {"check_same_thread": False, "timeout": 30},
        }
        self.engine: Engine = create_engine(database_url, **engine_options)

        @event.listens_for(self.engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def seed(self, products: Iterable[dict]) -> int:
        """按 product_id 幂等写入商品；重复执行会更新而不会新增重复行。"""
        count = 0
        with Session(self.engine) as session:
            for data in products:
                session.merge(Product(**data))
                count += 1
            session.commit()
        return count

    def count(self) -> int:
        with Session(self.engine) as session:
            return session.query(Product).count()

    def search(self, keyword: str, limit: int = 20) -> list[dict]:
        normalized = keyword.strip()
        with Session(self.engine) as session:
            exact = session.get(Product, normalized)
            if exact is not None:
                return [exact.to_dict()]

            tokens = [token for token in normalized.split() if token]
            if not tokens:
                return []

            conditions = []
            for token in tokens:
                pattern = f"%{token}%"
                conditions.extend(
                    (
                        Product.name.ilike(pattern),
                        Product.category.ilike(pattern),
                        Product.description.ilike(pattern),
                        cast(Product.specs, String).ilike(pattern),
                    )
                )
            products = (
                session.query(Product)
                .filter(or_(*conditions))
                .order_by(Product.stock.desc(), Product.product_id)
                .limit(limit)
                .all()
            )
            return [product.to_dict() for product in products]

    def search_catalog(
        self, category: str = "", min_price: float = 0,
        max_price: float | None = None, limit: int = 20,
    ) -> list[dict]:
        """Structured catalog search for recommendations; stable stock-first order."""
        with Session(self.engine) as session:
            query = session.query(Product)
            if category:
                query = query.filter(Product.category.ilike(f"%{category.strip()}%"))
            query = query.filter(Product.price >= min_price)
            if max_price is not None:
                query = query.filter(Product.price <= max_price)
            products = query.order_by(Product.stock.desc(), Product.product_id).limit(limit).all()
            return [product.to_dict() for product in products]


_repository: ProductRepository | None = None
_repository_lock = Lock()


def get_product_repository() -> ProductRepository:
    """获取进程内仓储；首次使用自动建表并导入6条初始商品。"""
    global _repository
    if _repository is None:
        with _repository_lock:
            if _repository is None:
                repository = ProductRepository(settings.database_url)
                repository.create_schema()
                if repository.count() == 0:
                    repository.seed(PRODUCTS.values())
                _repository = repository
    return _repository
