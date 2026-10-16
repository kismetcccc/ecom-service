from app.agent.tools.mock_data import PRODUCTS
from app.database.products import ProductRepository


def test_product_repository_seeds_and_searches():
    repository = ProductRepository("sqlite://")
    repository.create_schema()

    assert repository.seed(PRODUCTS.values()) == 6
    assert repository.count() == 6
    assert repository.search("ELEC-APP-002")[0]["name"] == "Apple AirPods Pro 2"
    assert repository.search("耳机")[0]["product_id"] == "ELEC-APP-002"
    assert repository.search("不存在的商品") == []


def test_product_seed_is_idempotent():
    repository = ProductRepository("sqlite://")
    repository.create_schema()

    repository.seed(PRODUCTS.values())
    repository.seed(PRODUCTS.values())
    assert repository.count() == 6
