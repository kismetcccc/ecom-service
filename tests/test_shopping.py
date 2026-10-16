"""Checkout integration tests use isolated databases; no model or payment calls."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from app.agent.tools.order_store import OrderStore
from app.config.settings import settings
from app.database.products import ProductRepository
from app.server import shopping
from app.server.auth import AuthService


@pytest.fixture
def shop(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "seed_demo_orders", False)
    repository = ProductRepository(f"sqlite:///{tmp_path / 'products.db'}")
    repository.create_schema()
    repository.seed([dict(product_id="P1", name="测试鞋", category="鞋", price=19.90,
                          stock=3, description="测试商品", specs={})])
    store = OrderStore(tmp_path / "orders.db")
    auth = AuthService(str(tmp_path / "auth.db"))
    tokens = {user: auth.register(user, "test-password") for user in ("new-user", "other-user")}

    def identity(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(401)
        return auth.verify_token(authorization[7:])

    monkeypatch.setattr(shopping, "get_product_repository", lambda: repository)
    monkeypatch.setattr(shopping, "get_order_store", lambda: store)
    app = FastAPI()
    app.state.test_identity = identity
    app.include_router(shopping.shopping_router(identity))
    with TestClient(app) as client:
        yield client, store, repository, {u: {"Authorization": f"Bearer {t}"} for u, t in tokens.items()}
    repository.engine.dispose()


def payload(key="checkout-key-0001", quantity=1, **changes):
    return dict(product_id="P1", quantity=quantity, expected_price="19.90", idempotency_key=key, **changes)


def test_registered_user_checkout_persistence_isolation_and_cancel(shop):
    client, store, repo, headers = shop
    owner = headers["new-user"]
    assert client.get("/v1/products").status_code == 401
    assert client.get("/v1/orders", headers=owner).json()["orders"] == []
    result = client.post("/v1/orders", json=payload(quantity=2), headers=owner)
    assert result.status_code == 200
    order = result.json()["order"]
    assert order["total"] == 39.8 and order["status"] == "purchased"
    assert order["payment_status"] == "simulated_paid"
    assert OrderStore(store.path).query_order("new-user", order["order_id"])["success"]
    assert client.get(f'/v1/orders/{order["order_id"]}', headers=headers["other-user"]).status_code == 404
    assert client.get("/v1/products", headers=owner).json()["products"][0]["stock"] == 1
    assert client.post("/v1/orders", json=payload(quantity=2), headers=owner).json()["replayed"]
    assert len(client.get("/v1/orders", headers=owner).json()["orders"]) == 1
    assert client.post(f'/v1/orders/{order["order_id"]}/cancel', headers=headers["other-user"]).status_code == 409
    assert client.post(f'/v1/orders/{order["order_id"]}/cancel', headers=owner).status_code == 200
    assert client.get("/v1/products", headers=owner).json()["products"][0]["stock"] == 3
    # A delayed duplicate cannot recreate an order after cancellation.
    assert client.post("/v1/orders", json=payload(quantity=2), headers=owner).json()["order"]["status"] == "cancelled"


def test_invalid_tampered_and_changed_requests_do_not_reserve(shop):
    client, store, repo, headers = shop
    owner = headers["new-user"]
    for invalid in (0, -1, True, 1.5, 100):
        assert client.post("/v1/orders", json=payload(quantity=invalid), headers=owner).status_code == 422
    assert client.post("/v1/orders", json=payload(owner_id="other-user"), headers=owner).status_code == 422
    changed = payload(); changed["expected_price"] = "0.01"
    assert client.post("/v1/orders", json=changed, headers=owner).status_code == 409
    missing = payload(); missing["product_id"] = "missing"
    assert client.post("/v1/orders", json=missing, headers=owner).status_code == 409
    assert client.post("/v1/orders", json=payload(quantity=4), headers=owner).status_code == 409
    assert store.available_products(repo.search("P1"))[0]["stock"] == 3
    assert client.post("/v1/orders", json=payload(), headers=owner).status_code == 200
    assert client.post("/v1/orders", json=payload(quantity=2), headers=owner).status_code == 409


def test_concurrent_checkout_cannot_oversell(shop):
    _, store, repo, _ = shop
    def buy(index):
        request = shopping.CheckoutRequest(**payload(key=f"concurrent-key-{index:04d}", quantity=2))
        return store.create_order(f"user-{index}", request, repo)
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(buy, range(5)))
    assert sum(result["success"] for result in results) == 1
    assert store.available_products(repo.search("P1"))[0]["stock"] == 1


def test_order_and_reservation_roll_back_together(shop):
    _, store, repo, _ = shop
    with store._connect() as conn:
        conn.execute("CREATE TRIGGER reject_purchase BEFORE INSERT ON purchase_reservations "
                     "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(Exception, match="injected failure"):
        store.create_order("new-user", shopping.CheckoutRequest(**payload()), repo)
    assert store.list_orders("new-user")["orders"] == []
    assert store.available_products(repo.search("P1"))[0]["stock"] == 3


def test_concurrent_same_key_creates_one_order(shop):
    _, store, repo, _ = shop
    request = shopping.CheckoutRequest(**payload())
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: store.create_order("new-user", request, repo), range(4)))
    assert all(r["success"] for r in results)
    assert len({r["order"]["order_id"] for r in results}) == 1
    assert store.available_products(repo.search("P1"))[0]["stock"] == 2


def test_purchase_refund_api_full_cycle_without_demo_seed(shop, monkeypatch):
    client, store, repo, headers = shop
    from app.server import api
    from app.agent.tools import order_store
    monkeypatch.setattr(order_store, "get_order_store", lambda: store)
    # Exercise the actual preview/confirm handlers with the same fixture identity.
    identity = client.app.state.test_identity
    client.app.dependency_overrides[api.require_authenticated_user] = identity
    for path, endpoint in [("preview", api.refund_preview), ("confirm", api.refund_confirm), ("resolve", api.resolve_refund)]:
        client.app.add_api_route(f"/v1/refunds/{path}", endpoint, methods=["POST"])
    owner = headers["new-user"]
    order = client.post("/v1/orders", json=payload(quantity=2), headers=owner).json()["order"]
    assert store.eligibility("new-user", order["order_id"])["eligible"]
    assert not store.query_logistics("new-user", order["order_id"])["success"]
    preview = client.post("/v1/refunds/preview", json={"order_id":order["order_id"], "reason":"不需要了"}, headers=owner).json()
    assert preview["simulation"]
    token = preview["preview_token"]
    assert store.apply_refund("new-user", token, token)["error_code"] == "CONFIRMATION_REQUIRED"
    assert client.post("/v1/refunds/confirm", json={"preview_token":token}, headers=headers["other-user"]).status_code == 409
    result = client.post("/v1/refunds/confirm", json={"preview_token":token}, headers=owner).json()
    assert result["status"] == "simulated_completed"
    assert client.post("/v1/refunds/confirm", json={"preview_token":token}, headers=owner).json()["application_id"] == result["application_id"]
    assert client.post("/v1/refunds/resolve", json={"preview_token":token}, headers=owner).json()["status"] == "simulated_completed"
    saved = store.query_order("new-user", order["order_id"])["order"]
    assert saved["status"] == "refunded" and saved["payment_status"] == "simulated_refunded"
    assert store.available_products(repo.search("P1"))[0]["stock"] == 3
    assert not store.cancel_purchase("new-user", order["order_id"])["success"]
    assert not store.eligibility("new-user", order["order_id"])["eligible"]
    assert client.post("/v1/orders", json=payload(quantity=2), headers=owner).json()["order"]["status"] == "refunded"


def test_legacy_pending_migration_is_scoped_and_idempotent(shop):
    _, store, repo, _ = shop
    request = shopping.CheckoutRequest(**payload())
    order = store.create_order("new-user", request, repo)["order"]
    with store._connect() as conn:
        order.update(status="pending_payment", payment_status="unpaid")
        order.pop("purchase_mode")
        conn.execute("UPDATE orders SET status='pending_payment',payload=? WHERE order_id=?", (json.dumps(order), order["order_id"]))
        conn.execute("INSERT INTO orders(order_id,owner_id,payload,status,updated_at) VALUES(?,?,?,?,?)",
                     ("EXTERNAL", "new-user", json.dumps(order), "pending_payment", 0))
    restored = OrderStore(store.path)
    assert restored.query_order("new-user", order["order_id"])["order"]["status"] == "purchased"
    assert restored.query_order("new-user", "EXTERNAL")["order"]["status"] == "pending_payment"
    with restored._connect() as conn:
        version = conn.execute("SELECT version FROM orders WHERE order_id=?", (order["order_id"],)).fetchone()[0]
    OrderStore(store.path)
    with restored._connect() as conn:
        assert conn.execute("SELECT version FROM orders WHERE order_id=?", (order["order_id"],)).fetchone()[0] == version
    assert restored.available_products(repo.search("P1"))[0]["stock"] == 2


def test_cancel_invalidates_refund_preview(shop):
    _, store, repo, _ = shop
    order = store.create_order("new-user", shopping.CheckoutRequest(**payload()), repo)["order"]
    preview = store.preview_refund("new-user", order["order_id"], "测试")
    token = preview["preview_token"]
    assert store.confirm_preview("new-user", token)["success"]
    assert store.cancel_purchase("new-user", order["order_id"])["success"]
    assert store.apply_refund("new-user", token, token)["error_code"] == "ORDER_CHANGED"
    assert store.available_products(repo.search("P1"))[0]["stock"] == 3
