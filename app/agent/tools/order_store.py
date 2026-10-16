"""Single SQLite source for demo orders and refund requests.

Mock rows are seeded once for local demonstration. Every lookup is scoped by
the authenticated user; writes are transactional and idempotent.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from app.agent.tools.mock_data import LOGISTICS, ORDERS
from app.config.settings import settings


_DEMO_OWNERS = {
    "ORD-20240115-001": "user-a",
    "ORD-20240120-002": "user-b",
    "ORD-20240110-003": "user-a",
    "ORD-20240118-004": "user-b",
    "ORD-20240122-005": "user-a",
}


def failure(code: str, message: str) -> dict:
    return {"success": False, "error_code": code, "error": message}


class OrderStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.order_database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at INTEGER NOT NULL,
                    origin TEXT NOT NULL DEFAULT 'business'
                )"""
            )
            order_columns = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
            if "origin" not in order_columns:
                conn.execute("ALTER TABLE orders ADD COLUMN origin TEXT NOT NULL DEFAULT 'demo_seed'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_owner ON orders(owner_id)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS refund_previews (
                    token TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    order_id TEXT NOT NULL, reason TEXT NOT NULL,
                    order_version INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    confirmed INTEGER NOT NULL DEFAULT 0
                )"""
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(refund_previews)")}
            if "confirmed" not in columns:
                conn.execute("ALTER TABLE refund_previews ADD COLUMN confirmed INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS refunds (
                    application_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    order_id TEXT NOT NULL, reason TEXT NOT NULL,
                    status TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(owner_id, idempotency_key), UNIQUE(owner_id, order_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS refund_audit (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    application_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    event TEXT NOT NULL, created_at INTEGER NOT NULL
                )"""
            )
            # Additive migration: persist server-produced skill provenance, never credentials.
            for table in ("refund_previews", "refunds"):
                if "skill_context" not in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN skill_context TEXT NOT NULL DEFAULT '{{}}'")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS order_logistics (
                    tracking_number TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    payload TEXT NOT NULL, updated_at INTEGER NOT NULL,
                    origin TEXT NOT NULL DEFAULT 'business'
                )"""
            )
            # 库存占用与订单在同一数据库提交；商品库 stock 是供给总量，
            # 可售库存 = 供给总量 - 此台账占用量，避免跨库双写事务。
            conn.execute(
                """CREATE TABLE IF NOT EXISTS purchase_reservations (
                    owner_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    order_id TEXT NOT NULL UNIQUE, product_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK(quantity > 0),
                    expected_price TEXT NOT NULL,
                    PRIMARY KEY(owner_id, idempotency_key)
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS purchase_product ON purchase_reservations(product_id)")
            # 仅迁移本购物入口创建的旧待付款订单；不改变退款中、已取消或外部订单。
            for row in conn.execute(
                "SELECT o.* FROM orders o JOIN purchase_reservations r ON r.order_id=o.order_id "
                "WHERE o.status='pending_payment' AND o.origin='business'"
            ).fetchall():
                payload = json.loads(row["payload"])
                payload.update(status="purchased", payment_status="simulated_paid", purchase_mode="simulation")
                conn.execute("UPDATE orders SET status='purchased', payload=?, version=version+1, updated_at=? WHERE order_id=? AND status='pending_payment'",
                             (json.dumps(payload, ensure_ascii=False), int(time.time()), row["order_id"]))
            if settings.seed_demo_orders:
                for order_id, order in ORDERS.items():
                    updated = int(datetime.fromisoformat(
                        order.get("delivered_at") or order.get("shipped_at") or order["created_at"]
                    ).timestamp())
                    conn.execute(
                        "INSERT OR IGNORE INTO orders(order_id,owner_id,payload,status,updated_at,origin) VALUES(?,?,?,?,?,?)",
                        (order_id, _DEMO_OWNERS[order_id], json.dumps(order, ensure_ascii=False), order["status"], updated, "demo_seed"),
                    )
                    tracking = order.get("tracking_number")
                    if tracking and tracking in LOGISTICS:
                        conn.execute(
                            "INSERT OR IGNORE INTO order_logistics VALUES(?,?,?,?,?)",
                            (tracking, _DEMO_OWNERS[order_id], json.dumps(LOGISTICS[tracking], ensure_ascii=False), updated, "demo_seed"),
                        )

    def available_products(self, products: list[dict]) -> list[dict]:
        """Return current sellable inventory without mutating catalog records."""
        with self._connect() as conn:
            reserved = dict(conn.execute(
                "SELECT r.product_id, SUM(r.quantity) FROM purchase_reservations r "
                "JOIN orders o ON o.order_id=r.order_id WHERE o.status NOT IN ('cancelled','refunded') GROUP BY r.product_id"
            ).fetchall())
        return [{**item, "stock": max(0, item["stock"] - reserved.get(item["product_id"], 0))}
                for item in products]

    def create_order(self, user_id: str, request, repository) -> dict:
        """One transaction owns deduplication, inventory reservation and order creation."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                "SELECT * FROM purchase_reservations WHERE owner_id=? AND idempotency_key=?",
                (user_id, request.idempotency_key),
            ).fetchone()
            if previous:
                if (previous["product_id"] != request.product_id
                        or previous["quantity"] != request.quantity
                        or Decimal(previous["expected_price"]) != request.expected_price):
                    return failure("IDEMPOTENCY_CONFLICT", "该提交凭据已用于其他订单，请重新核对")
                row = self._row(conn, user_id, previous["order_id"])
                order = json.loads(row["payload"])
                order["status"] = row["status"]
                return {"success": True, "order": order, "replayed": True}

            # 商品 ID 必须精确匹配；金额由数据库计算，前端金额仅用于检测价格变化。
            product = next((p for p in repository.search(request.product_id)
                            if p["product_id"] == request.product_id), None)
            if product is None:
                return failure("NOT_FOUND", "商品不存在，请重新选择")
            price = Decimal(str(product["price"]))
            if price != request.expected_price:
                return failure("PRICE_CHANGED", "商品价格已变化，请刷新商品并重新确认")
            reserved = conn.execute(
                "SELECT COALESCE(SUM(r.quantity),0) FROM purchase_reservations r "
                "JOIN orders o ON o.order_id=r.order_id WHERE r.product_id=? AND o.status NOT IN ('cancelled','refunded')",
                (request.product_id,),
            ).fetchone()[0]
            if product["stock"] - reserved < request.quantity:
                return failure("OUT_OF_STOCK", "可售库存不足，请刷新商品并调整数量")
            now = datetime.now()
            order_id = "ORD-" + now.strftime("%Y%m%d") + "-" + secrets.token_hex(8).upper()
            order = {
                "order_id": order_id, "status": "purchased",
                "items": [{"product_id": product["product_id"], "sku": product["product_id"],
                           "name": product["name"], "quantity": request.quantity, "price": float(price)}],
                "total": float(price * request.quantity),
                "created_at": now.isoformat(timespec="seconds"),
                "tracking_number": None, "shipped_at": None,
                "payment_status": "simulated_paid", "purchase_mode": "simulation",
            }
            conn.execute(
                "INSERT INTO orders(order_id,owner_id,payload,status,updated_at,origin) VALUES(?,?,?,?,?,?)",
                (order_id, user_id, json.dumps(order, ensure_ascii=False), "purchased", int(time.time()), "business"),
            )
            conn.execute(
                "INSERT INTO purchase_reservations VALUES(?,?,?,?,?,?)",
                (user_id, request.idempotency_key, order_id, product["product_id"], request.quantity, str(price)),
            )
            return {"success": True, "order": order, "replayed": False}

    def cancel_purchase(self, user_id: str, order_id: str) -> dict:
        """Cancel an unshipped simulated purchase and release inventory atomically."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, user_id, order_id)
            if row is None:
                return failure("NOT_FOUND", "未找到该订单")
            payload = json.loads(row["payload"])
            if (payload.get("purchase_mode") != "simulation"
                    or row["status"] not in {"purchased", "cancelled"} or payload.get("tracking_number")):
                return failure("INVALID_STATE", "仅未发货的模拟购买订单支持直接取消")
            payload.update(status="cancelled", payment_status="simulated_cancelled")
            conn.execute("UPDATE orders SET status='cancelled', payload=?, version=version+1, updated_at=? "
                         "WHERE order_id=? AND owner_id=? AND status='purchased'",
                         (json.dumps(payload, ensure_ascii=False), int(time.time()), order_id, user_id))
            return {"success": True, "order_id": order_id, "status": "cancelled"}

    @staticmethod
    def _row(conn: sqlite3.Connection, user_id: str, order_id: str):
        source_filter = "" if settings.seed_demo_orders else " AND origin!='demo_seed'"
        return conn.execute(
            "SELECT * FROM orders WHERE order_id=? AND owner_id=?" + source_filter,
            (order_id, user_id),
        ).fetchone()

    def query_order(self, user_id: str, order_id: str) -> dict:
        with self._connect() as conn:
            row = self._row(conn, user_id, order_id)
            if row is None:
                return failure("NOT_FOUND", "未找到该订单，请核实订单号")
            order = json.loads(row["payload"])
            order.pop("user", None)
            order["status"] = row["status"]
            return {
                "success": True,
                "order": order,
                "source": row["origin"],
                "purchase_mode": order.get("purchase_mode", ""),
                "as_of": row["updated_at"],
            }

    def list_orders(self, user_id: str) -> dict:
        with self._connect() as conn:
            source_filter = "" if settings.seed_demo_orders else " AND origin!='demo_seed'"
            rows = conn.execute(
                "SELECT * FROM orders WHERE owner_id=?" + source_filter + " ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        orders = []
        for row in rows:
            payload = json.loads(row["payload"])
            orders.append({
                "order_id": row["order_id"],
                "status": row["status"],
                "items_summary": "、".join(item["name"] for item in payload["items"]),
                "total": payload["total"],
                "created_at": payload["created_at"],
                "purchase_mode": payload.get("purchase_mode", ""),
                "source": row["origin"],
            })
        return {
            "success": True,
            "count": len(orders),
            "orders": orders,
            "source": "sqlite",
        }

    def query_logistics(self, user_id: str, order_id: str) -> dict:
        result = self.query_order(user_id, order_id)
        if not result["success"]:
            return result
        tracking = result["order"].get("tracking_number")
        if not tracking:
            return failure("NOT_SHIPPED", "该订单尚未发货，暂无物流信息")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM order_logistics WHERE tracking_number=? AND owner_id=?",
                (tracking, user_id),
            ).fetchone()
        if row is None or (not settings.seed_demo_orders and row["origin"] == "demo_seed"):
            return failure("NOT_FOUND", "暂无物流轨迹")
        return {
            "success": True,
            "logistics": json.loads(row["payload"]),
            "source": row["origin"],
            "as_of": row["updated_at"],
        }

    @staticmethod
    def _can_apply(row: sqlite3.Row) -> bool:
        if row["status"] == "purchased":
            return json.loads(row["payload"]).get("purchase_mode") == "simulation"
        if row["status"] not in {"pending", "shipped", "delivered"}:
            return False
        if row["status"] == "delivered":
            delivered = json.loads(row["payload"]).get("delivered_at")
            # Beyond seven days, quality claims require evidence and manual review.
            if not delivered:
                return False
            age_seconds = int(time.time()) - int(datetime.fromisoformat(delivered).timestamp())
            if age_seconds > 7 * 86400:
                return False
        return True

    def eligibility(self, user_id: str, order_id: str) -> dict:
        with self._connect() as conn:
            row = self._row(conn, user_id, order_id)
            if row is None:
                return failure("NOT_FOUND", "未找到该订单，请核实订单号")
            eligible = self._can_apply(row)
        return {
            "success": True,
            "eligible": eligible,
            "reason_code": "CAN_APPLY" if eligible else "NEEDS_MANUAL_REVIEW",
            "order_id": order_id,
            "policy_note": "初步校验，最终资格须以售后审核为准",
            "source": "order_status",
        }

    def preview_refund(self, user_id: str, order_id: str, reason: str, *, origin_workflow_id: str = "") -> dict:
        if not reason.strip() or len(reason) > 500:
            return failure("INVALID_ARGUMENT", "请填写 1 至 500 字的退款原因")
        with self._connect() as conn:
            conn.execute("DELETE FROM refund_previews WHERE expires_at<? OR used=1", (int(time.time()),))
            row = self._row(conn, user_id, order_id)
            if row is None:
                if not settings.seed_demo_orders:
                    return failure("UNAVAILABLE", "仅支持当前账号的模拟购买退款，正式售后接口尚未配置")
                return failure("NOT_FOUND", "未找到该订单，请核实订单号")
            simulated = json.loads(row["payload"]).get("purchase_mode") == "simulation"
            if not settings.seed_demo_orders and not simulated:
                return failure("UNAVAILABLE", "正式售后接口尚未配置，无法提交退款申请")
            if not self._can_apply(row):
                return failure("NOT_ELIGIBLE", "该订单暂不能自助申请，请联系人工核实")
            from app.agent.skills.return_runtime import load_return_skill, execution_snapshot
            loaded = load_return_skill()
            if not loaded.get("success") and loaded.get("error_code") != "SKILLS_DISABLED":
                return failure("UNAVAILABLE", "退款处理规范暂不可用，请稍后重试")
            snapshot = execution_snapshot(loaded, "previewed" if loaded.get("success") else "disabled")
            # Correlation hint only; client-supplied workflow IDs grant no authority.
            snapshot["origin_workflow_id"] = origin_workflow_id[:32]
            token = secrets.token_urlsafe(32)
            expires = int(time.time()) + 600
            conn.execute(
                """INSERT INTO refund_previews
                   (token,owner_id,order_id,reason,order_version,expires_at,skill_context)
                   VALUES(?,?,?,?,?,?,?)""",
                (token, user_id, order_id, reason.strip(), row["version"], expires, json.dumps(snapshot)),
            )
            order = json.loads(row["payload"])
        return {
            "success": True,
            "preview_token": token,
            "order_id": order_id,
            "reason": reason.strip(),
            "amount": order["total"],
            "expires_at": expires,
            "simulation": simulated,
            "skill_execution": snapshot,
            "message": "请核对订单和原因后明确确认；预览不等于提交",
        }

    def confirm_preview(self, user_id: str, preview_token: str) -> dict:
        """Only the authenticated confirmation endpoint invokes this method."""
        with self._connect() as conn:
            preview = conn.execute(
                "SELECT * FROM refund_previews WHERE token=? AND owner_id=?",
                (preview_token, user_id),
            ).fetchone()
            if preview is None or preview["expires_at"] < int(time.time()) or preview["used"]:
                return failure("CONFIRMATION_REQUIRED", "退款预览已失效，请重新核实")
            row = self._row(conn, user_id, preview["order_id"])
            if row is None or row["version"] != preview["order_version"]:
                return failure("ORDER_CHANGED", "订单状态已变化，请重新核实")
            snapshot = json.loads(preview["skill_context"])
            if snapshot and snapshot.get("state") != "disabled":
                snapshot["state"] = "confirmed"
            conn.execute("UPDATE refund_previews SET confirmed=1,skill_context=? WHERE token=?", (json.dumps(snapshot), preview_token))
            return {"success": True, "order_id": preview["order_id"]}

    def apply_refund(self, user_id: str, preview_token: str, idempotency_key: str) -> dict:
        if not preview_token or not idempotency_key or len(idempotency_key) > 128:
            return failure("INVALID_ARGUMENT", "缺少有效确认或幂等标识")
        if idempotency_key != preview_token:
            return failure("INVALID_ARGUMENT", "幂等标识必须与确认凭据绑定")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM refunds WHERE owner_id=? AND idempotency_key=?",
                (user_id, idempotency_key),
            ).fetchone()
            if existing:
                return self._refund_result(existing)
            preview = conn.execute(
                "SELECT * FROM refund_previews WHERE token=? AND owner_id=?",
                (preview_token, user_id),
            ).fetchone()
            if preview is None or preview["expires_at"] < int(time.time()) or preview["used"] or not preview["confirmed"]:
                return failure("CONFIRMATION_REQUIRED", "请重新预览并确认退款")
            row = self._row(conn, user_id, preview["order_id"])
            if (
                row is None
                or row["version"] != preview["order_version"]
                or not self._can_apply(row)
            ):
                return failure("ORDER_CHANGED", "订单状态已变化，请重新核实")
            duplicate = conn.execute(
                "SELECT 1 FROM refunds WHERE owner_id=? AND order_id=?",
                (user_id, preview["order_id"]),
            ).fetchone()
            if duplicate:
                return failure("ALREADY_PROCESSING", "该订单已有退款申请")
            payload = json.loads(row["payload"])
            simulated = payload.get("purchase_mode") == "simulation"
            if not settings.seed_demo_orders and not simulated:
                return failure("UNAVAILABLE", "正式售后接口尚未配置，无法提交退款申请")
            # 模拟购买的售后在本地完成；旧演示订单保持原有“审核中”流程。
            refund_state = "simulated_completed" if simulated else "processing"
            order_state = "refunded" if simulated else "refund_processing"
            payload["status"] = order_state
            if simulated:
                payload["payment_status"] = "simulated_refunded"
            application_id = "RF-" + secrets.token_hex(8).upper()
            now = int(time.time())
            snapshot = json.loads(preview["skill_context"])
            if snapshot and snapshot.get("state") != "disabled":
                snapshot["state"] = "submitted"
            conn.execute(
                "INSERT INTO refunds(application_id,owner_id,order_id,reason,status,idempotency_key,created_at,skill_context) VALUES(?,?,?,?,?,?,?,?)",
                (application_id, user_id, preview["order_id"], preview["reason"], refund_state, idempotency_key, now, json.dumps(snapshot)),
            )
            conn.execute(
                """UPDATE orders SET status=?, payload=?,
                   version=version+1, updated_at=?
                   WHERE order_id=? AND owner_id=?""",
                (order_state, json.dumps(payload, ensure_ascii=False), now, preview["order_id"], user_id),
            )
            conn.execute("UPDATE refund_previews SET used=1 WHERE token=?", (preview_token,))
            conn.execute(
                "INSERT INTO refund_audit(application_id,owner_id,event,created_at) VALUES(?,?,?,?)",
                (application_id, user_id, refund_state if simulated else "submitted", now),
            )
            return {
                "success": True,
                "application_id": application_id,
                "order_id": preview["order_id"],
                "status": refund_state,
                "skill_execution": snapshot,
                "created_at": now,
                "message": "模拟退款已完成，库存已释放，未发生真实资金退回" if simulated else "退款申请已受理，处理结果请以状态查询为准",
            }

    @staticmethod
    def _refund_result(row: sqlite3.Row) -> dict:
        return {
            "success": True,
            "application_id": row["application_id"],
            "order_id": row["order_id"],
            "status": row["status"],
            "skill_execution": json.loads(row["skill_context"]),
            "created_at": row["created_at"],
        }

    def refund_status(self, user_id: str, application_id: str = "", idempotency_key: str = "") -> dict:
        if not application_id and not idempotency_key:
            return failure("INVALID_ARGUMENT", "请提供申请编号或幂等标识")
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM refunds WHERE owner_id=?
                   AND (application_id=? OR idempotency_key=?)""",
                (user_id, application_id, idempotency_key),
            ).fetchone()
        return self._refund_result(row) if row else failure("NOT_FOUND", "未找到退款申请")


@lru_cache(maxsize=1)
def get_order_store() -> OrderStore:
    """Initialize schema once per process, then use short SQLite connections per call."""
    return OrderStore()
