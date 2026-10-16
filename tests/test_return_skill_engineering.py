"""Offline, isolated tests for the return skill and the transaction boundary."""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.agent.skills.loader import SkillManager
from app.agent.tools.manager import ToolManager
from app.agent.tools.order_store import OrderStore
from app.config.settings import settings
from app.multi_agent.workflow import run_return_workflow
from tests.test_return_workflow import agents_for, ORDER, DECISION


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    shutil.copytree(Path("app/agent/skills/definitions/process-return"), root / "process-return")
    monkeypatch.setattr(settings, "skills_dir", str(root))
    monkeypatch.setattr(settings, "skills_enabled", True)
    monkeypatch.setattr(settings, "seed_demo_orders", True)
    return root


def test_lazy_resources_and_real_tool_dispatch(bundle):
    manager = SkillManager(str(bundle))
    tools = ToolManager(local_tool_overrides={"load_skill": manager.load_skill})
    base = json.loads(tools.execute_tool("load_skill", {"skill_name": "process-return"}))
    assert base["success"] and "resources" in base
    assert "# 人工交接摘要" not in base["instructions"]
    resource = json.loads(tools.execute_tool("load_skill", {
        "skill_name": "process-return", "resource": "assets/handoff.md"}))
    assert resource["success"] and "人工交接摘要" in resource["content"]
    assert resource["bundle_version"] == base["bundle_version"]


@pytest.mark.parametrize("path", ["../../.env", "assets/unknown.md", "/etc/passwd"])
def test_resource_paths_are_allowlisted(bundle, path):
    assert not SkillManager(str(bundle)).load_skill("process-return", resource=path)["success"]


def test_resource_edit_invalidates_bundle_but_not_historical_receipt(bundle, tmp_path):
    manager = SkillManager(str(bundle))
    old = manager.load_skill("process-return")
    store = OrderStore(tmp_path / "orders.db")
    preview = store.preview_refund("user-a", ORDER, "不需要")
    path = bundle / "process-return/assets/handoff.md"
    path.write_text("新版交接模板", encoding="utf-8")
    assert manager.load_skill("process-return")["error_code"] == "SKILL_CHANGED_OR_UNAVAILABLE"
    latest = SkillManager(str(bundle)).load_skill("process-return")
    assert latest["version"] == old["version"]
    assert latest["bundle_version"] != old["bundle_version"]
    token = preview["preview_token"]
    assert store.confirm_preview("user-a", token)["success"]
    receipt = store.apply_refund("user-a", token, token)
    assert receipt["skill_execution"]["bundle_version"] == old["bundle_version"]


@pytest.mark.parametrize("mutation", ["../secret.md", "assets/missing.md"])
def test_invalid_manifest_not_discovered(bundle, mutation):
    path = bundle / "process-return/SKILL.md"
    path.write_text(path.read_text(encoding="utf-8").replace("assets/handoff.md", mutation), encoding="utf-8")
    assert SkillManager(str(bundle)).skill_count == 0


def test_workflow_exposes_stage_not_business_success(bundle):
    agents, calls = agents_for()
    result, _ = run_return_workflow(f"退款 {ORDER}", DECISION, agents)
    assert result.skill_execution.state == "waiting_confirmation"
    assert result.skill_execution.policy_scope == "rag_wiki_current_turn"
    assert result.refund_action.order_id == ORDER
    assert result.refund_action.workflow_id == result.skill_execution.run_id
    assert not {"apply_refund", "preview_refund"}.intersection(calls)


@pytest.mark.parametrize("utterance", ["换货", "部分退款", "退其中一件"])
def test_unsupported_scope_never_becomes_full_refund(bundle, utterance):
    agents, calls = agents_for()
    result, _ = run_return_workflow(f"{utterance} {ORDER}", DECISION, agents)
    assert result.refund_action is None and not calls
    assert result.skill_execution.reason_code == "UNSUPPORTED_SCOPE"
    assert result.skill_execution.resources == ["references/exceptions.md"]


def test_missing_bundle_stops_new_previews_not_existing_resolution(bundle, tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    preview = store.preview_refund("user-a", ORDER, "不需要")
    token = preview["preview_token"]
    (bundle / "process-return/SKILL.md").unlink()
    assert store.preview_refund("user-a", ORDER, "不需要")["error_code"] == "UNAVAILABLE"
    assert store.confirm_preview("user-a", token)["success"]
    result = store.apply_refund("user-a", token, token)
    assert store.refund_status("user-a", result["application_id"])["success"]


def test_disabled_skill_keeps_transaction_guards(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "skills_enabled", False)
    store = OrderStore(tmp_path / "orders.db")
    preview = store.preview_refund("user-a", ORDER, "不需要")
    assert preview["skill_execution"]["state"] == "disabled"
    token = preview["preview_token"]
    assert store.apply_refund("user-a", token, token)["error_code"] == "CONFIRMATION_REQUIRED"


def test_http_chain_snapshot_identity_and_lost_response(bundle, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.server import api
    from app.server.auth import AuthService
    from app.agent.tools import order_store
    store = OrderStore(tmp_path / "orders.db")
    auth = AuthService(str(tmp_path / "auth.db"))
    monkeypatch.setattr(api, "auth_service", auth)
    monkeypatch.setattr(order_store, "get_order_store", lambda: store)
    headers = {"Authorization": f"Bearer {auth.issue_token('user-a')}"}
    foreign = {"Authorization": f"Bearer {auth.issue_token('user-b')}"}
    agents, _ = agents_for()
    result, _ = run_return_workflow(f"退款 {ORDER}", DECISION, agents)
    with TestClient(api.app) as client:
        payload = {"order_id": result.refund_action.order_id, "workflow_id": result.workflow_id, "reason": "不需要了"}
        preview = client.post("/v1/refunds/preview", json=payload, headers=headers).json()
        token = preview["preview_token"]
        assert preview["skill_execution"]["origin_workflow_id"] == result.workflow_id
        assert preview["skill_execution"]["policy_scope"] == "system_precheck_only"
        assert store.apply_refund("user-a", token, token)["error_code"] == "CONFIRMATION_REQUIRED"
        assert client.post("/v1/refunds/confirm", json={"preview_token": token}, headers=foreign).status_code == 409
        assert client.post("/v1/refunds/confirm", json={"preview_token": token}, headers=headers).status_code == 200
        # Pretend the first successful submission response was lost.
        resolved = client.post("/v1/refunds/resolve", json={"preview_token": token}, headers=headers).json()
        repeated = client.post("/v1/refunds/confirm", json={"preview_token": token}, headers=headers).json()
        assert repeated["application_id"] == resolved["application_id"]
        assert resolved["skill_execution"]["state"] == "submitted"
        assert resolved["skill_execution"]["run_id"] == preview["skill_execution"]["run_id"]
        assert client.post("/v1/refunds/resolve", json={"preview_token": token}, headers=foreign).status_code == 404
    restored = OrderStore(store.path)
    assert restored.refund_status("user-a", resolved["application_id"])["skill_execution"] == resolved["skill_execution"]
    assert token not in json.dumps(resolved["skill_execution"])


def test_concurrent_confirmed_refund_has_one_receipt_and_one_audit(bundle, tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    token = store.preview_refund("user-a", ORDER, "不需要")["preview_token"]
    assert store.confirm_preview("user-a", token)["success"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: store.apply_refund("user-a", token, token), range(4)))
    assert len({r["application_id"] for r in replies}) == 1
    assert len({r["skill_execution"]["run_id"] for r in replies}) == 1
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_audit").fetchone()[0] == 1


def test_existing_database_migrates_without_changing_orders(bundle, tmp_path):
    import sqlite3
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE refund_previews (token TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
            order_id TEXT NOT NULL, reason TEXT NOT NULL, order_version INTEGER NOT NULL,
            expires_at INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0, confirmed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE refunds (application_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
            order_id TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
            idempotency_key TEXT NOT NULL, created_at INTEGER NOT NULL,
            UNIQUE(owner_id,idempotency_key), UNIQUE(owner_id,order_id));
            INSERT INTO refunds VALUES ('RF-OLD','owner','ORD-OLD','test','processing','legacy-key',1);
        ''')
    store = OrderStore(path)
    result = store.refund_status("owner", "RF-OLD")
    assert result["success"] and result["skill_execution"] == {}
    assert store.refund_status("other", "RF-OLD")["success"] is False


def test_resource_capability_boundary_is_same_as_main_skill(bundle):
    manager = SkillManager(str(bundle))
    result = manager.load_skill("process-return", available_tools={"load_skill"}, resource="assets/handoff.md")
    assert result["error_code"] == "SKILL_CAPABILITY_UNAVAILABLE"
