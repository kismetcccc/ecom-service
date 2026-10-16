import re

from fastapi.testclient import TestClient

from app.server.api import app


def test_frontend_is_served_at_root():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "并夕夕 · 小飞" in response.text
    assert "'/v1/chat'" in response.text
    assert "查看记忆" in response.text
    assert "清空短期记忆" in response.text
    assert "删除长期记忆" in response.text
    assert "安全登录" in response.text
    assert "Authorization" in response.text
    assert "/v1/auth/${mode}" in response.text
    assert "授权长期记忆" in response.text
    assert "撤回长期记忆授权" in response.text
    assert "导出我的记忆" in response.text
    assert "摘要版本" in response.text
    assert "/summaries" in response.text
    assert "/memory/consent" in response.text
    assert "memorySection" in response.text
    assert "纠正" in response.text
    assert "确认有效" in response.text
    assert "mutateMemoryFact" in response.text
    assert "Memory 指标" in response.text
    assert "showMemoryMetrics" in response.text
    assert "'/memory/metrics'" in response.text
    assert "'/memory/short-term'" in response.text
    assert "'/v1/users/" not in response.text
    assert "`/v1/users/${encodeURIComponent(state.userId)}/memory`" in response.text
    assert "!event.shiftKey && !isComposing" in response.text
    assert "sendMessage($('message').value)" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "条参考来源" in response.text
    assert "document.createElement('summary')" in response.text
    assert 'id="account-toggle"' in response.text
    assert "result.knowledge_status" in response.text
    assert "result.sources" in response.text
    assert "result.task_outcomes" in response.text
    assert "建议人工：" in response.text


def test_frontend_script_only_references_existing_element_ids():
    with TestClient(app) as client:
        html = client.get("/").text

    element_ids = set(re.findall(r'id="([^"]+)"', html))
    script_id_references = set(re.findall(r"\$\('([^']+)'\)", html))
    assert script_id_references <= element_ids
