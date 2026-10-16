from pathlib import Path
import re
from fastapi.testclient import TestClient
from app.server.api import app


def test_quality_dashboard_is_static_and_does_not_expose_report_files():
    with TestClient(app) as client:
        response = client.get('/quality')
    assert response.status_code == 200
    assert '质量评测台' in response.text
    assert "default-src 'none'" in response.headers['content-security-policy']
    assert 'fetch(' not in response.text
    assert 'innerHTML' not in response.text
    ids = set(re.findall(r'id="([^"]+)"', response.text))
    assert set(re.findall(r"\$\('([^']+)'\)", response.text)) <= ids


def test_chat_links_to_quality_dashboard():
    html = Path('app/server/static/index.html').read_text(encoding='utf-8')
    assert 'href="/quality"' in html
