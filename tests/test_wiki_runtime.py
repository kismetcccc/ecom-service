import hashlib
import json
from pathlib import Path

import pytest

from app.agent import wiki_runtime as wiki
from app.agent.tools import knowledge
from app.config.settings import settings
from app.multi_agent.wiki import WikiSnapshot, WikiDraft, WikiPage, Citation, source_hash
from app.agent.rag.evidence import attach_knowledge_evidence
from app.schemas.response import CustomerServiceResponse, IntentType
from app.scripts.manage_knowledge import main


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    source = tmp_path / "kb"
    source.mkdir()
    (source / "退货.md").write_text("# 退货\n\n需提供订单号。最终资格以审核为准。", encoding="utf-8")
    (source / "policies.json").write_text(json.dumps({"退货": {"status": "demo", "version": "demo-1"}}), encoding="utf-8")
    monkeypatch.setattr(settings, "kb_dir", str(source))
    monkeypatch.setattr(settings, "wiki_store_dir", str(tmp_path / "wiki"))
    monkeypatch.setattr(settings, "knowledge_mode", "rag")
    monkeypatch.setattr(settings, "rag_allow_demo_policies", True)
    chunks = wiki.current_sources()
    chunk = next(iter(chunks.values()))
    draft = WikiSnapshot(created_at="2026-10-10T00:00:00+00:00",
        source_hashes={key: source_hash(value) for key, value in chunks.items()},
        content=WikiDraft(pages=[WikiPage(title="退货申请流程", summary="退货需要核验订单。",
            steps=["提供订单号后核验"], citations=[Citation(chunk_id=chunk.chunk_id, quote="需提供订单号。")])]))
    return source, draft


def test_default_rag_preserves_existing_retriever_and_does_not_create_wiki(corpus, monkeypatch):
    calls = []
    monkeypatch.setattr(knowledge, "_search_knowledge_impl", lambda *args: calls.append(args) or {"success": True, "results": []})
    result = knowledge.search_knowledge("退货")
    assert result["provider"] == "rag" and len(calls) == 1
    assert not wiki.store_dir().exists()


def test_hot_switch_without_embeddings_or_rag_mutation(corpus, monkeypatch):
    source, draft = corpus
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    calls = []
    monkeypatch.setattr(knowledge, "_search_knowledge_impl", lambda *args: calls.append(args) or {"success": True, "results": []})
    release = wiki.publish(draft, "test-reviewer")
    assert wiki.read_state().mode == "rag"  # publishing is not enabling
    assert (wiki.store_dir() / "pages" / release.release_id / "index.md").exists()
    wiki.switch_mode("wiki")
    result = knowledge.search_knowledge("退货")
    assert result["success"] and result["provider"] == "wiki"
    assert result["wiki_pages"][0]["title"] == "退货申请流程"
    assert result["results"][0]["text"] != draft.content.pages[0].summary
    assert result["results"][0]["status"] == "demo"
    assert not calls
    wiki.switch_mode("rag")
    assert knowledge.search_knowledge("退货")["provider"] == "rag"
    assert len(calls) == 1
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}


def test_wiki_first_falls_back_and_strict_wiki_does_not(corpus, monkeypatch):
    _, draft = corpus
    wiki.publish(draft, "reviewer")
    calls = []
    monkeypatch.setattr(knowledge, "_search_knowledge_impl", lambda *args: calls.append(args) or {"success": True, "results": []})
    wiki.switch_mode("wiki")
    assert knowledge.search_knowledge("unrelated")["error_code"] == "NO_MATCH"
    assert not calls
    wiki.switch_mode("wiki_first")
    result = knowledge.search_knowledge("unrelated")
    assert result["provider"] == "rag" and result["wiki_fallback_reason"] == "NO_MATCH"
    assert len(calls) == 1
    assert knowledge.search_knowledge("退货")["provider"] == "wiki"
    assert len(calls) == 1


def test_source_drift_invalidates_wiki_and_can_return_to_rag(corpus, monkeypatch):
    source, draft = corpus
    wiki.publish(draft, "reviewer")
    wiki.switch_mode("wiki")
    (source / "退货.md").write_text("# 退货\n\n新版本规则", encoding="utf-8")
    assert knowledge.search_knowledge("退货")["error_code"] == "UNAVAILABLE"
    with pytest.raises(ValueError):
        wiki.switch_mode("wiki_first")
    wiki.switch_mode("rag")
    assert wiki.read_state().mode == "rag"


def test_published_wiki_first_falls_back_after_source_drift(corpus, monkeypatch):
    source, draft = corpus
    wiki.publish(draft, "reviewer")
    wiki.switch_mode("wiki_first")
    (source / "退货.md").write_text("# 新政策", encoding="utf-8")
    monkeypatch.setattr(knowledge, "_search_knowledge_impl", lambda *args: {"success": False, "error_code": "INDEX_MISSING", "results": []})
    result = knowledge.search_knowledge("退货")
    assert result["provider"] == "rag" and result["wiki_fallback_reason"] == "UNAVAILABLE"
    assert result["error_code"] == "INDEX_MISSING"


def test_demo_disable_blocks_existing_wiki(corpus, monkeypatch):
    _, draft = corpus
    wiki.publish(draft, "reviewer")
    wiki.switch_mode("wiki")
    monkeypatch.setattr(settings, "rag_allow_demo_policies", False)
    assert not knowledge.search_knowledge("退货")["success"]


def test_review_gate_rejects_unresolved_or_fabricated_citation(corpus):
    _, draft = corpus
    draft.content.pages[0].open_questions = ["资格条件尚不确定"]
    with pytest.raises(ValueError):
        wiki.publish(draft, "reviewer")
    draft.content.pages[0].open_questions = []
    draft.content.pages[0].citations[0].quote = "无条件退款"
    with pytest.raises(ValueError):
        wiki.publish(draft, "reviewer")
    assert not wiki.store_dir().exists()


def test_release_activation_and_invalid_identifiers(corpus):
    _, draft = corpus
    first = wiki.publish(draft, "reviewer")
    second = wiki.publish(draft, "reviewer")
    assert wiki.read_state().active_release == second.release_id
    wiki.activate(first.release_id)
    assert wiki.read_state().active_release == first.release_id
    with pytest.raises(ValueError):
        wiki.activate("../../kb")


def test_missing_release_and_corrupt_control_fail_closed_then_recover(corpus, monkeypatch):
    with pytest.raises(ValueError):
        wiki.switch_mode("wiki")
    control = wiki.store_dir() / "control.json"
    control.write_text("{broken", encoding="utf-8")
    assert knowledge.search_knowledge("退货")["error_code"] == "UNAVAILABLE"
    wiki.switch_mode("rag")
    assert wiki.read_state().mode == "rag"


def test_storage_cannot_overlap_rag(corpus, monkeypatch):
    source, _ = corpus
    monkeypatch.setattr(settings, "wiki_store_dir", str(source / "wiki"))
    with pytest.raises(ValueError):
        wiki.switch_mode("rag")
    assert not (source / "wiki").exists()


def test_cli_publish_switch_query_and_status(corpus, tmp_path, capsys):
    _, draft = corpus
    path = tmp_path / "draft.json"
    path.write_text(draft.model_dump_json(), encoding="utf-8")
    assert main(["publish", str(path), "--reviewed-by", "reviewer"]) == 0
    assert main(["switch", "wiki"]) == 0
    assert main(["query", "退货"]) == 0
    assert main(["status"]) == 0
    assert main(["list"]) == 0
    assert "wiki_ready" in capsys.readouterr().out


def test_wiki_preserves_original_policy_evidence(corpus):
    _, draft = corpus
    wiki.publish(draft, "reviewer")
    wiki.switch_mode("wiki")
    result = knowledge.search_knowledge("退货")
    response = CustomerServiceResponse(intent=IntentType.AFTER_SALE, confidence=1, reply="需核验订单")
    attached = attach_knowledge_evidence(response, [
        {"role": "assistant", "tool_calls": [{"id": "s1", "function": {"name": "search_knowledge"}}]},
        {"role": "tool", "tool_call_id": "s1", "content": json.dumps(result)},
    ])
    assert attached.knowledge_providers == ["wiki"]
    assert attached.sources[0].chunk_id == draft.content.pages[0].citations[0].chunk_id
    assert attached.knowledge_status == "VERIFIED_DEMO"


def test_staged_release_can_be_queried_without_changing_active_version(corpus, capsys):
    _, draft = corpus
    first = wiki.publish(draft, "reviewer")
    wiki.switch_mode("wiki")
    candidate = wiki.publish(draft, "reviewer", activate_release=False)
    before = (wiki.store_dir() / "control.json").read_bytes()
    assert main(["query", "退货", "--release", candidate.release_id]) == 0
    assert candidate.release_id in capsys.readouterr().out
    assert main(["impact", candidate.release_id]) == 0
    assert (wiki.store_dir() / "control.json").read_bytes() == before
    assert wiki.read_state().active_release == first.release_id
    wiki.activate(candidate.release_id)
    assert wiki.read_state().active_release == candidate.release_id


def test_staged_release_must_revalidate_sources_at_activation(corpus):
    source, draft = corpus
    first = wiki.publish(draft, "reviewer")
    candidate = wiki.publish(draft, "reviewer", activate_release=False)
    (source / "退货.md").write_text("规则已变更", encoding="utf-8")
    with pytest.raises(ValueError):
        wiki.activate(candidate.release_id)
    assert wiki.read_state().active_release == first.release_id


def test_failed_pointer_replace_keeps_previous_release(corpus, monkeypatch):
    _, draft = corpus
    first = wiki.publish(draft, "reviewer")
    def fail(*args):
        raise OSError("injected replace failure")
    monkeypatch.setattr(wiki.os, "replace", fail)
    with pytest.raises(OSError):
        wiki.publish(draft, "reviewer")
    assert wiki.read_state().active_release == first.release_id


def test_cli_stage_first_release_does_not_enable_it(corpus, tmp_path, capsys):
    _, draft = corpus
    path = tmp_path / "draft.json"
    path.write_text(draft.model_dump_json(), encoding="utf-8")
    assert main(["publish", str(path), "--reviewed-by", "reviewer", "--stage-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["active"] is False
    assert wiki.read_state().active_release is None
