import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.rag.chunker import Chunk
from app.multi_agent.wiki import compile_wiki, lint_wiki


def fixture_data(citation=None):
    source = Chunk("c1", "退货", "条件", "需提供订单号。最终资格以审核为准。")
    draft = {"pages": [{"title": "退货流程", "summary": "先核验订单", "steps": ["收集订单号"],
                        "citations": [citation or {"chunk_id": "c1", "quote": "需提供订单号。"}]}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(draft, ensure_ascii=False),
        ))]),
    )))
    return client, {"c1": source}


def test_wiki_detects_changed_removed_and_added_sources():
    client, sources = fixture_data()
    snapshot = compile_wiki(client, "fake", sources)
    assert snapshot.status == "draft"
    assert not lint_wiki(snapshot, sources)
    assert lint_wiki(snapshot, {"c1": replace(sources["c1"], version="new")})
    assert lint_wiki(snapshot, {})
    assert lint_wiki(snapshot, {**sources, "c2": replace(sources["c1"], chunk_id="c2")})


@pytest.mark.parametrize("citation", [
    {"chunk_id": "invented", "quote": "需提供订单号。"},
    {"chunk_id": "c1", "quote": "保证退款到账"},
])
def test_wiki_rejects_fabricated_citations(citation):
    client, sources = fixture_data(citation)
    with pytest.raises(ValueError):
        compile_wiki(client, "fake", sources)


def test_empty_or_oversized_sources_never_call_model():
    client, sources = fixture_data()
    with pytest.raises(ValueError):
        compile_wiki(None, "fake", {})
    with pytest.raises(ValueError):
        compile_wiki(None, "fake", {"c1": replace(sources["c1"], text="a" * 33000)})


@pytest.mark.parametrize("content,reason,expected", [
    ("", "stop", "空正文"),
    (None, "stop", "空正文"),
    ("  ", "stop", "空正文"),
    ("", "length", "长度上限"),
    ('{"pages":', "length", "长度上限"),
    (None, "content_filter", "过滤"),
])
def test_model_failure_has_actionable_diagnostic(content, reason, expected):
    client, sources = fixture_data()
    def create(**kwargs):
        assert kwargs["max_tokens"] == 8000
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=reason)],
            usage=SimpleNamespace(completion_tokens=8000,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=7900)))
    client.chat.completions.create = create
    with pytest.raises(ValueError, match=expected) as error:
        compile_wiki(client, "fake", sources, max_tokens=8000)
    assert "reasoning_tokens=7900" in str(error.value)


def test_missing_choices_is_rejected():
    client, sources = fixture_data()
    client.chat.completions.create = lambda **_: SimpleNamespace(choices=[])
    with pytest.raises(ValueError, match="候选结果"):
        compile_wiki(client, "fake", sources)


@pytest.mark.parametrize("disabled", [False, True])
def test_thinking_override_is_explicit_and_preserves_validated_generation(disabled):
    client, sources = fixture_data()
    original = client.chat.completions.create
    def create(**kwargs):
        assert kwargs.get("extra_body") == ({"enable_thinking": False} if disabled else None)
        return original(**kwargs)
    client.chat.completions.create = create
    snapshot = compile_wiki(client, "fake", sources, disable_thinking=disabled)
    assert not lint_wiki(snapshot, sources)


def test_impact_tracks_transitive_dependencies_and_cycles():
    from app.multi_agent.wiki import analyze_impact, source_hash
    client, sources = fixture_data()
    snapshot = compile_wiki(client, "fake", sources)
    sources["c2"] = replace(sources["c1"], chunk_id="c2")
    snapshot.source_hashes["c2"] = source_hash(sources["c2"])
    first = snapshot.content.pages[0]
    second = first.model_copy(deep=True)
    second.title = "售后入口"
    second.citations[0].chunk_id = "c2"
    second.related_titles = [first.title]
    third = second.model_copy(deep=True)
    third.title = "投诉入口"
    third.related_titles = [second.title]
    first.related_titles = [third.title]
    independent = second.model_copy(deep=True)
    independent.title = "独立页面"
    independent.related_titles = []
    snapshot.content.pages += [second, third, independent]
    report = analyze_impact(snapshot, {**sources, "c1": replace(sources["c1"], version="new")})
    rows = {row["title"]: row for row in report["pages"]}
    assert rows[first.title]["status"] == "direct"
    assert rows[third.title]["dependency_path"] == [third.title, second.title, first.title]
    assert rows[independent.title]["status"] == "not_identified"
    assert report["corpus_review_required"]
    assert not report["new_source_review_required"]


def test_new_sources_require_review_even_without_existing_dependency():
    from app.multi_agent.wiki import analyze_impact
    client, sources = fixture_data()
    snapshot = compile_wiki(client, "fake", sources)
    assert not analyze_impact(snapshot, sources)["corpus_review_required"]
    report = analyze_impact(snapshot, {**sources, "new": replace(sources["c1"], chunk_id="new")})
    assert report["added_sources"] == ["new"]
    assert report["new_source_review_required"]
    assert report["pages"][0]["status"] == "not_identified"
    removed = analyze_impact(snapshot, {})
    assert removed["removed_sources"] == ["c1"]
    assert removed["pages"][0]["status"] == "direct"
