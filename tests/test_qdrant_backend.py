"""Qdrant RAG backend contract tests without a running Docker service."""

import pytest

from app.agent.rag.backends import create_backend
from app.agent.rag.chunker import Chunk
from app.agent.rag.retriever import KnowledgeRetriever, RetrievedChunk
from app.agent.tools import knowledge
from app.config.settings import settings

qdrant_client = pytest.importorskip("qdrant_client")


class FakeEmbedder:
    model = "embedding-test-v1"

    def encode_one(self, _query):
        return [1.0, 0.0]


def _backend(client):
    return create_backend(
        "qdrant", url="http://unused", collection_name="test_kb", client=client
    )


def test_qdrant_build_search_and_atomic_alias_rebuild():
    client = qdrant_client.QdrantClient(":memory:")
    backend = _backend(client)
    first = Chunk("policy-1", "退换货政策", "七天无理由", "七天无理由退货规则")
    second = Chunk("faq-1", "常见问题", "付款", "付款常见问题")
    backend.upsert([first, second], [[1.0, 0.0], [0.0, 1.0]], "embedding-test-v1")
    assert backend.size() == 2
    retriever = KnowledgeRetriever(FakeEmbedder(), _backend(client))
    retriever.load()
    assert retriever.search("退货", top_k=1)[0].chunk == first
    old_physical = backend._active_collection()

    replacement = Chunk("policy-2", "退换货政策", "质量问题", "质量问题退货规则")
    backend.upsert([replacement], [[1.0, 0.0]], "embedding-test-v1")
    assert backend.size() == 1
    assert backend._active_collection() != old_physical
    assert client.collection_exists(old_physical)  # retained for deliberate rollback
    assert _backend(client).search([1.0, 0.0], top_k=2)[0].chunk == replacement


def test_qdrant_rejects_bad_vectors_without_switching_alias():
    client = qdrant_client.QdrantClient(":memory:")
    backend = _backend(client)
    chunk = Chunk("one", "文档", "章节", "正文")
    backend.upsert([chunk], [[1.0, 0.0]], "embedding-test-v1")
    old_physical = backend._active_collection()
    with pytest.raises(ValueError, match="维度"):
        backend.upsert([chunk, chunk], [[1.0, 0.0], [1.0]], "embedding-test-v1")
    assert backend._active_collection() == old_physical
    assert backend.size() == 1


def test_qdrant_versions_and_manual_rollback():
    client = qdrant_client.QdrantClient(":memory:")
    backend = _backend(client)
    backend.upsert([Chunk("old", "旧", "章节", "旧内容")], [[1.0, 0.0]], "embedding-test-v1")
    old = backend._active_collection()
    backend.upsert([Chunk("new", "新", "章节", "新内容")], [[0.0, 1.0]], "embedding-test-v1")
    assert len(backend.versions()) == 2
    assert sum(item["active"] for item in backend.versions()) == 1
    with pytest.raises(ValueError, match="模型"):
        backend.rollback(old, "wrong-model")
    backend.rollback(old, "embedding-test-v1")
    assert backend._active_collection() == old
    assert backend.search([1.0, 0.0], top_k=1)[0].chunk.chunk_id == "old"
    assert backend.versions()[0]["corpus_version"]
    assert backend.versions()[0]["vector_size"] == 2


def test_qdrant_rejects_query_dimension_mismatch():
    backend = _backend(qdrant_client.QdrantClient(":memory:"))
    backend.upsert([Chunk("one", "文档", "章节", "正文")], [[1.0, 0.0]], "embedding-test-v1")
    with pytest.raises(ValueError, match="维度"):
        backend.search([1.0], top_k=1)


def test_qdrant_publish_failure_preserves_active_index(monkeypatch):
    client = qdrant_client.QdrantClient(":memory:")
    backend = _backend(client)
    old = Chunk("old", "旧政策", "章节", "旧版内容")
    backend.upsert([old], [[1.0, 0.0]], "embedding-test-v1")
    old_physical = backend._active_collection()
    monkeypatch.setattr(client, "update_collection_aliases", lambda **_kwargs: False)
    with pytest.raises(RuntimeError, match="别名切换失败"):
        backend.upsert(
            [Chunk("new", "新政策", "章节", "新版内容")],
            [[0.0, 1.0]],
            "embedding-test-v1",
        )
    assert backend._active_collection() == old_physical
    assert backend.search([1.0, 0.0], top_k=1)[0].chunk == old


def test_qdrant_embedding_model_mismatch_is_detected():
    client = qdrant_client.QdrantClient(":memory:")
    backend = _backend(client)
    backend.upsert([Chunk("one", "文档", "章节", "正文")], [[1.0, 0.0]], "old-model")
    with pytest.raises(ValueError, match="不一致"):
        KnowledgeRetriever(FakeEmbedder(), _backend(client)).load()


def test_qdrant_local_persistence_survives_client_restart(tmp_path):
    path = tmp_path / "qdrant"
    first_client = qdrant_client.QdrantClient(path=str(path))
    first = _backend(first_client)
    first.upsert(
        [Chunk("one", "配送说明", "运费", "偏远地区运费说明")],
        [[1.0, 0.0]],
        "embedding-test-v1",
    )
    first_client.close()

    second_client = qdrant_client.QdrantClient(path=str(path))
    try:
        second = _backend(second_client)
        second.load()
        assert second.expected_embedding_model() == "embedding-test-v1"
        assert second.search([1.0, 0.0], top_k=1)[0].chunk.doc == "配送说明"
    finally:
        second_client.close()


def test_qdrant_configuration_is_wired_to_knowledge_and_builder(monkeypatch):
    from app.scripts.build_kb_index import _build_backend

    monkeypatch.setattr(settings, "rag_backend", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "http://127.0.0.1:6333")
    monkeypatch.setattr(settings, "qdrant_collection", "test_kb")
    assert knowledge._create_backend_from_settings()._name == "test_kb"
    backend, target = _build_backend("qdrant")
    assert backend._name == "test_kb"
    assert "127.0.0.1:6333" in target


def test_search_knowledge_handles_backend_outage(monkeypatch):
    class FailingRetriever:
        def search(self, _query, top_k):
            raise ConnectionError("internal qdrant endpoint")

    monkeypatch.setattr(settings, "rag_backend", "qdrant")
    monkeypatch.setattr(settings, "rag_fallback_backend", "")
    monkeypatch.setattr(knowledge, "_get_retriever", lambda: FailingRetriever())
    result = knowledge.search_knowledge("退货政策", top_k=3)
    assert result["error_code"] == "UNAVAILABLE"
    assert "internal qdrant endpoint" not in result["error"]


def test_search_knowledge_hides_initialization_exception(monkeypatch):
    def failed_init():
        raise ConnectionError("internal qdrant endpoint")

    monkeypatch.setattr(settings, "rag_backend", "qdrant")
    monkeypatch.setattr(settings, "rag_fallback_backend", "")
    monkeypatch.setattr(knowledge, "_get_retriever", failed_init)
    result = knowledge.search_knowledge("退货政策")
    assert result["error_code"] == "UNAVAILABLE"
    assert "internal qdrant endpoint" not in result["error"]


def test_qdrant_outage_uses_compatible_read_only_fallback(monkeypatch):
    class FallbackRetriever:
        backend = object()

        @staticmethod
        def search(_query, top_k):
            return [RetrievedChunk(
                Chunk("policy", "退换货政策", "条件", "七天无理由退货规则"),
                0.9,
            )][:top_k]

    monkeypatch.setattr(settings, "rag_backend", "qdrant")
    monkeypatch.setattr(settings, "rag_fallback_backend", "numpy")
    monkeypatch.setattr(knowledge, "_get_retriever", lambda: (_ for _ in ()).throw(
        ConnectionError("qdrant down")
    ))
    monkeypatch.setattr(
        knowledge, "_get_fallback_retriever",
        lambda: ("numpy", FallbackRetriever()),
    )
    result = knowledge.search_knowledge("退货政策")
    assert result["success"] is True
    assert result["backend"] == "numpy"
    assert result["results"][0]["doc"] == "退换货政策"
