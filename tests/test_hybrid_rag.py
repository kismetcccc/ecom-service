"""Offline ranking, policy and backend snapshot contracts; no model downloads."""

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.backends.numpy_backend import NumpyBackend
from app.agent.rag.chunker import Chunk
from app.agent.rag.hybrid import BM25Index, reciprocal_rank_fusion, tokenize
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.rag.reranker import CrossEncoderReranker
from app.agent.tools import knowledge
from app.config.settings import settings


class Embedder:
    model = "test-model"

    def encode_one(self, query):
        return [1.0, 0.0]


def chunk(key, text, **kwargs):
    return Chunk(key, "政策", "条款", text, **kwargs)


def test_bm25_chinese_sku_and_no_zero_score_matches():
    index = BM25Index([chunk("a", "耳机 SKU-123 保修期十二个月"),
                       chunk("b", "服装七天退货"), chunk("c", "")])
    assert index.search("SKU-123 保修", 1)[0].chunk.chunk_id == "a"
    assert index.search("退货", 1)[0].chunk.chunk_id == "b"
    assert index.search("zzzz", 5) == []
    assert "sku-123" in tokenize("ＳＫＵ-123")
    assert BM25Index([]).search("退货", 5) == []


def test_rrf_uses_rank_not_score_scale_and_deduplicates():
    a, b = chunk("a", "a"), chunk("b", "b")
    hits = reciprocal_rank_fusion(
        [RetrievedChunk(a, .9), RetrievedChunk(a, .9), RetrievedChunk(b, .1)],
        [RetrievedChunk(b, 10000)], 60)
    assert [h.chunk.chunk_id for h in hits] == ["b", "a"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0].dense_score == .1
    assert hits[0].bm25_score == 10000
    assert hits[1].score == pytest.approx(1 / 61)


def test_hybrid_recovers_lexical_hit_outside_dense_candidates(tmp_path):
    backend = NumpyBackend(tmp_path / "index.json")
    chunks = [chunk(str(i), "普通商品说明") for i in range(8)] + [chunk("sku", "ZX-999 售后条款")]
    backend.upsert(chunks, [[1., 0.]] * 8 + [[0., 1.]], Embedder.model)
    retriever = KnowledgeRetriever(Embedder(), backend, mode="hybrid", candidate_k=1, rerank_top_n=1)
    hits = retriever.search("ZX-999", 2)
    assert "sku" in [h.chunk.chunk_id for h in hits]
    assert next(h for h in hits if h.chunk.chunk_id == "sku").dense_score is None


def test_policy_filter_before_fusion_and_rerank(tmp_path):
    backend = NumpyBackend(tmp_path / "index.json")
    backend.upsert([chunk("draft", "退款", status="draft"), chunk("ok", "退款")],
                   [[1., 0.], [.8, .2]], Embedder.model)
    class Reranker:
        def rerank(self, query, hits):
            assert [h.chunk.chunk_id for h in hits] == ["ok"]
            return [replace(hits[0], score=.8, score_type="cross_encoder", rerank_status="ok")]
    retriever = KnowledgeRetriever(Embedder(), backend, mode="hybrid", reranker=Reranker())
    assert retriever.search("退款")[0].rerank_status == "ok"
    assert KnowledgeRetriever(Embedder(), backend, mode="hybrid", allow_demo=False).search("退款") == []


def test_numpy_hybrid_refreshes_published_corpus_and_checks_model(tmp_path):
    path = tmp_path / "index.json"
    backend = NumpyBackend(path)
    backend.upsert([chunk("old", "退款")], [[1., 0.]], Embedder.model)
    retriever = KnowledgeRetriever(Embedder(), backend, mode="hybrid")
    assert retriever.search("退款")[0].chunk.chunk_id == "old"
    NumpyBackend(path).upsert([chunk("new", "退款")], [[1., 0.]], Embedder.model)
    assert retriever.search("退款")[0].chunk.chunk_id == "new"
    NumpyBackend(path).upsert([chunk("bad", "退款")], [[1., 0.]], "different-model")
    with pytest.raises(ValueError, match="模型"):
        retriever.search("退款")


@pytest.mark.parametrize("failure_mode", ["rrf", "error"])
def test_rerank_failure_explicit_fallback_or_error(tmp_path, failure_mode):
    backend = NumpyBackend(tmp_path / "index.json")
    backend.upsert([chunk("a", "退款")], [[1., 0.]], Embedder.model)
    class Broken:
        def rerank(self, query, hits):
            raise RuntimeError("model unavailable")
    retriever = KnowledgeRetriever(Embedder(), backend, mode="hybrid", reranker=Broken(),
                                   rerank_failure_mode=failure_mode)
    if failure_mode == "error":
        with pytest.raises(RuntimeError):
            retriever.search("退款")
    else:
        result = retriever.search("退款")[0]
        assert result.score_type == "rrf" and result.rerank_status == "unavailable"


def test_cross_encoder_pairs_batching_sort_and_lazy_reuse(monkeypatch):
    instances = []
    class Model:
        def __init__(self, name, **kwargs):
            instances.append(name)
        def predict(self, pairs, **kwargs):
            assert pairs[0][0] == "退款"
            assert "错误" in pairs[0][1]
            assert kwargs["batch_size"] == 2
            return [.1, .9]
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Sigmoid=lambda: None)))
    reranker = CrossEncoderReranker("test", batch_size=2)
    hits = [RetrievedChunk(chunk("wrong", "错误"), .03), RetrievedChunk(chunk("right", "退款"), .02)]
    for _ in range(2):
        result = reranker.rerank("退款", hits)
        assert [h.chunk.chunk_id for h in result] == ["right", "wrong"]
        assert result[0].rerank_score == .9
    assert instances == ["test"]
    assert hits[0].score == .03  # Original RRF candidates remain untouched.


@pytest.mark.parametrize("scores", [[float("nan")], [], [2.]])
def test_cross_encoder_rejects_invalid_scores(monkeypatch, scores):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Sigmoid=lambda: None)))
    reranker = CrossEncoderReranker("test")
    reranker._model = SimpleNamespace(predict=lambda *a, **kw: scores)
    with pytest.raises(ValueError):
        reranker.rerank("退款", [RetrievedChunk(chunk("a", "退款"), .02)])


def test_tool_score_thresholds_are_separate(monkeypatch):
    hit = RetrievedChunk(chunk("a", "退款"), .02, score_type="rrf", rerank_status="unavailable")
    monkeypatch.setattr(knowledge, "_get_retriever", lambda: SimpleNamespace(search=lambda *a, **kw: [hit]))
    monkeypatch.setattr(settings, "rag_allow_demo_policies", True)
    monkeypatch.setattr(settings, "rag_min_score", .99)
    monkeypatch.setattr(settings, "rag_min_rrf_score", 0.)
    result = knowledge._search_knowledge_impl("退款")
    assert result["success"] and result["results"][0]["rerank_status"] == "unavailable"
    monkeypatch.setattr(settings, "rag_min_rrf_score", .03)
    assert knowledge._search_knowledge_impl("退款")["error_code"] == "LOW_RELEVANCE"
    hit.score_type, hit.score = "cross_encoder", .6
    monkeypatch.setattr(settings, "rag_min_rerank_score", .7)
    assert knowledge._search_knowledge_impl("退款")["error_code"] == "LOW_RELEVANCE"


def test_qdrant_snapshot_tracks_rebuild_and_rollback():
    qdrant = pytest.importorskip("qdrant_client")
    from app.agent.rag.backends.qdrant_backend import QdrantBackend
    backend = QdrantBackend("unused", client=qdrant.QdrantClient(":memory:"))
    backend.upsert([chunk("old", "退款")], [[1., 0.]], Embedder.model)
    old = backend._active_collection()
    retriever = KnowledgeRetriever(Embedder(), backend, mode="hybrid")
    assert retriever.search("退款")[0].chunk.chunk_id == "old"
    backend.upsert([chunk("new", "退款")], [[1., 0.]], Embedder.model)
    assert retriever.search("退款")[0].chunk.chunk_id == "new"
    backend.rollback(old, Embedder.model)
    assert retriever.search("退款")[0].chunk.chunk_id == "old"


def test_chroma_snapshot_supports_hybrid(tmp_path):
    pytest.importorskip("chromadb")
    from app.agent.rag.backends.chroma_backend import ChromaBackend
    backend = ChromaBackend(tmp_path / "chroma", "hybrid_test")
    backend.upsert([chunk("one", "退款"), chunk("two", "付款")], [[1., 0.], [0., 1.]], Embedder.model)
    result = KnowledgeRetriever(Embedder(), backend, mode="hybrid").search("退款", 1)
    assert result[0].chunk.chunk_id == "one"
    assert result[0].bm25_score > 0


def test_qdrant_alias_switch_during_snapshot_does_not_mix_versions(monkeypatch):
    qdrant = pytest.importorskip("qdrant_client")
    from app.agent.rag.backends.qdrant_backend import QdrantBackend
    client = qdrant.QdrantClient(":memory:")
    backend = QdrantBackend("unused", client=client)
    backend.upsert([chunk("old", "退款")], [[1., 0.]], Embedder.model)
    original = client.scroll
    switched = []
    def scroll(**kwargs):
        if not switched:
            switched.append(True)
            backend.upsert([chunk("new", "新退款")], [[1., 0.]], Embedder.model)
        return original(**kwargs)
    monkeypatch.setattr(client, "scroll", scroll)
    dense, corpus = backend.retrieval_snapshot([1., 0.], 3, Embedder.model)
    assert [h.chunk.chunk_id for h in dense] == ["old"]
    assert [c.chunk_id for c in corpus] == ["old"]
    assert backend.retrieval_snapshot([1., 0.], 3, Embedder.model)[1][0].chunk_id == "new"


def test_real_cross_encoder_local_model_inference(tmp_path):
    """Exercise the real torch/transformers adapter offline, not model quality."""
    pytest.importorskip("sentence_transformers")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrefund\npayment\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    tokenizer.save_pretrained(tmp_path)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, num_labels=1,
    ))
    model.save_pretrained(tmp_path)
    reranker = CrossEncoderReranker(str(tmp_path), local_files_only=True)
    result = reranker.rerank("refund", [RetrievedChunk(chunk("a", "refund"), .02),
                                       RetrievedChunk(chunk("b", "payment"), .03)])
    assert len(result) == 2
    assert all(h.score_type == "cross_encoder" and h.rerank_status == "ok" for h in result)
    assert all(0 <= h.score <= 1 for h in result)
