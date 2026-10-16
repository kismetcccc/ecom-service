"""知识库检索器：Dense 基线或 BM25 + Dense → RRF → Cross-encoder。

Retriever 负责查询编码、有效政策过滤、BM25 召回、融合和重排；
向量存储与 Dense 打分交给 VectorBackend 实现：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）
- QdrantBackend：独立向量服务 + 版本化集合/别名切换

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from dataclasses import replace
import logging

from app.agent.rag.hybrid import BM25Index, reciprocal_rank_fusion
from app.agent.rag.policy import is_current_policy
from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder

_logger = logging.getLogger(__name__)

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend, *,
                 mode: str = "dense", candidate_k: int = 40, rrf_k: int = 60,
                 reranker=None, rerank_top_n: int = 20,
                 rerank_failure_mode: str = "rrf", allow_demo: bool = True):
        if mode not in {"dense", "hybrid"} or rerank_failure_mode not in {"rrf", "error"}:
            raise ValueError("无效的检索或重排降级模式")
        if min(candidate_k, rrf_k, rerank_top_n) <= 0:
            raise ValueError("检索候选数量及 RRF k 必须为正数")
        self._embedder = embedder
        self._backend = backend
        self._loaded = False
        self._mode, self._candidate_k, self._rrf_k = mode, candidate_k, rrf_k
        self._reranker, self._rerank_top_n = reranker, rerank_top_n
        self._rerank_failure_mode, self._allow_demo = rerank_failure_mode, allow_demo
        self._sparse_index = None

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip() or top_k <= 0:
            return []
        if not self._loaded:
            self.load()
        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError("索引的 Embedding 模型已变化，请重建检索器或恢复兼容索引")
        q_vec = self._embedder.encode_one(query)
        if self._mode == "dense":
            return self._backend.search(q_vec, top_k=top_k)
        count = max(self._candidate_k, top_k, self._rerank_top_n)
        dense, chunks = self._backend.retrieval_snapshot(q_vec, count * 4, self._embedder.model)
        eligible = {c.chunk_id: c for c in chunks
                    if is_current_policy(c, allow_demo=self._allow_demo)}
        dense = [h for h in dense if h.chunk.chunk_id in eligible][:count]
        corpus = list(eligible.values())
        sparse_index = self._sparse_index
        if sparse_index is None or sparse_index.chunks != corpus:
            sparse_index = BM25Index(corpus)
            self._sparse_index = sparse_index
        sparse = sparse_index.search(query, count)
        fused = reciprocal_rank_fusion(dense, sparse, self._rrf_k)
        if self._reranker is None or not fused:
            return fused[:top_k]
        candidates = fused[:max(top_k, self._rerank_top_n)]
        try:
            return self._reranker.rerank(query, candidates)[:top_k]
        except Exception as exc:
            if self._rerank_failure_mode == "error":
                raise
            _logger.warning("rag_reranker_unavailable type=%s fallback=rrf", type(exc).__name__)
            return [replace(h, rerank_status="unavailable") for h in fused[:top_k]]
