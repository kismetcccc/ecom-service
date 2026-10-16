"""知识库检索工具：通过混合召回和 Cross-encoder 重排回答 FAQ、政策类问题。

与查订单/查物流这类「结构化数据查询」工具不同，
search_knowledge 面向非结构化文本（退换货政策、配送说明、FAQ 等），
返回 Top-K 命中片段及其来源，由 LLM 引用回答。

向量后端由 settings.rag_backend 切换：
- numpy ：手写余弦相似度，零依赖，教学透明
- chroma：嵌入式向量数据库，HNSW 索引，工程代表性
- qdrant：独立向量数据库服务，使用版本化集合和别名切换

为避免每次进程启动都重建索引，单例缓存 Retriever。
"""

import logging
import time
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import Embedder
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.rag.policy import is_current_policy
from app.server.middleware import request_id_var

_retriever: Optional[KnowledgeRetriever] = None
_fallback_retriever: Optional[KnowledgeRetriever] = None
_logger = logging.getLogger(__name__)


def _create_backend(name: str):
    """按名称创建后端；路径和连接参数仍只来自受信配置。"""
    name = name.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(settings.kb_index_path))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    if name == "qdrant":
        return create_backend(
            "qdrant",
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )
    raise ValueError(
        f"未知的 RAG 后端: {name}（可选: numpy / chroma / qdrant）"
    )


def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例。"""
    return _create_backend(settings.rag_backend)


def _create_retriever(backend_name: str) -> KnowledgeRetriever:
    embedder = Embedder(
        api_key=settings.embedding_api_key or settings.openai_api_key,
        base_url=settings.embedding_base_url or settings.openai_base_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    retriever = _configure_retriever(embedder, _create_backend(backend_name))
    retriever.load()
    return retriever


def _configure_retriever(embedder, backend) -> KnowledgeRetriever:
    """Production and evaluation share one retrieval configuration."""
    from app.agent.rag.reranker import get_reranker

    return KnowledgeRetriever(
        embedder=embedder, backend=backend,
        mode=settings.rag_retrieval_mode,
        candidate_k=settings.rag_candidate_k, rrf_k=settings.rag_rrf_k,
        allow_demo=settings.rag_allow_demo_policies,
        reranker=get_reranker(
            settings.rag_rerank_model, settings.rag_rerank_device,
            settings.rag_rerank_batch_size, settings.rag_rerank_max_length,
            settings.rag_rerank_local_files_only,
        ) if settings.rag_rerank_enabled and settings.rag_retrieval_mode == "hybrid" else None,
        rerank_top_n=settings.rag_rerank_top_n,
        rerank_failure_mode=settings.rag_rerank_failure_mode,
    )


def _get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _create_retriever(settings.rag_backend)
    return _retriever


def _get_fallback_retriever() -> tuple[str, KnowledgeRetriever] | None:
    global _fallback_retriever
    name = settings.rag_fallback_backend.strip().lower()
    if not name or name == settings.rag_backend.lower():
        return None
    if _fallback_retriever is None:
        _fallback_retriever = _create_retriever(name)
    return name, _fallback_retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever, _fallback_retriever
    _retriever = None
    _fallback_retriever = None


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Log retrieval outcome without storing the user's question or policy text."""
    started = time.perf_counter()
    result = _route_knowledge(query, top_k)
    scores = [item["score"] for item in result.get("results", [])]
    _logger.info(
        "rag_search request_id=%s backend=%s model=%s status=%s hits=%d "
        "index_version=%s score_min=%s score_max=%s score_type=%s rerank_status=%s elapsed_ms=%d",
        request_id_var.get(), result.get("backend", settings.rag_backend), settings.embedding_model,
        result.get("error_code", "OK"), len(scores),
        result.get("index_version", ""),
        min(scores) if scores else "", max(scores) if scores else "",
        next(iter(result.get("results", [])), {}).get("score_type", ""),
        next(iter(result.get("results", [])), {}).get("rerank_status", ""),
        int((time.perf_counter() - started) * 1000),
    )
    return result


def _route_knowledge(query: str, top_k: int = 3) -> dict:
    from app.agent.wiki_runtime import read_state, search_wiki
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error_code": "INVALID_ARGUMENT", "error": "query 不能为空", "results": []}
    try:
        top_k = max(1, min(int(top_k or 3), 5))
    except (TypeError, ValueError, OverflowError):
        return {"success": False, "error_code": "INVALID_ARGUMENT", "error": "top_k 必须是整数", "results": []}
    try:
        state = read_state()
    except (ValueError, OSError):
        return {"success": False, "error_code": "UNAVAILABLE", "error": "知识来源配置不可用，请联系管理员", "results": []}
    if state.mode == "rag":
        return {**_search_knowledge_impl(query, top_k), "provider": "rag", "knowledge_mode": "rag"}
    wiki = search_wiki(query, top_k, state)
    if state.mode == "wiki" or wiki["success"]:
        return {**wiki, "knowledge_mode": state.mode}
    return {**_search_knowledge_impl(query, top_k), "provider": "rag",
            "knowledge_mode": "wiki_first", "wiki_fallback_reason": wiki.get("error_code", "UNAVAILABLE")}


def _search_knowledge_impl(query: str, top_k: int = 3) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma" | "qdrant",
          "query": str,
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error_code": "INVALID_ARGUMENT", "error": "query 不能为空", "query": query, "results": []}

    try:
        top_k = max(1, min(int(top_k or 3), 5))
    except (TypeError, ValueError):
        return {
            "success": False,
            "error_code": "INVALID_ARGUMENT",
            "error": "top_k 必须是 1 到 5 的整数",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    backend_used = settings.rag_backend
    primary_missing = False
    try:
        # 多取候选后再进行政策发布状态过滤，避免草稿占满 Top-K。
        retriever = _get_retriever()
        hits = retriever.search(query, top_k=top_k * 4)
    except Exception as primary_error:
        primary_missing = isinstance(primary_error, FileNotFoundError)
        _logger.warning(
            "RAG primary unavailable backend=%s type=%s",
            settings.rag_backend, type(primary_error).__name__,
        )
        try:
            fallback = _get_fallback_retriever()
            if fallback is None:
                raise primary_error
            backend_used, retriever = fallback
            hits = retriever.search(query, top_k=top_k * 4)
            _logger.warning(
                "RAG fallback activated primary=%s fallback=%s",
                settings.rag_backend, backend_used,
            )
        except Exception as fallback_error:
            _logger.warning(
                "RAG fallback unavailable backend=%s type=%s",
                settings.rag_fallback_backend or "disabled",
                type(fallback_error).__name__,
            )
            missing = primary_missing and isinstance(fallback_error, FileNotFoundError)
            return {
                "success": False,
                "error_code": "INDEX_MISSING" if missing else "UNAVAILABLE",
                "error": ("知识库索引尚未准备好，请稍后重试或联系人工客服"
                          if missing else "知识库检索暂时不可用，请稍后重试"),
                "backend": settings.rag_backend,
                "query": query,
                "results": [],
            }

    eligible = [h for h in hits if is_current_policy(
        h.chunk, allow_demo=settings.rag_allow_demo_policies
    )]
    if not eligible:
        return {
            "success": False, "error_code": "NO_MATCH",
            "error": "没有找到当前有效的政策依据，请补充问题或联系人工客服",
            "backend": backend_used, "query": query, "results": [],
        }
    thresholds = {"dense": settings.rag_min_score, "rrf": settings.rag_min_rrf_score,
                  "cross_encoder": settings.rag_min_rerank_score}
    accepted = [h for h in eligible if h.score >= thresholds[h.score_type]]
    if not accepted:
        return {
            "success": False, "error_code": "LOW_RELEVANCE",
            "error": "检索结果相关性不足，暂时无法核实政策，请补充信息或联系人工客服",
            "backend": backend_used, "query": query, "results": [],
        }
    return {
        "success": True,
        "backend": backend_used,
        "index_version": getattr(getattr(retriever, "backend", None), "_corpus_version", ""),
        "query": query,
        "results": [
            {
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "score": round(h.score, 6),
                "score_type": h.score_type,
                "dense_score": h.dense_score,
                "bm25_score": h.bm25_score,
                "rrf_score": h.rrf_score,
                "rerank_score": h.rerank_score,
                "rerank_status": h.rerank_status,
                "text": h.chunk.text,
                "chunk_id": h.chunk.chunk_id,
                "policy_id": h.chunk.policy_id,
                "version": h.chunk.version,
                "status": h.chunk.status,
                "effective_from": h.chunk.effective_from,
                "effective_to": h.chunk.effective_to,
                "source_url": h.chunk.source_url,
            }
            for h in accepted[:top_k]
        ],
    }
