"""向量后端：Numpy、Chroma 与独立服务 Qdrant。

同一份知识库可在三种后端间切换：
- 手写余弦（教学透明，零额外依赖，适合小规模）
- Chroma 嵌入式持久化（单机原型）
- Qdrant 独立服务（多进程检索和索引版本发布）

通过 settings.rag_backend 选择，对上层 KnowledgeRetriever / search_knowledge 透明。
"""

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.numpy_backend import NumpyBackend

__all__ = ["RetrievedChunk", "VectorBackend", "NumpyBackend", "create_backend"]


def create_backend(name: str, **kwargs) -> VectorBackend:
    """工厂方法：根据名称创建对应后端。

    name: "numpy" | "chroma" | "qdrant"
    kwargs: 后端特定参数（index_path / persist_dir / collection 等）
    """
    name = (name or "numpy").lower()
    if name == "numpy":
        return NumpyBackend(index_path=kwargs["index_path"])
    if name == "chroma":
        from app.agent.rag.backends.chroma_backend import ChromaBackend

        return ChromaBackend(
            persist_dir=kwargs["persist_dir"],
            collection_name=kwargs.get("collection_name", "ecom_kb"),
        )
    if name == "qdrant":
        from app.agent.rag.backends.qdrant_backend import QdrantBackend

        return QdrantBackend(
            url=kwargs["url"],
            collection_name=kwargs.get("collection_name", "ecom_kb"),
            api_key=kwargs.get("api_key", ""),
            timeout=kwargs.get("timeout", 5.0),
            client=kwargs.get("client"),
        )
    raise ValueError(f"未知的 RAG 后端: {name}（可选: numpy / chroma / qdrant）")
