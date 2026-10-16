"""Qdrant vector store with staged collections and atomic alias switching."""

from __future__ import annotations

import math
import hashlib
from datetime import datetime, timezone
import uuid

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk


class QdrantBackend(VectorBackend):
    """Keep reads on the previous index until a replacement is fully uploaded."""

    def __init__(
        self,
        url: str,
        collection_name: str = "ecom_kb",
        api_key: str = "",
        timeout: float = 5.0,
        client=None,
    ):
        self._url = url
        self._name = collection_name
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._embedding_model = ""
        self._dimension = 0
        self._active_name: str | None = None
        self._corpus_version = ""

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError("请安装 qdrant-client：pip install -r requirements.txt") from exc
            self._client = QdrantClient(
                url=self._url, api_key=self._api_key or None, timeout=self._timeout
            )
        return self._client

    def _active_collection(self) -> str | None:
        aliases = self._get_client().get_aliases().aliases
        return next((item.collection_name for item in aliases if item.alias_name == self._name), None)

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        if not chunks or len(chunks) != len(vectors) or not embedding_model:
            raise ValueError("Qdrant 索引需要非空、等长的 chunks/vectors 和 embedding_model")
        dimension = len(vectors[0])
        if not dimension or any(
            len(vector) != dimension or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError("Qdrant 向量维度不一致或包含非有限数值")

        from qdrant_client import models

        client = self._get_client()
        previous = self._active_collection()
        staged = f"{self._name}_{uuid.uuid4().hex[:12]}"
        corpus_version = hashlib.sha256(
            "\n".join(f"{c.chunk_id}:{c.text}" for c in chunks).encode("utf-8")
        ).hexdigest()[:16]
        try:
            client.create_collection(
                collection_name=staged,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                metadata={
                    "embedding_model": embedding_model,
                    "vector_size": dimension,
                    "corpus_version": corpus_version,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            client.upsert(
                collection_name=staged,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "doc": chunk.doc,
                            "section": chunk.section,
                            "text": chunk.text,
                            "policy_id": chunk.policy_id,
                            "version": chunk.version,
                            "status": chunk.status,
                            "effective_from": chunk.effective_from,
                            "effective_to": chunk.effective_to,
                            "owner": chunk.owner,
                            "source_url": chunk.source_url,
                            "content_hash": chunk.content_hash,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
                wait=True,
            )
            if client.count(collection_name=staged, exact=True).count != len(chunks):
                raise RuntimeError("Qdrant 索引写入数量不匹配，保持旧索引不变")
            operations = []
            if previous:
                operations.append(
                    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self._name))
                )
            operations.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=staged, alias_name=self._name)
                )
            )
            if not client.update_collection_aliases(change_aliases_operations=operations):
                raise RuntimeError("Qdrant 别名切换失败，保持旧索引不变")
        except Exception:
            # Only the new staging collection is disposable; the active index is untouched.
            try:
                if client.collection_exists(staged) and self._active_collection() != staged:
                    client.delete_collection(staged)
            except Exception:
                pass  # Preserve the original publish error if cleanup also fails.
            raise
        self._embedding_model = embedding_model
        self._dimension = dimension
        self._active_name = staged
        self._corpus_version = corpus_version
        # Keep the old physical collection for a deliberate rollback/retention policy.

    def load(self) -> None:
        active = self._active_collection()
        if not active:
            raise FileNotFoundError(
                f"Qdrant 别名 {self._name!r} 不存在；请先运行 "
                "python -m app.scripts.build_kb_index --backend qdrant"
            )
        info = self._get_client().get_collection(active)
        metadata = info.config.metadata or {}
        self._embedding_model = metadata.get("embedding_model", "")
        self._dimension = int(metadata.get("vector_size", 0))
        self._active_name = active
        self._corpus_version = str(metadata.get("corpus_version", ""))
        if not self._embedding_model:
            raise ValueError("Qdrant 索引缺少 embedding_model 元数据，请重建索引")
        if self._dimension <= 0:
            raise ValueError("Qdrant 索引缺少向量维度元数据，请重建索引")

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self._embedding_model or self._active_collection() != self._active_name:
            self.load()
        return self._search_at(query_vector, top_k, self._active_name, self._dimension)

    def _search_at(self, query_vector, top_k, collection, dimension):
        if len(query_vector) != dimension:
            raise ValueError(
                f"查询向量维度({len(query_vector)}) 与索引维度({dimension}) 不一致"
            )
        response = self._get_client().query_points(
            collection_name=collection, query=query_vector, limit=top_k, with_payload=True
        )
        result = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id", "")),
                doc=str(payload.get("doc", "")),
                section=str(payload.get("section", "")),
                text=str(payload.get("text", "")),
                policy_id=str(payload.get("policy_id", "")),
                version=str(payload.get("version", "")),
                status=str(payload.get("status", "")),
                effective_from=str(payload.get("effective_from", "")),
                effective_to=str(payload.get("effective_to", "")),
                owner=str(payload.get("owner", "")),
                source_url=str(payload.get("source_url", "")),
                content_hash=str(payload.get("content_hash", "")),
            )
            result.append(RetrievedChunk(chunk=chunk, score=float(point.score)))
        return result

    def retrieval_snapshot(self, query_vector, top_k, embedding_model):
        # Pin the physical collection so alias publication cannot mix two corpora.
        active = self._active_collection()
        if not active:
            raise FileNotFoundError("Qdrant 知识库索引不存在")
        client = self._get_client()
        metadata = client.get_collection(active).config.metadata or {}
        if metadata.get("embedding_model") != embedding_model:
            raise ValueError("索引的 Embedding 模型已变化，请重建检索器")
        chunks, offset = [], None
        while True:
            points, offset = client.scroll(collection_name=active, limit=256,
                                            offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                fields = {key: value for key, value in (point.payload or {}).items()
                          if key in Chunk.__dataclass_fields__}
                chunks.append(Chunk(**fields))
            if offset is None:
                break
        dense = self._search_at(query_vector, top_k, active, int(metadata.get("vector_size", 0)))
        return dense, chunks

    def size(self) -> int:
        if not self._active_collection():
            return 0
        return self._get_client().count(collection_name=self._name, exact=True).count

    def expected_embedding_model(self) -> str:
        if not self._embedding_model or self._active_collection() != self._active_name:
            try:
                self.load()
            except FileNotFoundError:
                return ""
        return self._embedding_model

    def versions(self) -> list[dict]:
        """List physical index versions without touching the active alias."""
        client = self._get_client()
        active = self._active_collection()
        versions = []
        for collection in client.get_collections().collections:
            if not collection.name.startswith(f"{self._name}_"):
                continue
            info = client.get_collection(collection.name)
            meta = info.config.metadata or {}
            versions.append({
                "collection": collection.name,
                "active": collection.name == active,
                "embedding_model": meta.get("embedding_model", ""),
                "vector_size": meta.get("vector_size", 0),
                "corpus_version": meta.get("corpus_version", ""),
                "built_at": meta.get("built_at", ""),
            })
        return sorted(versions, key=lambda item: item["built_at"], reverse=True)

    def rollback(self, target: str, expected_model: str) -> None:
        """Explicitly repoint the alias to a compatible retained collection."""
        if target not in {item["collection"] for item in self.versions()}:
            raise ValueError("目标不是当前知识库保留的物理集合")
        info = self._get_client().get_collection(target)
        metadata = info.config.metadata or {}
        if metadata.get("embedding_model") != expected_model:
            raise ValueError("目标索引的 Embedding 模型与当前配置不一致")
        if not metadata.get("vector_size") or self._get_client().count(target, exact=True).count == 0:
            raise ValueError("目标索引缺少维度或没有向量")
        from qdrant_client import models

        active = self._active_collection()
        if active == target:
            return
        operations = []
        if active:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self._name))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=target, alias_name=self._name)
            )
        )
        if not self._get_client().update_collection_aliases(change_aliases_operations=operations):
            raise RuntimeError("Qdrant 回滚别名切换失败")
        self.load()
