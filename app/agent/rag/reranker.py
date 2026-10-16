"""Lazy, shared CrossEncoder with bounded batches and explicit failure semantics."""

from dataclasses import replace
from functools import lru_cache
import math
import threading
import time

from app.agent.rag.backends.base import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8,
                 max_length: int = 512, local_files_only: bool = False):
        self.model_name, self.device = model_name, device
        self.batch_size, self.max_length = batch_size, max_length
        self.local_files_only = local_files_only
        self._model = None
        self._lock = threading.Lock()
        self._retry_at = 0.0

    def rerank(self, query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not hits:
            return []
        with self._lock:
            if time.monotonic() < self._retry_at:
                raise RuntimeError("Cross-encoder 暂不可用，处于重试冷却期")
            try:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(
                        self.model_name, device=self.device, max_length=self.max_length,
                        local_files_only=self.local_files_only, trust_remote_code=False,
                    )
                # Explicit sigmoid makes score semantics consistent across model configs.
                from torch import nn
                scores = self._model.predict(
                    [(query, f"{h.chunk.doc}\n{h.chunk.section}\n{h.chunk.text}") for h in hits],
                    batch_size=self.batch_size, show_progress_bar=False,
                    activation_fn=nn.Sigmoid(), convert_to_numpy=True,
                )
                if len(scores) != len(hits):
                    raise ValueError("Cross-encoder 返回的分数数量不匹配")
                values = [float(score) for score in scores]
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError("Cross-encoder 必须返回有限的单标签相关性分数")
            except Exception:
                self._retry_at = time.monotonic() + 60
                raise
        ranked = [replace(hit, score=value, score_type="cross_encoder",
                          rerank_score=value, rerank_status="ok")
                  for hit, value in zip(hits, values)]
        return sorted(ranked, key=lambda h: (-h.score, -(h.rrf_score or 0), h.chunk.chunk_id))


@lru_cache(maxsize=4)
def get_reranker(model_name: str, device: str, batch_size: int, max_length: int,
                 local_files_only: bool) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name, device, batch_size, max_length, local_files_only)
