"""Download/load the configured reranker before accepting user requests."""

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.reranker import get_reranker
from app.config.settings import settings


def main():
    reranker = get_reranker(settings.rag_rerank_model, settings.rag_rerank_device,
                            settings.rag_rerank_batch_size, settings.rag_rerank_max_length,
                            settings.rag_rerank_local_files_only)
    result = reranker.rerank("退货政策", [RetrievedChunk(
        Chunk("warmup", "售后", "退货", "签收后七天内可申请退货"), 0.)])
    print(f"Cross-encoder ready: {settings.rag_rerank_model}; status={result[0].rerank_status}")


if __name__ == "__main__":
    main()
