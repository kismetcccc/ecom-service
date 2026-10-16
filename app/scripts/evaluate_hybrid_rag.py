"""Live, paired retrieval ablation through the production knowledge tool.

Uses one embedding and one corpus snapshot per query for every ranking method.
Does not change routing, thresholds, index contents, or the serving processes.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.hybrid import BM25Index, reciprocal_rank_fusion
from app.agent.rag.policy import is_current_policy
from app.agent.tools import knowledge
from app.agent.wiki_runtime import read_state
from app.config.settings import settings
from app.evaluation.statistics import percentile, retrieval_scores


def summarize(rows):
    labeled = [r for r in rows if r.get("relevant_chunk_ids")]
    methods = {}
    for name in ("dense", "bm25", "rrf", "cross_encoder"):
        metrics = [r["metrics"][name] for r in labeled]
        methods[name] = {
            key: sum(m[key] for m in metrics) / len(metrics) if metrics else None
            for key in ("hit_at_1", "recall_at_k", "mrr_at_k", "ndcg_at_k")
        }
    return {"labeled_cases": len(labeled), "methods": methods,
            "successful_queries": sum(r["success"] for r in rows),
            "queries": len(rows),
            "cross_encoder_verified_queries": sum(r["rerank_verified"] for r in rows),
            "warm_tool_latency_ms": {"median": percentile([r["elapsed_ms"] for r in rows], 50),
                                     "p95": percentile([r["elapsed_ms"] for r in rows], 95)},
            "rerank_latency_ms": {"median": percentile([r["rerank_ms"] for r in rows], 50),
                                  "p95": percentile([r["rerank_ms"] for r in rows], 95)}}


def probe_mcp(output):
    """Probe the running MCP process using its real timeout and routing settings."""
    from app.mcp_client import MCPClient
    dataset = json.loads(Path("app/evaluation/retrieval_cases.json").read_text(encoding="utf-8"))
    selected = [c for c in dataset["cases"] if c["id"] in {"return-03", "delivery-04", "member-03"}]
    client = MCPClient(settings.mcp_server_url)
    rows = []
    try:
        client.connect()
        for case in selected:
            started = time.perf_counter()
            result = json.loads(client.call_tool("search_knowledge", {"query": case["query"], "top_k": 3}))
            rows.append({"id": case["id"], "query": case["query"],
                         "elapsed_ms": (time.perf_counter() - started) * 1000,
                         "result": result})
            print(f"MCP {case['id']} success={result.get('success')} provider={result.get('provider')} "
                  f"score_type={next(iter(result.get('results', [])), {}).get('score_type')}", flush=True)
    finally:
        client.close()
    passed = bool(rows) and all(r["result"].get("success") and r["result"].get("provider") == "rag"
                               and r["result"].get("knowledge_mode") == "rag"
                               and r["result"].get("results") and all(
                                   h.get("rerank_status") == "ok" and h.get("score_type") == "cross_encoder"
                                   for h in r["result"]["results"])
                               for r in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": passed, "mcp_call_timeout": settings.mcp_call_timeout,
                                  "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow live embedding requests")
    parser.add_argument("--dataset", type=Path, default=Path("app/evaluation/retrieval_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcp-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("live embedding requests require --live")
    if read_state().mode != "rag":
        parser.error("switch knowledge routing to rag before this evaluation")
    if args.mcp_only:
        return probe_mcp(args.output)
    if settings.rag_retrieval_mode != "hybrid" or not settings.rag_rerank_enabled:
        parser.error("enable hybrid retrieval and cross-encoder first")
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    cases = list(dataset["cases"])
    cases.extend([
        {"id": "diagnostic-weather", "query": "明天北京天气怎么样？", "relevant_chunk_ids": []},
        {"id": "diagnostic-medical", "query": "偏头痛应该服用哪种药？", "relevant_chunk_ids": []},
        {"id": "diagnostic-stock", "query": "今天股票市场哪只股票涨得最多？", "relevant_chunk_ids": []},
    ])
    retriever = knowledge._get_retriever()
    if retriever._reranker is None:
        raise RuntimeError("reranker missing")
    retriever._embedder._client = retriever._embedder._client.with_options(timeout=45, max_retries=0)
    actual_reranker = retriever._reranker
    started = time.perf_counter()
    actual_reranker.rerank("退货", [RetrievedChunk(Chunk("warmup", "售后", "退货", "七天内申请退货"), 0)])
    warmup_ms = (time.perf_counter() - started) * 1000
    capture = {}
    original_snapshot = retriever.backend.retrieval_snapshot

    def snapshot(*a, **kw):
        dense, corpus = original_snapshot(*a, **kw)
        capture.update(dense=dense, corpus=corpus)
        return dense, corpus

    class RecordingReranker:
        def rerank(self, query, hits):
            start = time.perf_counter()
            capture["rerank_candidates"] = hits
            try:
                ranked = actual_reranker.rerank(query, hits)
                capture["reranked"] = ranked
                return ranked
            finally:
                capture["rerank_ms"] = (time.perf_counter() - start) * 1000

    retriever.backend.retrieval_snapshot = snapshot
    retriever._reranker = RecordingReranker()
    rows = []
    report = {"metadata": {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "label_status": dataset.get("label_status"),
        "backend": settings.rag_backend, "embedding_model": settings.embedding_model,
        "rerank_model": settings.rag_rerank_model, "rerank_device": settings.rag_rerank_device,
        "candidate_k": settings.rag_candidate_k, "rrf_k": settings.rag_rrf_k,
        "rerank_top_n": settings.rag_rerank_top_n, "top_k": 3,
        "thresholds": {"dense": settings.rag_min_score, "rrf": settings.rag_min_rrf_score,
                       "cross_encoder": settings.rag_min_rerank_score},
        "model_warmup_ms": warmup_ms,
        "scope": "Single sequential development-set run; shared embedding/snapshot; no answer generation. "
                 "No-answer diagnostics are excluded from ranking metrics. Model warmup excluded from query latency.",
    }, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, case in enumerate(cases):
            capture.clear()
            start = time.perf_counter()
            result = knowledge.search_knowledge(case["query"], top_k=3)
            elapsed = (time.perf_counter() - start) * 1000
            corpus = [c for c in capture.get("corpus", []) if is_current_policy(c, allow_demo=settings.rag_allow_demo_policies)]
            valid_ids = {c.chunk_id for c in corpus}
            if result.get("success") and not set(case["relevant_chunk_ids"]) <= valid_ids:
                raise ValueError(f"stale dataset label: {case['id']}")
            count = max(settings.rag_candidate_k, 12, settings.rag_rerank_top_n)
            dense = [h for h in capture.get("dense", []) if h.chunk.chunk_id in valid_ids][:count]
            sparse = BM25Index(corpus).search(case["query"], count)
            fused = reciprocal_rank_fusion(dense, sparse, settings.rag_rrf_k)
            ranking = {"dense": [h.chunk.chunk_id for h in dense if h.score >= settings.rag_min_score][:3],
                       "bm25": [h.chunk.chunk_id for h in sparse[:3]],
                       "rrf": [h.chunk.chunk_id for h in fused if h.score >= settings.rag_min_rrf_score][:3],
                       "cross_encoder": [h["chunk_id"] for h in result.get("results", [])]
                       if all(h.get("score_type") == "cross_encoder" for h in result.get("results", [])) else []}
            row = {**case, "success": bool(result.get("success")), "elapsed_ms": elapsed,
                   "rerank_ms": capture.get("rerank_ms", 0), "ranking": ranking,
                   "rerank_verified": bool(result.get("results")) and all(
                       h.get("score_type") == "cross_encoder" and h.get("rerank_status") == "ok"
                       for h in result.get("results", [])),
                   "metrics": {name: {**retrieval_scores(case["relevant_chunk_ids"], ids, 3),
                                      "hit_at_1": float(bool(ids) and ids[0] in case["relevant_chunk_ids"])}
                               for name, ids in ranking.items()},
                   "corpus_size": len(corpus),
                   "candidate_recall": {name: len(set(case["relevant_chunk_ids"]) & {h.chunk.chunk_id for h in hits}) / len(case["relevant_chunk_ids"])
                                        if case["relevant_chunk_ids"] else None
                                        for name, hits in (("dense", dense), ("bm25", sparse), ("rrf", capture.get("rerank_candidates", [])))},
                   "rerank_candidates": [{"chunk_id": h.chunk.chunk_id, "rrf_score": h.rrf_score,
                                          "dense_score": h.dense_score, "bm25_score": h.bm25_score,
                                          "rerank_score": h.rerank_score} for h in capture.get("reranked", [])],
                   "tool_result": result}
            rows.append(row)
            if corpus:
                report["metadata"]["corpus_sha256"] = hashlib.sha256(json.dumps(
                    [asdict(c) for c in sorted(corpus, key=lambda c: c.chunk_id)], ensure_ascii=False, sort_keys=True
                ).encode()).hexdigest()
            report["summary"] = summarize(rows)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{index + 1}/{len(cases)} {case['id']} success={row['success']} rerank={row['rerank_verified']} elapsed_ms={elapsed:.0f}", flush=True)
            if result.get("error_code") in {"UNAVAILABLE", "INDEX_MISSING"}:
                raise RuntimeError("retrieval infrastructure unavailable; stopped remaining live calls")
    finally:
        retriever.backend.retrieval_snapshot = original_snapshot
        retriever._reranker = actual_reranker
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if all(r["rerank_verified"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
