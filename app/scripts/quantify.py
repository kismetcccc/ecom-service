"""Reproducible measurements. Run --help; no network calls without --live."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

from app.evaluation.statistics import percentile, rate, retrieval_scores

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("performance", "quality", "scale", "cost", "stability")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(category, key, label, value, unit, samples, scope):
    return dict(category=category, key=key, label=label, value=value, unit=unit,
                samples=samples, scope=scope)


def distribution(rows):
    return {"count": len(rows), **{f"p{p}": percentile(rows, p) for p in (50, 95, 99)}}


def inventory():
    from app.config.settings import settings
    from app.agent.rag.chunker import chunk_markdown_dir
    kb = Path(settings.kb_dir)
    chunks = chunk_markdown_dir(kb)
    files = sorted(kb.glob("*.md"))
    metrics = [
        metric("scale", "documents", "知识文档", len(files), "count", len(files), "本地知识源 Markdown 文件"),
        metric("scale", "chunks", "知识切片", len(chunks), "count", len(files), "当前源码切分结果，不代表已部署索引"),
        metric("scale", "skills", "Skill 数量", len(list((ROOT / "app/agent/skills/definitions").glob("*/SKILL.md"))), "count", None, "本地 Skill 包"),
    ]
    path = Path(settings.kb_index_path)
    extra = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        vectors = data["vectors"]
        dimensions = sorted({len(v) for v in vectors})
        extra = {"numpy_index_sha256": digest(path), "numpy_embedding_model": data.get("embedding_model"),
                 "numpy_index_matches_source": data["chunks"] == [c.to_dict() for c in chunks]}
        metrics += [metric("scale", "numpy_vectors", "本地备选索引向量", len(vectors), "count", len(vectors), "NumPy 索引；不代表当前 Qdrant 集合"),
                    metric("scale", "vector_dimension", "本地向量维度", dimensions[0] if len(dimensions) == 1 else None, "count", len(vectors), "从 NumPy 索引实际读取")]
    return metrics, [], {"configured_backend": settings.rag_backend,
                        "knowledge_sha256": hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files + [kb / "policies.json"])).hexdigest(), **extra}


def retrieval(args):
    from app.config.settings import settings
    from app.agent.rag.chunker import chunk_markdown_dir
    from app.agent.tools.knowledge import _create_backend
    from openai import OpenAI
    import httpx
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    source_ids = {c.chunk_id for c in chunk_markdown_dir(Path(settings.kb_dir))}
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("dataset must contain unique nonempty cases")
    for c in cases:
        if not c["relevant_chunk_ids"] or not set(c["relevant_chunk_ids"]) <= source_ids:
            raise ValueError(f"invalid or stale labels: {c['id']}")
    backend_name = args.backend or settings.rag_backend
    backend = _create_backend(backend_name)
    backend.load()
    if backend.expected_embedding_model() != settings.embedding_model:
        raise ValueError("embedding model mismatch; build a compatible index first")
    rows = []
    # Explicit timeout, no hidden retries, same configured proxy policy as chat.
    with OpenAI(api_key=settings.embedding_api_key or settings.openai_api_key,
                base_url=settings.embedding_base_url or settings.openai_base_url,
                timeout=args.timeout, max_retries=0,
                http_client=httpx.Client(trust_env=settings.openai_trust_env)) as client:
        for c in cases:
            started = time.perf_counter()
            ids, error, tokens = [], None, None
            try:
                response = client.embeddings.create(model=settings.embedding_model, input=[c["query"]])
                tokens = response.usage.total_tokens if response.usage else None
                hits = backend.search(response.data[0].embedding, top_k=args.k)
                ids = [h.chunk.chunk_id for h in hits]
            except Exception as exc:
                error = type(exc).__name__  # never persist credentials or upstream bodies
            rows.append(dict(id=c["id"], query=c["query"], relevant=c["relevant_chunk_ids"], retrieved=ids,
                             error=error, tokens=tokens, elapsed_ms=(time.perf_counter()-started)*1000,
                             **retrieval_scores(c["relevant_chunk_ids"], ids, args.k)))
    metrics = [metric("quality", key, f"{label}@{args.k}", sum(r[key] for r in rows)/len(rows), "ratio", len(rows),
                      "开发集宏平均；失败查询按 0 分；原始向量检索，不含回答生成及政策过滤")
               for key, label in (("recall_at_k", "Recall"), ("mrr_at_k", "MRR"), ("ndcg_at_k", "nDCG"))]
    for p in (50, 95, 99):
        metrics.append(metric("performance", f"retrieval_p{p}", f"检索 P{p}", percentile([r["elapsed_ms"] for r in rows], p), "ms", len(rows), "串行 Embedding API + 向量搜索；包含失败；非聊天延迟"))
    metrics.append(metric("stability", "retrieval_errors", "检索错误率", sum(r["error"] is not None for r in rows)/len(rows), "ratio", len(rows), "本次评测窗口"))
    known = [r["tokens"] for r in rows if r["tokens"] is not None]
    metrics.append(metric("cost", "embedding_tokens", "检索 Embedding Token", sum(known) if known else None, "tokens", len(known), "仅查询向量化 usage；不含索引构建和聊天"))
    metrics.append(metric("cost", "embedding_usage_coverage", "Embedding 用量采集率", len(known)/len(rows), "ratio", len(rows), "有 usage 的查询数 / 全部查询数"))
    return metrics, rows, {"backend": backend_name, "embedding_model": settings.embedding_model,
                           "index_vectors": backend.size(), "dataset_sha256": digest(args.dataset),
                           "index_corpus_version": getattr(backend, "_corpus_version", None),
                           "index_sha256": digest(settings.kb_index_path) if backend_name == "numpy" else None,
                           "label_status": dataset.get("label_status"), "k": args.k,
                           "error_rate": rate(sum(r["error"] is not None for r in rows), len(rows))}


async def http_benchmark(args):
    import httpx
    from app.schemas.response import CustomerServiceResponse
    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(f"set {args.token_env} to a test account bearer token")
    rows, run_id = [], uuid4().hex[:12]
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False,
                                headers={"Authorization": f"Bearer {token}"}) as client:
        # Authentication failures should not be reported as successful high throughput.
        identity = await client.get(args.url.rstrip("/") + "/v1/auth/me")
        identity.raise_for_status()
        if identity.json().get("user_id") != args.user_id:
            raise ValueError("token identity does not match --user-id")
        async def request(index):
            session = f"quant-{run_id}-{index}"
            started = time.perf_counter()
            status, error, review = None, None, None
            try:
                response = await client.post(args.url.rstrip("/") + "/v1/chat", json={
                    "user_id": args.user_id, "session_id": session, "message": args.message})
                status = response.status_code
                response.raise_for_status()
                body = response.json()
                result = CustomerServiceResponse.model_validate(body["result"])
                if body.get("user_id") != args.user_id or body.get("session_id") != session or not result.reply.strip():
                    raise ValueError("invalid chat response")
                review = result.quality_review.status if result.quality_review else None
            except Exception as exc:
                error = type(exc).__name__
            return dict(id=index, status=status, error=error, review=review,
                        elapsed_ms=(time.perf_counter()-started)*1000)
        for i in range(args.warmup):
            await request(f"warmup-{i}")
        queue = iter(range(args.requests))
        async def worker():
            for i in queue:
                rows.append(await request(i))
        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started
    ok = [r for r in rows if r["error"] is None]
    metrics = [metric("performance", "rps", "完成请求 RPS", len(rows)/elapsed, "rps", len(rows), "所有结果 / 测量墙钟时间；闭环固定并发"),
               metric("performance", "successful_rps", "有效响应 RPS", len(ok)/elapsed, "rps", len(rows), "HTTP 成功且结构有效；不代表语义正确"),
               metric("stability", "request_errors", "请求错误率", (len(rows)-len(ok))/len(rows), "ratio", len(rows), "HTTP、连接、超时和结构错误"),
               metric("performance", "ttft", "首字延迟 TTFT", None, "ms", 0, "当前 /v1/chat 非流式，无法测量首 token")]
    for p in (50, 95, 99):
        metrics.append(metric("performance", f"chat_p{p}", f"有效响应 P{p}", percentile([r["elapsed_ms"] for r in ok], p), "ms", len(ok), "客户端发送至完整响应；不含客户端排队；服务端等待包含在内"))
    return metrics, rows, {"requests": args.requests, "concurrency": args.concurrency, "warmup": args.warmup,
                          "wall_seconds": elapsed, "all_request_latency_ms": distribution([r["elapsed_ms"] for r in rows]),
                          "status_counts": dict(Counter(str(r["status"]) for r in rows)),
                          "review_status_counts": dict(Counter(str(r["review"]) for r in rows)),
                          "server_revision": None,
                          "model_execution": "未采集服务端调用轨迹；问候可能走规则快捷回复，不能当作模型推理吞吐",
                          "success_rate": rate(len(ok), len(rows)), "message": args.message,
                          "scope": "单测试账户、每请求独立会话；不代表多用户生产容量；会留下测试会话"}


def bundle(paths):
    """Package evidence without recomputing or blending unrelated populations."""
    metrics, rows, sources = [], [], []
    for i, filename in enumerate(paths):
        d = json.loads(Path(filename).read_text(encoding="utf-8"))
        source = {"file": Path(filename).name, "sha256": digest(filename),
                  "generated_at": d.get("generated_at", d.get("metadata", {}).get("generated_at")),
                  "metadata": d.get("metadata", {})}
        if d.get("report_type") == "quantitative_metrics" and d.get("schema_version") == 1:
            source["kind"] = d.get("category")
            items = d["metrics"]
            cases = d["cases"]
        elif d.get("report_type") == "engineering_contract":
            cases = [c for s in d["suites"] for c in s["cases"]]
            passed = sum(c["status"] == "passed" for c in cases)
            items = [metric("stability", "engineering_pass_rate", "工程回归通过率", passed/len(cases) if cases else None, "ratio", len(cases), "离线工程契约；不是模型回答准确率")]
            source.update(kind="engineering", code_sha256=d.get("code_sha256"), test_sha256=d.get("test_sha256"),
                          gate_passed=d["summary"].get("gate_passed"))
        elif d.get("metadata", {}).get("report_type") == "live_model_with_fixture_tools":
            cases, s = d["cases"], d["summary"]
            n = len(cases)
            passed = sum(c["passed"] for c in cases)
            items = [metric("quality", "task_pass_rate", "模型用例通过率", passed/n if n else None, "ratio", n, "真实模型 + fixture 工具；规则/裁判阈值通过，非人工准确率"),
                     metric("quality", "bad_case_rate", "未通过用例占比", 1-passed/n if n else None, "ratio", n, "包括规则未达标、运行与评分异常"),
                     metric("cost", "agent_tokens", "Agent 已观测 Token", s["total_tokens"], "tokens", n, "不含 Judge；缺失 usage 时为下界"),
                     metric("cost", "avg_agent_tokens", "单用例 Token", s["avg_tokens_per_case"], "tokens", n, "可能包含多轮；不含 Judge"),
                     metric("cost", "usage_coverage", "Agent 用量采集率", s["usage_coverage"]["value"], "ratio", s["usage_coverage"]["denominator"], "有 usage 的模型调用 / 全部调用")]
            source["kind"] = "live"
        else:
            raise ValueError(f"unsupported report: {filename}")
        metrics.extend({**m, "key": f"source{i}:{m['key']}"} for m in items)
        rows.extend({**c, "source": source["file"]} for c in cases)
        sources.append(source)
    return metrics, rows, {"sources": sources, "scope": "多份独立测量并列展示；时间窗口和样本不混合"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "retrieval", "http", "bundle"):
        p = sub.add_parser(name)
        p.add_argument("--output", default=f"reports/quant-{name}.json")
        if name == "bundle":
            p.add_argument("--report", action="append", required=True)
        if name in ("retrieval", "http"):
            p.add_argument("--live", action="store_true", help="真实 API 调用，可能消耗模型额度")
            p.add_argument("--timeout", type=float, default=30)
        if name == "retrieval":
            p.add_argument("--dataset", default="app/evaluation/retrieval_cases.json")
            p.add_argument("--backend", choices=["numpy", "chroma", "qdrant"])
            p.add_argument("--k", type=int, default=3)
        if name == "http":
            p.add_argument("--url", default="http://127.0.0.1:8000")
            p.add_argument("--token-env", default="ECOM_BENCH_TOKEN")
            p.add_argument("--user-id", required=True)
            p.add_argument("--requests", type=int, default=10)
            p.add_argument("--concurrency", type=int, default=1)
            p.add_argument("--warmup", type=int, default=0)
            p.add_argument("--message", default="你好，请简单介绍你可以帮助我做什么")
    args = parser.parse_args(argv)
    if args.command in ("retrieval", "http") and not args.live:
        parser.error("真实评测需要 --live")
    if any(getattr(args, k, 1) <= 0 for k in ("requests", "concurrency", "k", "timeout")) or getattr(args, "warmup", 0) < 0:
        parser.error("sample count, concurrency, k, timeout must be positive; warmup >= 0")
    functions = {"inventory": lambda: inventory(), "retrieval": lambda: retrieval(args),
                 "http": lambda: asyncio.run(http_benchmark(args)), "bundle": lambda: bundle(args.report)}
    metrics, rows, metadata = functions[args.command]()
    code_paths = sorted((ROOT / "app").rglob("*.py"))
    report = dict(schema_version=1, report_type="quantitative_metrics", category=args.command,
                  generated_at=datetime.now(timezone.utc).isoformat(), metrics=metrics, cases=rows,
                  metadata={"python": platform.python_version(), "platform": platform.platform(),
                            "runner_code_sha256": hashlib.sha256(b"".join(p.relative_to(ROOT).as_posix().encode()+p.read_bytes() for p in code_paths)).hexdigest(),
                            **metadata},
                  unmeasured=["生产 DAU / 月度可用性", "人工节省工时 / GPU 占用", "未配置价格时的实际货币成本"],
                  notes=["null 表示未测；小样本 P95/P99 仅供观察，不作容量承诺", "不同口径报告不可直接宣称性能提升"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path.resolve()), "samples": len(rows), "metrics": len(metrics)}, ensure_ascii=False))
    return 1 if any(r.get("error") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
