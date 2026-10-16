"""2026-10-10 engineering baseline: isolated pytest workers, never model accuracy."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from app.evaluation.statistics import percentile, rate

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "react": ["tests/test_prompt_contracts.py", "tests/test_context_budget.py", "tests/test_llm_backpressure.py"],
    "rag_wiki": ["tests/test_rag_policy_evidence.py", "tests/test_qdrant_backend.py", "tests/test_workflow_wiki.py", "tests/test_wiki_runtime.py", "tests/test_hybrid_rag.py"],
    "mcp": ["tests/test_mcp_requirements.py"],
    "memory": ["tests/test_memory_requirements.py", "tests/test_memory_quality.py"],
    "multi_agent": ["tests/test_multi_agent_requirements.py", "tests/test_return_workflow.py", "tests/test_fulfillment_agent.py", "tests/test_agent_quality.py"],
    "isolation_evaluation": ["tests/test_comprehensive_review.py", "tests/test_concurrency.py", "tests/test_evaluation_upgrade.py"],
    "frontend": ["tests/test_frontend.py", "tests/test_quality_dashboard.py"],
}


def dependency_versions():
    """2026-10-11: missing distribution metadata must not discard test results."""
    versions = {}
    for name in ("pytest", "openai", "mcp", "httpx", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def read_junit(path):
    root = ET.parse(path).getroot()
    rows = []
    for case in root.iter("testcase"):
        status = "failed" if case.find("failure") is not None else "error" if case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed"
        rows.append({"id": f'{case.get("classname", "")}.{case.get("name", "")}',
                     "status": status, "duration_ms": float(case.get("time", "0")) * 1000})
    return rows


def run_suite(name, paths, directory, timeout):
    directory.mkdir(parents=True)
    keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
            "LOCALAPPDATA", "APPDATA", "PATHEXT", "COMSPEC", "LANG", "LC_ALL"}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update({"ECOM_EVAL_WORKER": "1", "OPENAI_API_KEY": "offline-placeholder", "EMBEDDING_API_KEY": "offline-placeholder",
                "OPENAI_BASE_URL": "http://127.0.0.1:1/v1", "EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1",
                "MCP_ENABLED": "false", "MEMORY_ENABLED": "false", "KNOWLEDGE_MODE": "rag",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                "DATABASE_URL": "sqlite:///" + (directory / "products.db").as_posix(),
                "MEMORY_DIR": str(directory / "memory"), "CONCURRENT_SESSION_DIR": str(directory / "sessions"),
                "CHROMA_PERSIST_DIR": str(directory / "chroma"), "WIKI_STORE_DIR": str(directory / "wiki"),
                "KB_INDEX_PATH": str(directory / "index.json"),
                "SESSION_PATH": str(directory / "session.json")})
    for key in ("ORDER_DATABASE_PATH", "AUTH_DATABASE_PATH", "MEMORY_CONSENT_DB_PATH"):
        env[key] = str(directory / (key.lower() + ".db"))
    shutil.copytree(ROOT / "app/agent/rag/knowledge", directory / "knowledge")
    env["KB_DIR"] = str(directory / "knowledge")
    junit = directory / "junit.xml"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-p", "app.evaluation.offline_guard", "--basetemp", str(directory / "tmp"),
               "--junitxml", str(junit), *paths]
    started, status = time.perf_counter(), "completed"
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code, status = None, "timeout"
    rows = read_junit(junit) if junit.exists() else []
    if exit_code != 0 and status == "completed":
        status = "failed"
    passed = sum(row["status"] == "passed" for row in rows)
    return {"name": name, "status": status, "exit_code": exit_code,
            "wall_ms": (time.perf_counter() - started) * 1000,
            "tests": len(rows), "passed": passed,
            "failed": sum(row["status"] in {"failed", "error"} for row in rows),
            "skipped": sum(row["status"] == "skipped" for row in rows),
            "pass_rate": rate(passed, len(rows)),
            "test_duration_ms": {"p50": percentile([r["duration_ms"] for r in rows], 50),
                                 "p95": percentile([r["duration_ms"] for r in rows], 95)},
            "cases": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/engineering-latest.json")
    parser.add_argument("--suite", choices=list(SUITES), action="append")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("timeout must be positive")
    selected = list(dict.fromkeys(args.suite or SUITES))
    dependencies = dependency_versions()
    missing = [name for name, version in dependencies.items() if version is None]
    if missing:
        print(f"Warning: missing package metadata: {', '.join(missing)}. "
              f"Interpreter: {sys.executable}. Test results will still be saved.", flush=True)
    run_root = Path(tempfile.mkdtemp(prefix="engineering-"))
    suites = []
    for name in selected:
        print(f"Running {name}...", flush=True)
        suites.append(run_suite(name, SUITES[name], run_root / name, args.timeout))
    files = sorted({path for name in selected for path in SUITES[name]})
    test_hash = hashlib.sha256(b"".join(path.encode() + (ROOT / path).read_bytes() for path in files)).hexdigest()
    code_files = sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "app/server/static").glob("*.html"))
    code_hash = hashlib.sha256(b"".join(path.relative_to(ROOT).as_posix().encode() + path.read_bytes() for path in code_files)).hexdigest()
    report = {"schema_version": 1, "report_type": "engineering_contract", "upgrade_date": "2026-10-10",
              "generated_at": datetime.now(timezone.utc).isoformat(), "test_sha256": test_hash,
              "code_sha256": code_hash, "python": sys.version.split()[0],
              "python_executable": sys.executable,
              "dependencies": dependencies,
              "missing_dependency_metadata": missing,
              "isolation": {"worker_process": True, "temporary_storage": True,
                            "dotenv_disabled": True, "environment_allowlist": True,
                            "external_http_guard": "httpx/httpx2; loopback test servers allowed",
                            "os_security_sandbox": False},
              "summary": {"tests": sum(s["tests"] for s in suites), "passed": sum(s["passed"] for s in suites),
                          "gate_passed": all(s["status"] == "completed" and s["tests"] > 0 and s["skipped"] == 0 for s in suites)},
              "unmeasured": ["真实模型任务成功率", "真实检索 Recall@K / MRR / nDCG", "线上请求 P95", "真实模型费用"],
              "suites": suites}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output.resolve()), "worker_artifacts": str(run_root), **report["summary"]}, ensure_ascii=False))
    return 0 if report["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
