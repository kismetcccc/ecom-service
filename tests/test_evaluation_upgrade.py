"""2026-10-10 regression: metrics must not turn missing evidence into success."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.evaluation.evaluator import Evaluator, EvalResult
from app.evaluation.sandbox import Sandbox
from app.evaluation.statistics import rate, percentile, retrieval_scores
from app.evaluation.trace import RunTrace
from app.evaluation import metrics
from app.schemas.response import CustomerServiceResponse, IntentType


def test_statistics_denominators_and_small_sample_uncertainty():
    assert rate(0, 0)["value"] is None
    result = rate(3, 3)
    assert result["value"] == 1
    assert 0 < result["wilson95"][0] < 1
    assert percentile([], 95) is None
    assert percentile([1, 2, 100], 95) == 100


def test_retrieval_duplicate_hits_cannot_inflate_scores():
    scores = retrieval_scores(["a", "b"], ["a", "a", "x"], 3)
    assert scores["recall_at_k"] == .5
    assert scores["mrr_at_k"] == 1
    assert scores["ndcg_at_k"] < 1
    assert retrieval_scores([], [], 3)["recall_at_k"] is None


def test_no_assertions_and_no_response_cannot_pass():
    evaluator = Evaluator(None, None, "fake", use_judge=False)
    assert not evaluator._decide_pass(EvalResult("x", "x"))
    evaluator.sandbox = NS(run=lambda _: RunTrace("x", ["hi"]))
    assert evaluator.run_case(EvalCase("x", "x", ["hi"])).error == "MissingFinalResponse"


def test_failed_llm_call_is_counted_with_unknown_usage(tmp_path):
    sandbox = Sandbox(tmp_root=str(tmp_path))
    trace = RunTrace("x", ["hi"])
    def fail(**kwargs):
        raise TimeoutError("test")
    with pytest.raises(TimeoutError):
        sandbox._wrap_create(fail, trace)(model="fake")
    assert trace.num_llm_calls == 1
    assert not trace.llm_calls[0].success
    assert not trace.llm_calls[0].usage_available
    assert trace.to_dict()["llm_failed_calls"] == 1
    assert trace.llm_calls[0].latency_ms >= 0


def test_partial_tokens_not_reported_as_complete_budget(tmp_path):
    sandbox = Sandbox(tmp_root=str(tmp_path))
    trace = RunTrace("x", ["hi"], final_response=CustomerServiceResponse(
        intent=IntentType.OTHER, confidence=1, reply="hi"))
    sandbox._record_llm_call(trace, NS(usage=None, choices=[]), 5, "react")
    evaluator = Evaluator(NS(run=lambda _: trace), None, "fake", use_judge=False)
    result = evaluator.run_case(EvalCase("x", "x", ["hi"], max_tokens=100, expected_keywords=["hi"]))
    assert not result.passed
    assert result.token_pass is None
    report = evaluator._aggregate([result])
    assert report["summary"]["usage_coverage"]["value"] == 0
    assert report["summary"]["token_measurement"] == "partial_lower_bound"


def test_sandbox_restores_settings_on_constructor_failure_and_reuses_safely(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_root=str(tmp_path))
    previous = settings.memory_enabled, settings.mcp_enabled, settings.memory_dir
    seen = []
    def fail(path):
        seen.append(path)
        assert not settings.memory_enabled and not settings.mcp_enabled
        assert Path(settings.memory_dir).is_relative_to(tmp_path)
        raise RuntimeError("injected")
    monkeypatch.setattr(sandbox, "_build_agent", fail)
    for _ in range(2):
        trace = sandbox.run(EvalCase("../../escape", "x", ["hi"]))
        assert trace.error.startswith("RuntimeError")
        assert (settings.memory_enabled, settings.mcp_enabled, settings.memory_dir) == previous
    assert seen[0] != seen[1]
    assert all(Path(path).is_relative_to(tmp_path) for path in seen)


def test_sandbox_default_tools_never_call_real_executor(tmp_path):
    sandbox = Sandbox(tmp_root=str(tmp_path))
    called = []
    trace = RunTrace("x", [])
    result = sandbox._wrap_execute_tool(lambda *args: called.append(args), trace)("apply_refund", {})
    assert not called
    assert json.loads(result)["error_code"] == "SANDBOX_TOOL_UNCONFIGURED"


@pytest.mark.parametrize("payload", ['{"faithful":"false"}', '{"faithful":1}', '{}'])
def test_faithfulness_judge_rejects_non_boolean(payload):
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(choices=[NS(message=NS(content=payload))]))))
    score, reason = metrics.judge_faithfulness(client, "fake", "answer", [])
    assert score == 0 and "失败" in reason


@pytest.mark.parametrize("payload", ['{"score":6}', '{"score":true}', '{"score":"5"}'])
def test_quality_judge_rejects_invalid_score(payload):
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(choices=[NS(message=NS(content=payload))]))))
    assert metrics.judge_answer_quality(client, "fake", "q", "a")[0] == 0


@pytest.mark.parametrize("result", ['broken', '{"success":false,"error_code":"UNAVAILABLE"}', '{"success":"true"}'])
def test_single_react_failure_never_executes_following_write(result):
    from app.agent.chat import EcomAgent
    agent = object.__new__(EcomAgent)
    agent.max_react_steps = 1
    agent.model, agent.temperature = "fake", 0
    agent.context_budget = NS(reserved_output_tokens=100)
    agent.raw_messages = []
    agent._prepare_messages = lambda: []
    agent._print_action = lambda *args: None
    agent._print_observation = lambda *args: None
    calls = [NS(id=str(i), function=NS(name=name, arguments="{}"))
             for i, name in enumerate(["query_order", "apply_refund"])]
    agent.client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(choices=[NS(message=NS(content=None, tool_calls=calls))]))))
    executed = []
    agent.tool_manager = NS(tool_definitions=[], execute_tool=lambda name, args: executed.append(name) or result)
    assert "核实" in agent._react_loop()
    assert executed == ["query_order"]
    assert json.loads(agent.raw_messages[-1]["content"])["error_code"] == "DEPENDENCY_BLOCKED"


def test_exact_fixture_miss_never_invents_order(tmp_path):
    from app.evaluation.fixtures import load_tool_fixtures
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps([{"name": "query_order", "arguments": {"order_id": "A"},
                                "result": {"success": True, "order": {"order_id": "A"}}}]))
    execute = load_tool_fixtures(path)
    assert json.loads(execute("query_order", {"order_id": "A"}))["success"]
    assert json.loads(execute("query_order", {"order_id": "B"}))["error_code"] == "FIXTURE_MISS"


def test_junit_failures_and_skips_are_not_success(tmp_path):
    from app.evaluation.benchmark import read_junit
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite><testcase name="ok" time="0.1"/><testcase name="skip"><skipped/></testcase><testcase name="bad"><failure/></testcase></testsuite></testsuites>')
    assert [row["status"] for row in read_junit(path)] == ["passed", "skipped", "failed"]


def test_offline_guard_covers_installed_http_clients(monkeypatch):
    import importlib
    from app.evaluation.offline_guard import pytest_configure
    modules = []
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        monkeypatch.setattr(module.HTTPTransport, "handle_request", module.HTTPTransport.handle_request)
        monkeypatch.setattr(module.AsyncHTTPTransport, "handle_async_request", module.AsyncHTTPTransport.handle_async_request)
        modules.append(module)
    pytest_configure(None)
    for module in modules:
        with module.Client() as client, pytest.raises(RuntimeError, match="OFFLINE_EVALUATION"):
            client.get("https://example.invalid")


def test_retrieval_labels_are_scored_against_observed_last_query():
    from app.evaluation.trace import ToolObservation
    trace = RunTrace("x", ["policy"], final_response=CustomerServiceResponse(intent=IntentType.OTHER, confidence=1, reply="ok"))
    trace.tool_observations = [ToolObservation("search_knowledge", {}, json.dumps({"success": True, "results": [{"chunk_id": "irrelevant"}, {"chunk_id": "a"}]}))]
    evaluator = Evaluator(NS(run=lambda _: trace), None, "fake", use_judge=False)
    case = EvalCase("x", "x", ["policy"], relevant_chunk_ids=["a", "b"], retrieval_k=2)
    result = evaluator.run_case(case)
    assert result.recall_at_k == .5 and result.mrr_at_k == .5
    assert not result.passed
    assert evaluator._aggregate([result])["summary"]["metric_sample_counts"]["recall_at_k"] == 1


def test_eval_cli_requires_explicit_live_without_constructing_client(monkeypatch):
    from app.scripts import run_eval
    monkeypatch.setattr('sys.argv', ['run_eval'])
    def forbidden(**kwargs):
        raise AssertionError("must not construct model client")
    monkeypatch.setattr(run_eval, 'OpenAI', forbidden)
    run_eval.main()


def test_live_cli_failed_gate_returns_nonzero_and_closes_client(tmp_path, monkeypatch):
    from app.scripts import run_eval
    dataset, fixtures, output = [tmp_path / name for name in ('cases.json', 'fixtures.json', 'report.json')]
    dataset.write_text(json.dumps({'cases': [{'id': 'x', 'description': 'x', 'turns': ['hi']}]}))
    fixtures.write_text('[]')
    monkeypatch.setattr('sys.argv', ['run_eval', '--live', '--no-judge', '--dataset', str(dataset), '--tool-fixtures', str(fixtures), '--output', str(output)])
    closed = []
    monkeypatch.setattr(run_eval, 'OpenAI', lambda **kwargs: NS(close=lambda: closed.append(True)))
    monkeypatch.setattr(run_eval, 'Sandbox', lambda **kwargs: None)
    monkeypatch.setattr(run_eval, 'Evaluator', lambda **kwargs: NS(run_all=lambda cases: {'summary': {'total': 1, 'passed': 0}, 'cases': []}))
    monkeypatch.setattr(run_eval, '_print_report', lambda report: None)
    assert run_eval.main() == 1
    assert closed == [True]
    assert json.loads(output.read_text(encoding='utf-8'))['metadata']['report_type'] == 'live_model_with_fixture_tools'


@pytest.mark.parametrize('status,exit_code', [('completed', 0), ('failed', 1)])
def test_benchmark_saves_report_when_mcp_metadata_is_missing(tmp_path, monkeypatch, status, exit_code):
    from app.evaluation import benchmark
    original_version = benchmark.importlib.metadata.version
    def version(name):
        if name == 'mcp':
            raise benchmark.importlib.metadata.PackageNotFoundError(name)
        return original_version(name)
    monkeypatch.setattr(benchmark.importlib.metadata, 'version', version)
    monkeypatch.setattr(benchmark.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path / 'workers'))
    monkeypatch.setattr(benchmark, 'run_suite', lambda *args: {
        'name': 'mcp', 'status': status, 'tests': 1,
        'passed': int(status == 'completed'), 'skipped': 0,
    })
    output = tmp_path / 'report.json'
    assert benchmark.main(['--suite', 'mcp', '--output', str(output)]) == exit_code
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['dependencies']['mcp'] is None
    assert report['dependencies']['pytest']
    assert report['missing_dependency_metadata'] == ['mcp']
    assert report['python_executable']
    assert report['summary']['gate_passed'] is (status == 'completed')
