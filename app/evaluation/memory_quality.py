"""MR-19：Memory 专项质量指标、数据集运行器和 CI 门禁。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from app.agent.memory.privacy import filter_memory_text


DEFAULT_DATASET = Path(__file__).with_name("memory_cases.json")


def _ratio(numerator: int, denominator: int, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def extraction_scores(cases: Iterable[dict]) -> dict:
    """计算事实抽取的微平均 precision、recall 和 F1。"""
    true_positive = predicted = expected = 0
    for case in cases:
        expected_set = {str(item).casefold() for item in case["expected"]}
        actual_set = {str(item).casefold() for item in case["actual"]}
        true_positive += len(expected_set & actual_set)
        predicted += len(actual_set)
        expected += len(expected_set)
    precision = _ratio(true_positive, predicted, 1.0)
    recall = _ratio(true_positive, expected, 1.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_ratio(2 * precision * recall, precision + recall, 1.0), 4),
    }


def conflict_accuracy(cases: Iterable[dict]) -> float:
    results = [
        set(case["expected_active"]) == set(case["actual_active"])
        and not (set(case.get("obsolete", [])) & set(case["actual_active"]))
        for case in cases
    ]
    return round(_ratio(sum(results), len(results), 1.0), 4)


def retrieval_recall(cases: Iterable[dict]) -> float:
    scores = []
    for case in cases:
        relevant = set(case["relevant_ids"])
        retrieved = set(case["retrieved_ids"][:case.get("top_k", 5)])
        scores.append(_ratio(len(relevant & retrieved), len(relevant), 1.0))
    return round(_ratio(sum(scores), len(scores), 1.0), 4)


def unsafe_write_rate(cases: Iterable[dict]) -> float:
    cases = list(cases)
    written = sum(filter_memory_text(case["text"]) is not None for case in cases)
    return round(_ratio(written, len(cases), 0.0), 4)


def evaluate_dataset(path: str | Path = DEFAULT_DATASET) -> dict:
    """运行离线 Memory 数据集；P0 隐私、注入和纠正指标必须满分。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sample_counts = {name: len(data.get(name, [])) for name in (
        "extraction_cases", "conflict_cases", "retrieval_cases", "sensitive_cases", "injection_cases")}
    complete = all(sample_counts.values())
    extraction = extraction_scores(data.get("extraction_cases", []))
    conflict = conflict_accuracy(data.get("conflict_cases", []))
    retrieval = retrieval_recall(data.get("retrieval_cases", []))
    sensitive_rate = unsafe_write_rate(data.get("sensitive_cases", []))
    injection_rate = unsafe_write_rate(data.get("injection_cases", []))
    metrics = {
        "fact_extraction_precision": extraction["precision"],
        "fact_extraction_recall": extraction["recall"],
        "fact_extraction_f1": extraction["f1"],
        "conflict_update_accuracy": conflict,
        "relevant_memory_recall": retrieval,
        "sensitive_write_rate": sensitive_rate,
        "prompt_injection_write_rate": injection_rate,
    }
    thresholds = data.get("thresholds", {})
    checks = {
        name: (
            value <= thresholds[name]
            if name.endswith("_rate")
            else value >= thresholds[name]
        )
        for name, value in metrics.items()
        if name in thresholds
    }
    p0_passed = (
        complete and conflict == 1.0
        and sensitive_rate == 0.0
        and injection_rate == 0.0
    )
    return {
        "dataset_version": data.get("version", 1),
        "report_type": "memory_fixture_contract",
        "sample_counts": sample_counts,
        "measurement_sources": {
            "extraction_conflict_retrieval": "dataset-provided actual values; NOT a live model measurement",
            "sensitive_injection": "executed filter_memory_text on labeled strings",
        },
        "metrics": metrics,
        "threshold_checks": checks,
        "p0_passed": p0_passed,
        "passed": p0_passed and all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Memory 专项质量评估")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = evaluate_dataset(args.dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
