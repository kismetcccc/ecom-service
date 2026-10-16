"""MR-19 Memory 专项质量门禁。"""

from app.evaluation.memory_quality import (
    conflict_accuracy,
    evaluate_dataset,
    extraction_scores,
    retrieval_recall,
)


def test_memory_quality_dataset_passes_all_thresholds():
    report = evaluate_dataset()
    assert report["p0_passed"] is True
    assert report["passed"] is True
    assert all(report["threshold_checks"].values())


def test_memory_quality_metrics_detect_regressions():
    extraction = extraction_scores([{
        "expected": ["name", "color"],
        "actual": ["name", "wrong"],
    }])
    assert extraction == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert conflict_accuracy([{
        "expected_active": ["new"],
        "actual_active": ["old", "new"],
        "obsolete": ["old"],
    }]) == 0.0
    assert retrieval_recall([{
        "relevant_ids": ["a", "b"],
        "retrieved_ids": ["a", "x"],
        "top_k": 2,
    }]) == 0.5


def test_empty_memory_dataset_is_not_a_perfect_gate(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{}')
    report = evaluate_dataset(path)
    assert not report["passed"]
    assert report["report_type"] == "memory_fixture_contract"
    assert not any(report["sample_counts"].values())
