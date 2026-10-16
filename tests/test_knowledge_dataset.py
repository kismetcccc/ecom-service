"""Offline integrity checks; not claims about model accuracy."""
import hashlib
import json
import shutil

import pytest

from app.evaluation import knowledge_dataset as dataset


def test_committed_dataset_is_valid_and_has_all_answerability_classes():
    report = dataset.validate()
    assert report["source_chapters_covered"] == 19
    rows = dataset.read_jsonl(dataset.DATA / "qa_cases.jsonl")
    assert {row["answerability"] for row in rows} == dataset.ANSWERABILITIES
    assert all(len(row["required_evidence_groups"]) > 1 for row in rows
               if row["category"] == "cross_source")


@pytest.fixture
def isolated_dataset(tmp_path, monkeypatch):
    destination = tmp_path / "data"
    shutil.copytree(dataset.DATA, destination)
    monkeypatch.setattr(dataset, "DATA", destination)
    return destination


def reseal(path):
    """Simulate an editor updating checksums, leaving semantic invariants to validator."""
    manifest_path = dataset.DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset.dump(manifest_path, manifest)


def test_validator_rejects_paraphrase_split_leakage(isolated_dataset):
    path = isolated_dataset / "qa_cases.jsonl"
    rows = dataset.read_jsonl(path)
    rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
    dataset.jsonl(path, rows)
    reseal(path)
    with pytest.raises(AssertionError, match="Family leakage"):
        dataset.validate()


def test_validator_rejects_nonexistent_evidence(isolated_dataset):
    path = isolated_dataset / "qa_cases.jsonl"
    rows = dataset.read_jsonl(path)
    rows[0]["required_evidence_groups"] = [["fabricated-source"]]
    dataset.jsonl(path, rows)
    reseal(path)
    with pytest.raises(AssertionError):
        dataset.validate()


def test_validator_rejects_source_metadata_drift(isolated_dataset, tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    shutil.copytree(dataset.KB, knowledge)
    catalog = knowledge / "policies.json"
    values = json.loads(catalog.read_text(encoding="utf-8"))
    next(iter(values.values()))["version"] = "changed-without-label-review"
    dataset.dump(catalog, values)
    monkeypatch.setattr(dataset, "KB", knowledge)
    with pytest.raises(AssertionError, match="Source drift"):
        dataset.validate()
