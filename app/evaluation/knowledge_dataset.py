"""Build/validate synthetic knowledge evaluation data. No API calls or online writes.

Run: python -m app.evaluation.knowledge_dataset --build
     python -m app.evaluation.knowledge_dataset --validate
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from app.agent.rag.chunker import chunk_markdown_dir
from app.multi_agent.wiki import source_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "app/evaluation/data/knowledge_v1"
KB = ROOT / "app/agent/rag/knowledge"
PREFIXES = {"R": "退换货政策", "D": "配送说明", "M": "会员权益", "F": "常见问题FAQ"}
PAGES = {
    "R": "退换货条件与退款流程", "D": "配送范围时效与异常处理",
    "M1": "会员等级与核心权益", "M2": "会员等级与核心权益", "M5": "会员等级与核心权益",
    "M3": "积分与生日特权", "M4": "积分与生日特权",
    "F1": "订单与支付常见问题", "F2": "商品与价格常见问题",
    "F3": "售后服务常见问题", "F4": "账户与安全常见问题",
}
STYLES = ("standard", "colloquial", "challenge")
ANSWERABILITIES = {"answerable", "partial", "unanswerable", "needs_tool", "clarify"}


def read_tsv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_id(code):
    return f"{PREFIXES[code[0]]}#{int(code[1:]):02d}-00"


def grouping(seed):
    # Keep all questions about the same primary source chapter together,
    # including its unknowns/adversarial prompts. Cross-source cases get a
    # separate challenge bucket; shared source text is disclosed, not blind.
    if seed["category"] == "cross_source":
        return "cross_source"
    return seed["sources"].split(",")[0] if seed["sources"] != "-" else "no_source"


def split_for(group):
    if group in {"R2", "D6", "M4", "F2", "cross_source"}:
        return "test"
    if group in {"R4", "D3", "M1", "F4", "no_source"}:
        return "dev"
    return "train"


def make_row(seed, index):
    codes = [] if seed["sources"] == "-" else seed["sources"].split(",")
    context = [chunk_id(code) for code in codes]
    positive = seed["answerability"] in {"answerable", "partial"}
    required = context if positive else []
    points = seed["answer_points"].split("；")
    pages = list(dict.fromkeys(PAGES.get(code, PAGES.get(code[0])) for code in codes))
    return {
        "id": f"{seed['id']}-{index + 1}", "family_id": seed["id"],
        "split_group": grouping(seed), "split": split_for(grouping(seed)),
        "category": seed["category"], "style": STYLES[index],
        "query": seed[f"query_{STYLES[index]}"], "answerability": seed["answerability"],
        "source_scope": "project_demo_only", "label_status": "synthetic_source_checked_not_independently_reviewed",
        "required_evidence_groups": [[item] for item in required],
        "relevant_chunk_ids": required, "context_chunk_ids": context,
        "reference_page_titles": pages, "page_labels_usage": "diagnostic_only_not_semantic_gold",
        "answer_points": points, "reference_answer": "按本项目演示资料：" + "；".join(points) + "。",
        "forbidden_answer_example": seed["bad_answer"],
        "must_not": ["把演示政策宣称为现实法律或正式平台承诺", "在没有成功工具证据时宣称已经完成业务操作"],
        "evaluation_layers": ["answer", "citation", "safety"] + (["retrieval"] if positive else []),
        "boundary": ("only_partial_answer_supported" if seed["answerability"] == "partial"
                     else seed["answerability"]),
        "expected_tool_policy": "read_only_policy_search; any business write requires separate authorization",
    }


def build():
    seeds = read_tsv("seeds.tsv")
    assert all(None not in row and all(v is not None for v in row.values()) for row in seeds)
    assert len({row["id"] for row in seeds}) == len(seeds)
    sources = {c.chunk_id: c for c in chunk_markdown_dir(KB)}
    snapshot = {key: {"sha256": source_hash(c), "chunk": c.to_dict()} for key, c in sources.items()}
    dump(DATA / "source_snapshot.json", snapshot)
    cases = [make_row(seed, i) for seed in seeds for i in range(3)]
    jsonl(DATA / "qa_cases.jsonl", cases)
    by_family = {row["family_id"]: row for row in cases if row["style"] == "standard"}
    dialogues = []
    for seed in read_tsv("dialogues.tsv"):
        base = by_family[seed["base_id"]]
        row = {**base, "id": seed["id"], "style": "multi_turn", "query": seed["query"],
               "dialogue": [{"role": "user", "content": seed["previous_user"]},
                            {"role": "assistant", "content": seed["previous_assistant"]},
                            {"role": "user", "content": seed["query"]}],
               "history_provenance": "authored_fixture_not_observed_model_output"}
        dialogues.append(row)
    jsonl(DATA / "dialogue_cases.jsonl", dialogues)
    pairs = []
    for base in by_family.values():
        for verdict, text in [("approve", base["reference_answer"]), ("reject", base["forbidden_answer_example"])]:
            pairs.append({"id": base["family_id"] + "-" + verdict, "family_id": base["family_id"],
                          "split_group": base["split_group"], "split": base["split"],
                          "request": base["query"], "draft": text, "expected": verdict,
                          "source_chunk_ids": base["context_chunk_ids"],
                          "answer_points": base["answer_points"],
                          "label_status": "synthetic_judge_calibration_not_observed_badcase",
                          "reject_reason": base["forbidden_answer_example"] if verdict == "reject" else None})
    jsonl(DATA / "review_pairs.jsonl", pairs)
    lifecycle = [{**row, "split": "regression", "requires_isolated_fixture": True,
                  "execution": "specification_only_no_runner_in_this_data_package"}
                 for row in read_tsv("lifecycle.tsv")]
    jsonl(DATA / "lifecycle_cases.jsonl", lifecycle)
    for split in ("train", "dev", "test"):
        subset = [row for row in cases if row["split"] == split]
        jsonl(DATA / f"qa_{split}.jsonl", subset)
        # Existing quantify retrieval entry point requires nonempty labels.
        positives = [r for r in subset if r["relevant_chunk_ids"]]
        dump(DATA / f"retrieval_{split}.json", {
            "label_status": "合成分组划分；非独立盲测；只含可回答及部分可回答的证据检索题",
            "cases": [{"id": r["id"], "query": r["query"], "relevant_chunk_ids": r["relevant_chunk_ids"]}
                      for r in positives]})
        dump(DATA / f"agent_{split}.json", {
            "label_status": "兼容EvalCase的有损导出；完整语义和负例评分必须读qa_cases；业务写操作须隔离",
            "cases": [{"id": r["id"], "description": r["category"] + ": " + r["boundary"],
                       "turns": [r["query"]], "expected_tools": ["search_knowledge"] if r["relevant_chunk_ids"] else [],
                       "relevant_chunk_ids": r["relevant_chunk_ids"], "retrieval_k": 3}
                      for r in subset]})
    observed = read_jsonl(DATA / "observed_badcases.jsonl")
    files = sorted(DATA.glob("*.jsonl")) + sorted(DATA.glob("*.tsv")) + [
        p for p in sorted(DATA.glob("*.json")) if p.name != "manifest.json"]
    manifest = {
        "version": "knowledge-v1-2026-10-15", "authoring": "Codex authored synthetic cases grounded in demo sources",
        "independent_human_review": False, "live_model_evaluation_run": False,
        "counts": {"semantic_seed_families": len(seeds), "qa": len(cases), "multi_turn": len(dialogues),
                   "judge_calibration": len(pairs), "lifecycle_specifications": len(lifecycle),
                   "observed_badcases": len(observed),
                   "total_primary_records": len(cases) + len(dialogues) + len(pairs) + len(lifecycle) + len(observed)},
        "qa_by_category": dict(Counter(r["category"] for r in cases)),
        "qa_by_answerability": dict(Counter(r["answerability"] for r in cases)),
        "qa_by_split": dict(Counter(r["split"] for r in cases)),
        "split_groups": {g: split_for(g) for g in sorted({r["split_group"] for r in cases})},
        "source_count": len(sources),
        "warning": "改写、对话和审校对共享场景，不是相互独立样本；分组切分仍共享政策事实，不能称真实用户盲测",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    dump(DATA / "manifest.json", manifest)
    return validate()


def validate():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, f"File changed: {name}"
    snapshot = json.loads((DATA / "source_snapshot.json").read_text(encoding="utf-8"))
    current = {c.chunk_id: source_hash(c) for c in chunk_markdown_dir(KB)}
    assert current == {key: value["sha256"] for key, value in snapshot.items()}, "Source drift: re-review labels"
    cases = read_jsonl(DATA / "qa_cases.jsonl")
    dialogues = read_jsonl(DATA / "dialogue_cases.jsonl")
    pairs = read_jsonl(DATA / "review_pairs.jsonl")
    assert len({r["query"] for r in cases}) == len(cases), "Duplicate QA query"
    family_splits, group_splits = defaultdict(set), defaultdict(set)
    ids = set()
    for row in cases + dialogues + pairs:
        assert row["id"] not in ids, row["id"]
        ids.add(row["id"])
        family_splits[row["family_id"]].add(row["split"])
        group_splits[row["split_group"]].add(row["split"])
        refs = row.get("context_chunk_ids", row.get("source_chunk_ids", []))
        assert set(refs) <= snapshot.keys(), row["id"]
        if "answerability" in row:
            assert row["answerability"] in ANSWERABILITIES
            positive = row["answerability"] in {"answerable", "partial"}
            assert bool(row["relevant_chunk_ids"]) == positive
            assert all(group and set(group) <= snapshot.keys() for group in row["required_evidence_groups"])
            assert row["answer_points"] and row["forbidden_answer_example"]
    assert all(len(v) == 1 for v in family_splits.values()), "Family leakage"
    assert all(len(v) == 1 for v in group_splits.values()), "Topic leakage"
    covered = {c for r in cases for c in r["context_chunk_ids"]}
    assert covered == snapshot.keys(), "Uncovered source chapters"
    from app.evaluation.dataset import load_dataset
    for split in ("train", "dev", "test"):
        assert load_dataset(DATA / f"agent_{split}.json")
        rows = read_jsonl(DATA / f"qa_{split}.jsonl")
        assert rows == [r for r in cases if r["split"] == split]
        exported = json.loads((DATA / f"retrieval_{split}.json").read_text(encoding="utf-8"))["cases"]
        expected = [r for r in rows if r["relevant_chunk_ids"]]
        assert [r["id"] for r in exported] == [r["id"] for r in expected]
        assert all(set(r["relevant_chunk_ids"]) <= snapshot.keys() for r in exported)
    return {"validation": "passed", **manifest["counts"], "source_chapters_covered": len(covered),
            "qa_by_split": manifest["qa_by_split"], "note": "Data checks only; no model quality measured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build() if args.build else validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
