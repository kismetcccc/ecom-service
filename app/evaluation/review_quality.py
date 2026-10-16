"""Small labeled reviewer benchmark; live model use must be explicitly enabled."""

import argparse
import json
from pathlib import Path

from app.agent.context_budget import ContextBudgetManager
from app.config.settings import settings
from app.multi_agent.quality import QualityReviewAgent
from app.schemas.response import CustomerServiceResponse, IntentType


def summarize(rows):
    good = [row for row in rows if row["expected"] == "approve"]
    bad = [row for row in rows if row["expected"] == "reject"]
    false_accepts = sum(row["status"] == "approved" for row in bad)
    false_rejects = sum(row["status"] == "blocked" for row in good)
    unavailable = sum(row["status"] not in {"approved", "blocked"} for row in rows)
    matched = sum((row["status"] == "approved" and row["expected"] == "approve") or
                  (row["status"] == "blocked" and row["expected"] == "reject") for row in rows)
    return {
        "case_count": len(rows), "matched": matched,
        "false_accepts": false_accepts, "false_rejects": false_rejects,
        "unavailable": unavailable,
        "model_calls": sum(row["model_calls"] for row in rows),
        "passed": bool(rows) and matched == len(rows),
        "cases": rows,
    }


def run_cases(agent, cases):
    rows = []
    # Isolated CLI evaluation: exactly one review attempt, no repair generation.
    previous = settings.multi_agent_review_mode, settings.multi_agent_review_max_repairs
    settings.multi_agent_review_mode = "enforce"
    settings.multi_agent_review_max_repairs = 0
    try:
        for case in cases:
            messages = []
            if case["tool"]:
                messages = [
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "eval-call", "function": {
                        "name": case["tool"], "arguments": '{"order_id":"ORD-EVAL-001"}',
                    }}]},
                    {"role": "tool", "tool_call_id": "eval-call", "content": json.dumps(case["data"], ensure_ascii=False)},
                ]
            response = CustomerServiceResponse(intent=IntentType.OTHER, confidence=1, reply=case["draft"])
            result = agent.evaluate(case["request"], response, messages)
            rows.append({"id": case["id"], "expected": case["expected"],
                         "status": result.quality_review.status,
                         "model_calls": result.quality_review.model_calls,
                         "elapsed_ms": result.quality_review.elapsed_ms})
    finally:
        settings.multi_agent_review_mode, settings.multi_agent_review_max_repairs = previous
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="调用当前模型评估，可能产生费用")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("review_cases.json").read_text(encoding="utf-8"))
    if not args.live:
        print(json.dumps({"cases": len(cases), "live": False, "message": "Add --live to evaluate the configured model; no model called."}))
        return 0
    from app.config.openai_client import create_openai_client
    with create_openai_client() as client:
        reviewer = QualityReviewAgent(
            client, settings.multi_agent_review_model or settings.model_name,
            ContextBudgetManager(settings.context_window_tokens, settings.context_reserved_output_tokens, settings.context_keep_recent_messages),
        )
        report = run_cases(reviewer, cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
