"""Safe, compact console traces for business tool calls."""

import json


def action_summary(name: str, arguments: dict) -> str:
    keys = ", ".join(sorted(arguments))
    return f"{name}({keys})"


def result_summary(raw: str) -> str:
    try:
        result = json.loads(raw)
    except (TypeError, ValueError):
        return "格式异常"
    if not isinstance(result, dict):
        return "格式异常"
    if result.get("success") is True:
        return f"成功 source={result.get('source', 'unknown')}"
    return f"失败 code={result.get('error_code', 'UNKNOWN')}"
