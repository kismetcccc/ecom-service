"""Declarative exact-match tool responses; never execute code from a dataset."""
import json
from pathlib import Path


def load_tool_fixtures(path):
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("fixture file exceeds 1 MB")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("fixtures must be an array")
    lookup = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not isinstance(row.get("arguments"), dict)
                or not isinstance(row.get("result"), dict)
                or type(row["result"].get("success")) is not bool):
            raise ValueError("invalid fixture contract")
        key = (row["name"], json.dumps(row["arguments"], sort_keys=True, ensure_ascii=False))
        if key in lookup:
            raise ValueError("duplicate fixture")
        lookup[key] = json.dumps(row["result"], ensure_ascii=False)
    def execute(name, arguments):
        return lookup.get((name, json.dumps(arguments, sort_keys=True, ensure_ascii=False)),
                          '{"success":false,"error_code":"FIXTURE_MISS"}')
    return execute
