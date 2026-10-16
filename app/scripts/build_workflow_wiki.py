"""RAG → draft Wiki. Generation is explicit; linting never calls an LLM."""

import argparse
from pathlib import Path
from uuid import uuid4

from app.config.settings import settings
from app.multi_agent.wiki import WikiSnapshot, compile_wiki, lint_wiki, load_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=Path(settings.kb_dir))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--generate", action="store_true", help="调用模型生成待审核草稿")
    action.add_argument("--lint", type=Path, help="离线核对已有草稿与 RAG 来源")
    parser.add_argument("--output-dir", type=Path, default=Path("app/sessions/workflow_wiki"))
    parser.add_argument("--max-tokens", type=int, default=4000, help="Wiki 生成输出预算，默认 4000")
    parser.add_argument("--no-thinking", action="store_true",
                        help="向支持 enable_thinking 的服务发送 false，关闭思考模式（如百炼混合思考模型）")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须大于 0")
    sources = load_sources(args.kb_dir, allow_demo=settings.rag_allow_demo_policies)
    if args.lint:
        snapshot = WikiSnapshot.model_validate_json(args.lint.read_text(encoding="utf-8"))
        errors = lint_wiki(snapshot, sources)
        for error in errors:
            print(error)
        print("来源校验通过；语义正确性仍需人工审核" if not errors else "来源校验失败")
        return 1 if errors else 0
    if not args.generate:
        print(f"可编译来源片段：{len(sources)}；使用 --generate 调用模型生成草稿，使用 --lint 离线校验")
        return 0
    from app.config.openai_client import create_openai_client
    try:
        with create_openai_client() as client:
            snapshot = compile_wiki(client, settings.model_name, sources, max_tokens=args.max_tokens,
                                    disable_thinking=args.no_thinking)
    except ValueError as exc:
        print(f"Wiki 生成失败：{exc}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"draft-{uuid4().hex}.json"
    with destination.open("x", encoding="utf-8") as output:
        output.write(snapshot.model_dump_json(indent=2))
    print(f"已生成待审核草稿：{destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
