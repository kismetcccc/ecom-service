"""Local administration of independent Wiki releases and hot knowledge routing."""

import argparse
import json
from pathlib import Path

from app.agent.wiki_runtime import (
    activate, current_sources, load_release, publish, read_state,
    store_dir, switch_mode, validate_release,
)
from app.multi_agent.wiki import WikiSnapshot, analyze_impact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("status")
    actions.add_parser("list")
    change = actions.add_parser("switch")
    change.add_argument("mode", choices=["rag", "wiki", "wiki_first"])
    release = actions.add_parser("publish")
    release.add_argument("draft", type=Path)
    release.add_argument("--stage-only", action="store_true", help="仅生成候选版本，不改变在线指针")
    release.add_argument("--reviewed-by", required=True, help="实际审核人；先审阅草稿并处理 open_questions")
    restore = actions.add_parser("activate")
    restore.add_argument("release_id", help="切换/回滚到已有、仍与来源一致的版本")
    impact = actions.add_parser("impact", help="分析来源变化及页面依赖传播；不修改任何版本")
    impact.add_argument("release_id", nargs="?", help="省略时分析当前激活版本")
    query = actions.add_parser("query")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--release", help="只读试查候选 Wiki 版本，不切换线上模式且不回退 RAG")
    args = parser.parse_args(argv)
    try:
        if args.action == "publish":
            if args.draft.stat().st_size > 1_000_000:
                raise ValueError("草稿超过大小限制")
            snapshot = WikiSnapshot.model_validate_json(args.draft.read_text(encoding="utf-8"))
            result = publish(snapshot, args.reviewed_by, activate_release=not args.stage_only)
            output = {"release_id": result.release_id, "mode": read_state().mode,
                      "active": read_state().active_release == result.release_id,
                      "index": str(store_dir() / "pages" / result.release_id / "index.md")}
        elif args.action == "switch":
            output = switch_mode(args.mode).model_dump()
        elif args.action == "activate":
            output = activate(args.release_id).model_dump()
        elif args.action == "impact":
            release = load_release(args.release_id or read_state().active_release)
            output = {"release_id": release.release_id,
                      **analyze_impact(release.snapshot, current_sources())}
        elif args.action == "query":
            if args.release:
                from app.agent.wiki_runtime import KnowledgeState, search_wiki
                if not args.query.strip() or not 1 <= args.top_k <= 10:
                    raise ValueError("查询不能为空，top-k 须在 1 到 10 之间")
                output = search_wiki(args.query, args.top_k,
                                     KnowledgeState(mode="wiki", active_release=args.release))
            else:
                from app.agent.tools.knowledge import search_knowledge
                output = search_knowledge(args.query, args.top_k)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0 if output["success"] else 1
        elif args.action == "list":
            output = []
            for path in sorted((store_dir() / "releases").glob("*.json")):
                release = load_release(path.stem)
                output.append({"release_id": release.release_id, "reviewed_by": release.reviewed_by,
                               "published_at": release.published_at})
        else:
            state = read_state()
            output = state.model_dump()
            output["wiki_ready"] = False
            if state.active_release:
                try:
                    validate_release(load_release(state.active_release), current_sources())
                    output["wiki_ready"] = True
                except (ValueError, OSError) as exc:
                    output["wiki_error"] = str(exc)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"操作失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
