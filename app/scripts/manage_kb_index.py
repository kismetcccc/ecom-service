"""Inspect or explicitly roll back a Qdrant knowledge index alias.

Usage: python -m app.scripts.manage_kb_index list
       python -m app.scripts.manage_kb_index rollback --to ecom_kb_<version>
"""

import argparse
import json

from app.config.settings import settings
from app.agent.rag.backends.qdrant_backend import QdrantBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant 知识库索引版本管理")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出物理集合和当前别名")
    rollback = commands.add_parser("rollback", help="人工切回保留的物理集合")
    rollback.add_argument("--to", required=True, help="list 输出中的物理集合名")
    args = parser.parse_args()
    backend = QdrantBackend(
        url=settings.qdrant_url, collection_name=settings.qdrant_collection,
        api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout_seconds,
    )
    if args.command == "list":
        print(json.dumps(backend.versions(), ensure_ascii=False, indent=2))
    else:
        # 回滚必须与当前查询模型一致，避免旧向量被新模型误用。
        backend.rollback(args.to, expected_model=settings.embedding_model)
        print(f"已回滚别名 {settings.qdrant_collection} -> {args.to}")


if __name__ == "__main__":
    main()
