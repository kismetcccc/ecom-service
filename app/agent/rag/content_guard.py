"""Conservative pre-publication checks for untrusted knowledge text."""

import re

from app.agent.rag.chunker import Chunk

_BLOCKED = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(r"(?:sk-|api[_-]?key\s*[:=]\s*)[A-Za-z0-9_-]{20,}", re.I),
    re.compile(r"(?:忽略|无视)(?:之前|以上|前面).{0,30}(?:指令|规则|提示词)"),
    re.compile(r"(?:ignore|disregard) (?:all )?(?:previous|prior) instructions", re.I),
)


def validate_publishable(chunks: list[Chunk], *, allow_demo: bool) -> list[Chunk]:
    """Exclude drafts/retired and stop release if likely secrets/instructions appear."""
    publishable = []
    for chunk in chunks:
        if chunk.status == "demo" and allow_demo or chunk.status == "published":
            if any(pattern.search(chunk.text) for pattern in _BLOCKED):
                raise ValueError(f"文档 {chunk.doc} 包含疑似凭据或提示注入，禁止发布")
            publishable.append(chunk)
    if not publishable:
        raise ValueError("没有已审核发布的知识片段可构建索引")
    return publishable
