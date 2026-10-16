"""Deterministic publication gate for retrieved policy chunks."""

from datetime import date
from urllib.parse import urlsplit

from app.agent.rag.chunker import Chunk


def is_current_policy(chunk: Chunk, *, allow_demo: bool, today: date | None = None) -> bool:
    """Draft/retired/expired material must never become a current policy citation."""
    today = today or date.today()
    if chunk.status == "demo":
        return allow_demo
    if chunk.status != "published":
        return False
    if not all((chunk.policy_id, chunk.version, chunk.owner,
                chunk.source_url, chunk.effective_from)):
        return False
    if urlsplit(chunk.source_url).scheme != "https":
        return False
    try:
        return (date.fromisoformat(chunk.effective_from) <= today and
                (not chunk.effective_to or today <= date.fromisoformat(chunk.effective_to)))
    except ValueError:
        return False
