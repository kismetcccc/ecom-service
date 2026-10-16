"""Reusable embedding cache for offline, versioned index builds.

Only chunk hashes, model identifiers and vectors are stored; raw policy text is not.
The published index is still rebuilt as a fresh version, so removed chunks vanish.
"""

import json
import math
import os
from pathlib import Path

from app.agent.rag.chunker import Chunk


def encode_changed(chunks: list[Chunk], embedder, cache_path: Path) -> tuple[list[list[float]], int]:
    """Embed only changed (model, text hash) pairs; write cache atomically."""
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if data.get("model") != embedder.model:
        data = {"model": embedder.model, "vectors": {}}
    previous = data.get("vectors", {})
    if not isinstance(previous, dict):
        previous = {}
    keys = [chunk.content_hash or chunk.chunk_id for chunk in chunks]
    missing_keys = list(dict.fromkeys(key for key in keys if key not in previous))
    text_by_key = {key: chunk.text for key, chunk in zip(keys, chunks)}
    fresh = embedder.encode([text_by_key[key] for key in missing_keys]) if missing_keys else []
    if len(fresh) != len(missing_keys):
        raise ValueError("Embedding 返回数量与新增片段不一致")
    combined = {key: previous[key] for key in keys if key in previous}
    combined.update(zip(missing_keys, fresh))
    vectors = [combined[key] for key in keys]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions or any(
        not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise ValueError("Embedding 缓存或新向量的维度/数值无效")
    # Prune deleted chunks, then replace the cache file only after validation.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps({"model": embedder.model, "vectors": combined}), encoding="utf-8")
        os.replace(temp, cache_path)
    finally:
        temp.unlink(missing_ok=True)
    return vectors, len(missing_keys)
