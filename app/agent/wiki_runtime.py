"""Independent, versioned Wiki publications and hot knowledge-mode selection.

Only the local administration CLI writes this store. Retrieval is read-only and
never loads or mutates a vector index. Original source chunks remain citations.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import settings
from app.multi_agent.wiki import WikiSnapshot, lint_wiki, load_sources


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def store_dir():
    root = resolve_path(settings.wiki_store_dir)
    # Misconfiguration must not turn a Wiki publish into a write inside RAG storage.
    for value in (settings.kb_dir, settings.chroma_persist_dir, settings.kb_index_path):
        protected = resolve_path(value)
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("Wiki storage must be separate from RAG sources and indexes")
    return root


class KnowledgeState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["rag", "wiki", "wiki_first"] = "rag"
    active_release: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class WikiRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    reviewed_by: str = Field(min_length=1, max_length=120)
    published_at: str
    snapshot: WikiSnapshot


def _read(path):
    if path.stat().st_size > 1_000_000:
        raise ValueError("Wiki file exceeds size limit")
    return path.read_text(encoding="utf-8")


def read_state():
    path = store_dir() / "control.json"
    if not path.exists():
        return KnowledgeState(mode=settings.knowledge_mode)
    return KnowledgeState.model_validate_json(_read(path))


@contextmanager
def _admin_lock():
    root = store_dir()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "control.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_state(state):
    root = store_dir()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as out:
            temporary = Path(out.name)
            out.write(state.model_dump_json(indent=2))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, root / "control.json")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def current_sources():
    return load_sources(resolve_path(settings.kb_dir), allow_demo=settings.rag_allow_demo_policies)


def load_release(release_id):
    if not isinstance(release_id, str) or not re.fullmatch(r"[a-f0-9]{32}", release_id):
        raise ValueError("Invalid Wiki release identifier")
    release = WikiRelease.model_validate_json(_read(store_dir() / "releases" / f"{release_id}.json"))
    if release.release_id != release_id:
        raise ValueError("Wiki release identifier mismatch")
    return release


def validate_release(release, sources):
    errors = lint_wiki(release.snapshot, sources)
    if any(page.open_questions for page in release.snapshot.content.pages):
        errors.append("发布前须处理页面中的待确认问题")
    if not release.reviewed_by.strip():
        errors.append("缺少审核人")
    if errors:
        raise ValueError("; ".join(errors))


def _export_markdown(release):
    root = store_dir() / "pages" / release.release_id
    root.mkdir(parents=True, exist_ok=False)
    pages = release.snapshot.content.pages
    names = {page.title: f"page-{index + 1}.md" for index, page in enumerate(pages)}
    index_text = ["# LLM Wiki", "", f"版本：{release.release_id}", "",
                  "此目录是发布稿的阅读视图；修改请编辑草稿 JSON 后重新发布。", ""]
    for page in pages:
        index_text.append(f"- [{page.title}]({names[page.title]})")
        lines = [f"# {page.title}", "", page.summary, "", "## 建议步骤", ""]
        lines.extend(f"{i + 1}. {step}" for i, step in enumerate(page.steps))
        lines.extend(["", "## 原文引用", ""])
        for citation in page.citations:
            lines.extend([f"来源片段：`{citation.chunk_id}`", "", "> " + citation.quote.replace("\n", "\n> "), ""])
        lines.extend(["## 关联页面", ""])
        lines.extend(f"- [{title}]({names[title]})" for title in page.related_titles)
        lines.extend(["", "## 引用本页", ""])
        lines.extend(f"- [{other.title}]({names[other.title]})" for other in pages if page.title in other.related_titles)
        lines.extend(["", "[返回目录](index.md)", ""])
        (root / names[page.title]).write_text("\n".join(lines), encoding="utf-8")
    (root / "index.md").write_text("\n".join(index_text) + "\n", encoding="utf-8")


def publish(snapshot, reviewed_by, *, activate_release=True):
    release = WikiRelease(release_id=uuid4().hex, reviewed_by=reviewed_by.strip(),
                          published_at=datetime.now(timezone.utc).isoformat(), snapshot=snapshot)
    validate_release(release, current_sources())
    if len(release.model_dump_json().encode("utf-8")) > 1_000_000:
        raise ValueError("Wiki release exceeds size limit")
    with _admin_lock():
        # Recheck after acquiring the administrative lock, not only before it.
        validate_release(release, current_sources())
        state = read_state()
        root = store_dir() / "releases"
        root.mkdir(exist_ok=True)
        _export_markdown(release)
        with (root / f"{release.release_id}.json").open("x", encoding="utf-8") as out:
            out.write(release.model_dump_json(indent=2))
            out.flush()
            os.fsync(out.fileno())
        # Release artifacts are complete before atomically updating the pointer.
        if activate_release:
            _write_state(state.model_copy(update={"active_release": release.release_id}))
    return release


def switch_mode(mode):
    with _admin_lock():
        try:
            state = read_state()
        except (ValueError, OSError):
            if mode != "rag":
                raise
            # Emergency return to RAG can repair a corrupt pointer; immutable
            # releases stay available for later explicit activation.
            state = KnowledgeState(mode="rag")
        updated = KnowledgeState(mode=mode, active_release=state.active_release)
        if mode != "rag":
            validate_release(load_release(updated.active_release), current_sources())
        _write_state(updated)
    return updated


def activate(release_id):
    with _admin_lock():
        validate_release(load_release(release_id), current_sources())
        state = read_state().model_copy(update={"active_release": release_id})
        _write_state(state)
    return state


def _tokens(text):
    parts = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text.lower())
    return {token for part in parts for token in
            ([part] if not re.fullmatch(r"[\u3400-\u9fff]+", part) or len(part) == 1
             else [part[i:i + 2] for i in range(len(part) - 1)])}


def search_wiki(query, top_k, state):
    base = {"backend": "wiki", "provider": "wiki", "query": query, "results": []}
    if not state.active_release:
        return {**base, "success": False, "error_code": "INDEX_MISSING", "error": "尚未发布 Wiki"}
    try:
        release = load_release(state.active_release)
        sources = current_sources()
        validate_release(release, sources)
    except (ValueError, OSError):
        return {**base, "success": False, "error_code": "UNAVAILABLE", "error": "Wiki 版本不可用或来源已变化，请重新审核发布"}
    tokens = _tokens(query)
    ranked = []
    for index, page in enumerate(release.snapshot.content.pages):
        body = " ".join([page.title, page.summary, *page.steps, *(c.quote for c in page.citations)])
        score = len(tokens & _tokens(body)) + len(tokens & _tokens(page.title))
        if score:
            ranked.append((score, index, page))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    results, context, seen = [], [], set()
    for score, index, page in ranked[:top_k]:
        ids = list(dict.fromkeys(c.chunk_id for c in page.citations))
        if len(seen | set(ids)) > 20:
            continue
        context.append({"page_id": f"page-{index + 1}", "title": page.title,
                        "summary": page.summary, "steps": page.steps, "source_chunk_ids": ids,
                        "related_titles": page.related_titles,
                        "notice": "已审核的模型归纳；政策事实须以 results 中的原文为准"})
        for chunk_id in ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = sources[chunk_id]
            results.append({**chunk.to_dict(), "score": round(score / max(1, 2 * len(tokens)), 4)})
    if not results:
        return {**base, "success": False, "error_code": "NO_MATCH", "error": "Wiki 没有匹配的已发布页面"}
    return {**base, "success": True, "results": results, "wiki_pages": context,
            "index_version": release.release_id, "score_type": "lexical_overlap"}
