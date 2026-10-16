import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config.settings import settings

SESSION_VERSION = 2


class SessionCorruptedError(ValueError):
    """CR-03：损坏状态必须保留原件并显式报错，禁止静默覆盖成新会话。"""


def _cipher(path: Path):
    from app.agent.memory.privacy import MemoryCipher
    # CR-04：生产通过独立 Secret 注入密钥；本机开发密钥随用户目录持久保存。
    return MemoryCipher(path.parent / ".keys", settings.session_encryption_key)


def save_session(
    path: str,
    messages: list[dict],
    summary: Optional[str],
    short_term_memory: Optional[dict] = None,
    summary_versions: Optional[list[dict]] = None,
    summary_audit: Optional[list[dict]] = None,
) -> None:
    """把对话状态原子写入 JSON 文件。

    messages 只包含原始 user/assistant 条目（不含 system / summary）。
    short_term_memory 为短期记忆的序列化数据（第7期）。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SESSION_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
        "short_term_memory": short_term_memory,
        # MR-02：摘要版本与审计随会话原子保存，旧会话缺少字段时按空列表迁移。
        "summary_versions": summary_versions or [],
        "summary_audit": summary_audit or [],
    }

    # CR-03：独立暂存文件避免多个写者争抢固定 .tmp；fsync 后再原子替换。
    descriptor, temporary = tempfile.mkstemp(prefix=file_path.name + ".", suffix=".tmp", dir=file_path.parent)
    tmp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(_cipher(file_path).encrypt_json(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_session(path: str) -> Optional[dict]:
    """不存在时返回 None；损坏或不可读取时保留原件并拒绝覆盖。"""
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        raw = file_path.read_bytes()
        # 旧 JSON 继续可读，下一次成功保存自动改为密文，不批量覆盖历史文件。
        data = json.loads(raw) if raw.lstrip().startswith(b"{") else _cipher(file_path).decrypt_json(raw)
    except (ValueError, OSError) as e:
        raise SessionCorruptedError("会话存储无法读取，请保留文件并联系管理员恢复") from e

    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise SessionCorruptedError("会话存储格式异常，请联系管理员恢复")

    return {
        "version": data.get("version", 1),
        "summary": data.get("summary"),
        "messages": data.get("messages", []),
        "short_term_memory": data.get("short_term_memory"),
        "summary_versions": data.get("summary_versions", []),
        "summary_audit": data.get("summary_audit", []),
    }


def delete_session(path: str) -> None:
    """删除会话文件，不存在时静默。"""
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
