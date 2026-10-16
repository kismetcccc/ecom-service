"""MR-12：Memory 隐私同意、敏感数据过滤与静态加密。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from cryptography.fernet import Fernet, InvalidToken
from app.security_keys import load_or_create_key


_SECRET_PATTERNS = (
    re.compile(r"(?i)(密码|password|passwd|验证码|短信码|otp|cvv|安全码)\s*[:：]?\s*\S+"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),  # 完整身份证号
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),      # 常见银行卡号
    re.compile(r"(?i)\b(?:sk|ak)-[A-Za-z0-9_-]{12,}\b"),
)

_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)(忽略|无视|覆盖|绕过).{0,20}(指令|规则|提示词|system)"),
    re.compile(r"(?i)(泄露|输出|显示|告诉我).{0,20}(系统提示词|system prompt|开发者指令)"),
    re.compile(r"(?i)(调用|执行|触发).{0,12}(工具|函数|tool|function)"),
    re.compile(r"(?i)(请|必须).{0,12}(永久记住|保存).{0,20}(指令|提示词|命令)"),
    re.compile(r"(?i)<\|?(system|assistant|developer)\|?>|\[(system|developer)\]"),
)


def contains_prohibited_secret(text: str) -> bool:
    """代码层兜底：命中凭据、验证码或完整金融/身份号码时禁止记忆。"""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def contains_prompt_injection(text: str) -> bool:
    """MR-16：识别不应跨轮持久化的指令、工具操纵和角色伪造内容。"""
    return any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS)


def filter_memory_text(text: str) -> str | None:
    value = text.strip()
    if (
        not value
        or contains_prohibited_secret(value)
        or contains_prompt_injection(value)
    ):
        return None
    return value


class MemoryCipher:
    """使用 Fernet 对完整 Memory 文档实施认证加密。"""

    def __init__(self, memory_dir: str | Path, configured_key: str = ""):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        key = configured_key.encode("ascii") if configured_key else self._load_key()
        self.fernet = Fernet(key)

    def _load_key(self) -> bytes:
        return load_or_create_key(self.memory_dir / ".memory.key", Fernet.generate_key, 44, strip=True)

    def encrypt_json(self, payload: dict) -> bytes:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.fernet.encrypt(raw)

    def decrypt_json(self, raw: bytes) -> dict:
        # MR-12：兼容并自动迁移历史明文 JSON；后续 save 会改写为密文。
        if raw.lstrip().startswith(b"{"):
            return json.loads(raw.decode("utf-8"))
        try:
            decrypted = self.fernet.decrypt(raw)
        except InvalidToken as error:
            raise ValueError("长期记忆密文无法验证") from error
        return json.loads(decrypted.decode("utf-8"))


class MemoryConsentStore:
    """以 SQLite 事务保存用户长期记忆授权状态。"""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_consents (
                    user_id TEXT PRIMARY KEY,
                    granted INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def is_granted(self, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT granted FROM memory_consents WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return bool(row and row[0])

    def set_granted(self, user_id: str, granted: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_consents(user_id, granted) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    granted = excluded.granted,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, int(granted)),
            )
