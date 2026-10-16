"""MR-05：本地账户、密码哈希与签名访问令牌。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import time
from app.identifiers import valid_identifier
from app.security_keys import load_or_create_key


class AuthError(ValueError):
    pass


class AuthService:
    """使用 PBKDF2 保存密码，并用 HMAC-SHA256 签发短期令牌。"""

    def __init__(
        self,
        database_path: str,
        secret_key: str = "",
        token_ttl_seconds: int = 43200,
    ):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_ttl_seconds = token_ttl_seconds
        self.secret = (
            secret_key.encode("utf-8") if secret_key
            else self._load_or_create_secret()
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password_hash BLOB NOT NULL,
                    salt BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

            # CR-01：数据库唯一约束阻止 Windows 上指向同一路径的大小写账户。
            # 历史库如已有冲突则拒绝启动，交由管理员核对归属，绝不自动合并数据。
            try:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS users_filesystem_identity "
                    "ON users(lower(rtrim(user_id, '.')))"
                )
            except sqlite3.IntegrityError as error:
                raise AuthError("历史账户存在大小写或路径冲突，请核对账户数据后再启动") from error
            connection.execute(
                "CREATE TABLE IF NOT EXISTS revoked_tokens (digest TEXT PRIMARY KEY, expires_at INTEGER NOT NULL)"
            )

    def _load_or_create_secret(self) -> bytes:
        return load_or_create_key(self.database_path.with_suffix(".secret"), lambda: secrets.token_bytes(32), 32)

    @staticmethod
    def validate_credentials(user_id: str, password: str) -> None:
        if not valid_identifier(user_id):
            raise AuthError("用户名只能包含字母、数字、点、下划线和短横线，长度为 1-64")
        if not 8 <= len(password) <= 128:
            raise AuthError("密码长度必须为 8-128 个字符")

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        # MR-05：PBKDF2 使用独立随机盐，避免数据库泄露后直接暴露明文密码。
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 310_000
        )

    def register(self, user_id: str, password: str) -> str:
        self.validate_credentials(user_id, password)
        salt = secrets.token_bytes(16)
        password_hash = self._password_hash(password, salt)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?)",
                    (user_id, password_hash, salt, int(time.time())),
                )
        except sqlite3.IntegrityError as error:
            raise AuthError("用户名已存在") from error
        return self.issue_token(user_id)

    def login(self, user_id: str, password: str) -> str:
        self.validate_credentials(user_id, password)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT password_hash, salt FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        # 即使用户不存在也执行一次哈希，降低用户名枚举的时间侧信道。
        salt = row[1] if row else bytes(16)
        actual = self._password_hash(password, salt)
        expected = row[0] if row else bytes(len(actual))
        if not hmac.compare_digest(actual, expected):
            raise AuthError("用户名或密码错误")
        return self.issue_token(user_id)

    def issue_token(self, user_id: str) -> str:
        now = int(time.time())
        payload = {"sub": user_id, "iat": now, "exp": now + self.token_ttl_seconds,
                   "jti": secrets.token_hex(16)}
        encoded = self._b64encode(json.dumps(
            payload, separators=(",", ":")
        ).encode("utf-8"))
        signature = self._b64encode(hmac.new(
            self.secret, encoded.encode("ascii"), hashlib.sha256
        ).digest())
        return f"{encoded}.{signature}"

    def verify_token(self, token: str) -> str:
        try:
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = self._b64encode(hmac.new(
                self.secret, encoded.encode("ascii"), hashlib.sha256
            ).digest())
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise AuthError("访问令牌无效")
            payload = json.loads(self._b64decode(encoded))
            user_id = payload["sub"]
            if int(payload["exp"]) <= int(time.time()):
                raise AuthError("访问令牌已过期")
            if not valid_identifier(user_id):
                raise AuthError("访问令牌无效")
            with self._connect() as connection:
                if connection.execute("SELECT 1 FROM revoked_tokens WHERE digest=?", (hashlib.sha256(token.encode()).hexdigest(),)).fetchone():
                    raise AuthError("访问令牌已退出，请重新登录")
            return user_id
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            if isinstance(error, AuthError):
                raise
            raise AuthError("访问令牌无效") from error

    def revoke_token(self, token: str) -> None:
        """CR-09/15：退出撤销当前令牌；其他设备的独立令牌保持有效。"""
        self.verify_token(token)
        payload = json.loads(self._b64decode(token.split(".", 1)[0]))
        with self._connect() as connection:
            connection.execute("DELETE FROM revoked_tokens WHERE expires_at<=?", (int(time.time()),))
            connection.execute("INSERT OR IGNORE INTO revoked_tokens VALUES (?, ?)",
                               (hashlib.sha256(token.encode()).hexdigest(), payload["exp"]))

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
