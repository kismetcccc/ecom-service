"""MR-14：支持多进程事务合并的 SQLite Memory 状态存储。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sqlite3


class StaleMemoryGeneration(ValueError):
    """CR-05：删除前创建的任务不可再写入当前记忆。"""


class MemoryStateStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # MR-14：并发首次启动时其他进程可能已在切换 WAL；锁冲突可安全忽略，
            # 后续事务仍受 busy_timeout 和 BEGIN IMMEDIATE 保护。
            try:
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower():
                    raise
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_states (
                    user_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    encrypted_payload BLOB NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_generations "
                "(user_id TEXT PRIMARY KEY, generation INTEGER NOT NULL)"
            )

    def generation(self, user_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT generation FROM memory_generations WHERE user_id=?", (user_id,),
            ).fetchone()
        return row[0] if row else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def load(self, user_id: str) -> tuple[bytes, int] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT encrypted_payload, version FROM memory_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def merge_and_save(
        self,
        user_id: str,
        incoming: dict,
        decode: Callable[[bytes], dict],
        encode: Callable[[dict], bytes],
        merge: Callable[[dict, dict], dict],
        expected_generation: int = 0,
    ) -> tuple[dict, int]:
        """MR-14：排他事务内读取、合并、写入，杜绝读旧值后的覆盖丢失。"""
        connection = self._connect()
        transaction_started = False
        try:
            connection.isolation_level = None
            connection.execute("BEGIN IMMEDIATE")
            transaction_started = True
            generation = connection.execute(
                "SELECT generation FROM memory_generations WHERE user_id=?", (user_id,),
            ).fetchone()
            if (generation[0] if generation else 0) != expected_generation:
                raise StaleMemoryGeneration("记忆已删除或授权已撤回，旧任务禁止回写")
            row = connection.execute(
                "SELECT encrypted_payload, version FROM memory_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            current = decode(row[0]) if row else {}
            merged = merge(current, incoming)
            next_version = (row[1] if row else 0) + 1
            connection.execute(
                """
                INSERT INTO memory_states(user_id, version, encrypted_payload)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    version = excluded.version,
                    encrypted_payload = excluded.encrypted_payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, next_version, encode(merged)),
            )
            connection.execute("COMMIT")
            return merged, next_version
        except Exception:
            if transaction_started:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def delete(self, user_id: str) -> bool:
        with self._connect() as connection:
            # CR-05：删除和代次递增位于同一事务，防止旧提取任务复活数据。
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO memory_generations VALUES (?, 1) "
                "ON CONFLICT(user_id) DO UPDATE SET generation=generation+1", (user_id,),
            )
            cursor = connection.execute(
                "DELETE FROM memory_states WHERE user_id = ?", (user_id,)
            )
        return cursor.rowcount > 0

    def exists(self, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM memory_states WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None
