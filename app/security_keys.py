"""本地开发密钥的独占创建与完整性读取；生产优先使用配置密钥。"""

import os
import time
from pathlib import Path
from typing import Callable


def load_or_create_key(path: Path, generate: Callable[[], bytes], size: int, *, strip: bool = False) -> bytes:
    """CR-15：并发读者等待创建者写完，空文件不能当作有效密钥使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        for _ in range(20):
            key = path.read_bytes()
            if strip:
                key = key.strip()
            if len(key) == size:
                return key
            time.sleep(0.05)
        raise ValueError("本地密钥文件不完整，请保留原件并恢复正确密钥")
    key = generate()
    with os.fdopen(descriptor, "wb") as target:
        target.write(key)
        target.flush()
        os.fsync(target.fileno())
    return key
