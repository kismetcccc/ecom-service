"""不会因终端编码能力不足而中断业务流程的控制台输出。"""

from __future__ import annotations

import sys
from typing import TextIO


def safe_print(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """兼容 GBK 等控制台；无法显示的字符以转义形式写出。"""
    target = file or sys.stdout
    output = sep.join(str(value) for value in values) + end
    try:
        target.write(output)
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        escaped = output.encode(encoding, errors="backslashreplace").decode(encoding)
        target.write(escaped)
    if flush:
        target.flush()
