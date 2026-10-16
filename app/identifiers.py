"""CR-01：HTTP 身份与文件存储共用的跨平台标识校验。"""

import re

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def valid_identifier(value: str) -> bool:
    # Windows 会折叠结尾的点，设备名即使带扩展名也不能用作文件名。
    return (
        isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))
        and not value.endswith(".") and value.split(".", 1)[0].lower() not in _RESERVED
    )
