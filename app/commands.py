"""CLI 客户端保留命令。"""

CONTROL_COMMANDS = frozenset({
    "quit", "exit", "reset", "memory", "memory on", "memory off", "skills"
})


def is_control_command(value: str) -> bool:
    """判断输入是否为必须由客户端处理的独立控制命令。"""
    return value.strip().casefold() in CONTROL_COMMANDS
