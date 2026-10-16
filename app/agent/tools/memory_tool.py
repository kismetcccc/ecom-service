"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager

_memory_manager: MemoryManager | None = None


def set_memory_manager(manager: MemoryManager) -> None:
    """由 Agent 初始化时调用，注入 MemoryManager 引用。"""
    global _memory_manager
    _memory_manager = manager


def recall_memory(manager: MemoryManager | None, query: str = "") -> dict:
    """读取指定 MemoryManager，供并发 Agent 绑定自己的用户上下文。"""
    if manager is None or not manager.memory_enabled:
        return {"success": False, "error": "记忆系统未启用"}

    relevant_facts = manager.ltm.retrieve(query)[0] if query else manager.ltm.active_facts
    result: dict = {
        "success": True,
        # MR-06：工具输出保持 JSON 可序列化，并携带事实元数据。
        "short_term_facts": manager.stm.to_dict()["facts"],
        # MR-12：未授权时工具也不得旁路读取长期记忆。
        "long_term_facts": [
            {
                "content": f.content,
                "category": f.category,
                "id": f.id,
                "source": f.source,
                "confidence": f.confidence,
                "source_message_id": f.source_message_id,
            }
            for f in relevant_facts
        ] if manager.long_term_consent else [],
        "long_term_consent": manager.long_term_consent,
    }

    if manager.long_term_consent and manager.ltm.interaction_summaries:
        result["recent_interactions"] = [
            s["summary"] for s in manager.ltm.interaction_summaries[-3:]
        ]

    return result


def recall_user_memory(query: str = "") -> dict:
    """查询当前用户的记忆信息（长期记忆和短期记忆）。"""
    return recall_memory(_memory_manager, query)
