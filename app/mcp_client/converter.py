"""MCP tool schema to model schema; internal credentials stay hidden."""

from copy import deepcopy


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """将 MCP Tool 列表转换为 OpenAI function calling 所需的 tools 格式。

    MCP Tool 的 inputSchema 本身就是 JSON Schema，和 OpenAI parameters 字段格式一致，
    只需要调整外层信封结构。
    """
    result = []
    for tool in mcp_tools:
        schema = deepcopy(getattr(tool, "inputSchema", None) or tool.input_schema)
        schema.get("properties", {}).pop("auth_token", None)
        schema["required"] = [name for name in schema.get("required", []) if name != "auth_token"]
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return result
