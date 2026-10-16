"""MCP Client：同步封装，通过 Streamable HTTP 连接 MCP Server。

内部使用后台线程运行异步事件循环，对外暴露同步接口，
使得现有的同步 Agent 代码无需改动即可调用 MCP 工具。
"""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import json
import threading

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.mcp_client.converter import mcp_tools_to_openai
from app.config.settings import settings


class MCPClient:
    """同步 MCP 客户端，通过 Streamable HTTP 连接远程 MCP Server。"""

    def __init__(self, server_url: str):
        self._server_url = server_url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._connected = threading.Event()
        self._close_event: asyncio.Event | None = None
        self._call_slots = threading.BoundedSemaphore(max(1, settings.mcp_max_inflight))

    def connect(self) -> list[dict]:
        """连接 MCP Server，发现工具，返回 OpenAI 格式的工具定义列表。"""
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._close_event = asyncio.Event()
        self._connected.clear()
        tool_definitions: list[dict] = []
        error_holder: list[Exception] = []

        def run_loop():
            try:
                loop.run_until_complete(self._run(tool_definitions, error_holder))
            finally:
                loop.close()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        if not self._connected.wait(timeout=settings.mcp_connect_timeout):
            self.close()
            raise TimeoutError("MCP 连接超时")

        if error_holder:
            self.close()
            raise error_holder[0]

        if not tool_definitions:
            self.close()
            raise RuntimeError("MCP 未发现可用工具")
        return tool_definitions

    async def _run(self, tool_definitions: list[dict], error_holder: list[Exception]):
        """后台协程：建立连接 → 发现工具 → 保持存活等待调用。"""
        try:
            async with streamable_http_client(self._server_url) as streams:
                read, write = streams[:2]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session

                    tools_result = await session.list_tools()
                    openai_tools = mcp_tools_to_openai(tools_result.tools)
                    tool_definitions.extend(openai_tools)

                    self._connected.set()
                    await self._close_event.wait()
        except Exception as e:
            error_holder.append(e)
            self._connected.set()
        finally:
            self._session = None

    def call_tool(self, name: str, arguments: dict) -> str:
        """调用 MCP 工具，返回 JSON 字符串结果。"""
        if not self._call_slots.acquire(timeout=settings.mcp_call_timeout):
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "BUSY", "error": "工具服务繁忙，请稍后核实"}, ensure_ascii=False)
        try:
            return self._call_tool_once(name, arguments)
        finally:
            self._call_slots.release()

    def _call_tool_once(self, name: str, arguments: dict) -> str:
        if not self._session or not self._loop:
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "UNAVAILABLE", "error": "MCP 客户端未连接"}, ensure_ascii=False)

        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        try:
            result = future.result(timeout=settings.mcp_call_timeout)
        except FutureTimeoutError:
            future.cancel()
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "TIMEOUT", "error": "工具响应超时，结果待核实" if name == "apply_refund" else "工具响应超时"}, ensure_ascii=False)
        except Exception:
            future.cancel()
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "UNAVAILABLE", "error": "工具暂时不可用"}, ensure_ascii=False)

        if getattr(result, "isError", None) or getattr(result, "is_error", False):
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "TOOL_ERROR", "error": "工具处理失败"}, ensure_ascii=False)

        if len(result.content) != 1 or getattr(result.content[0], "type", "") != "text":
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "BAD_RESPONSE", "error": "工具返回格式异常"}, ensure_ascii=False)
        raw = result.content[0].text
        if len(raw) > 65536:
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "BAD_RESPONSE", "error": "工具结果过大，结果待核实"}, ensure_ascii=False)
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "BAD_RESPONSE", "error": "工具返回格式异常"}, ensure_ascii=False)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("success"), bool):
            return json.dumps({"success": False, "error_code": "RESULT_UNKNOWN" if name == "apply_refund" else "BAD_RESPONSE", "error": "工具返回缺少业务状态"}, ensure_ascii=False)
        return json.dumps(parsed, ensure_ascii=False)

    def close(self):
        """关闭 MCP 连接，清理后台线程。"""
        if self._close_event and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._close_event.set)
        elif self._close_event:
            self._close_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._session = None
        self._loop = None
        self._thread = None
