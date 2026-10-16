"""Tool routing with identity, approved schemas and safe failure handling."""

import json
import hashlib
import logging
import time
from typing import Callable, Optional

from app.agent.tools.context import ToolContext, issue_context_token
from app.agent.tools.registry import PROTECTED_TOOLS
from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool
from app.config.settings import settings
from app.console import safe_print
from app.server.middleware import request_id_var


_logger = logging.getLogger(__name__)
_KNOWN_TOOLS = {d["function"]["name"] for d in LOCAL_TOOL_DEFINITIONS}
_SAFE_LOCAL_FALLBACK = {"query_product", "search_products", "search_knowledge"}
_TRANSIENT_ERRORS = {"TIMEOUT", "UNAVAILABLE", "BAD_RESPONSE", "TOOL_ERROR", "BUSY"}
# UNAVAILABLE 表示依赖已明确不可用（如 Qdrant 未启动），立即重试只会放大故障。
# TIMEOUT/BUSY 仍允许按配置进行一次只读重试。
_RETRYABLE_READ_ERRORS = {"TIMEOUT", "BUSY"}
_BUSINESS_ERROR_CODES = {
    "NOT_FOUND", "NOT_SHIPPED", "NOT_ELIGIBLE", "NEEDS_MANUAL_REVIEW",
    "ALREADY_PROCESSING", "CONFIRMATION_REQUIRED", "ORDER_CHANGED",
    "INVALID_ARGUMENT", "UNAUTHORIZED", "UNKNOWN_TOOL", "TOOL_ERROR",
    "TIMEOUT", "UNAVAILABLE", "BAD_RESPONSE", "RESULT_UNKNOWN", "BUSY",
    # RAG-09：保留检索的业务状态，避免 MCP 通道把它们泛化为 TOOL_ERROR。
    "INDEX_MISSING", "NO_MATCH", "LOW_RELEVANCE",
}
_LOCAL_DEFS = {d["function"]["name"]: d for d in LOCAL_TOOL_DEFINITIONS}


def _schema_compatible(remote: dict, approved: dict) -> bool:
    """Reject remote argument changes and keep approved local descriptions."""
    actual = remote.get("function", {}).get("parameters", {})
    expected = approved["function"]["parameters"]
    if set(actual.get("required", [])) != set(expected.get("required", [])):
        return False
    actual_props = actual.get("properties", {})
    expected_props = expected.get("properties", {})
    return set(actual_props) == set(expected_props) and all(
        actual_props[name].get("type") == expected_props[name].get("type")
        for name in expected_props
    )


class ToolManager:
    """聚合本地工具和 MCP 工具，提供统一的工具定义和调度接口。"""

    def __init__(
        self,
        use_mcp: bool = False,
        mcp_server_url: str = "",
        allowed_tools: Optional[set] = None,
        disabled_tools: Optional[set] = None,
        local_tool_overrides: Optional[dict[str, Callable]] = None,
        user_id: str = "",
        session_id: str = "default",
    ):
        self._mcp_client = None
        self._tool_source: dict[str, str] = {}
        self._tool_defs: list[dict] = []
        self._local_tool_overrides = local_tool_overrides or {}
        self._context = ToolContext(user_id, session_id) if user_id else None
        self._use_mcp = use_mcp
        self._server_url = mcp_server_url
        self._failures = 0
        self._retry_at = 0.0
        self._allowed_tools = allowed_tools
        self._disabled_tools = disabled_tools

        if use_mcp and mcp_server_url:
            self._init_mcp(mcp_server_url)
        else:
            self._init_local()

        if allowed_tools is not None:
            self._filter_tools(allowed_tools)
        if disabled_tools:
            self._exclude_tools(disabled_tools)

    def _init_local(self):
        """只加载本地工具。"""
        self._tool_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tool_defs:
            self._tool_source[td["function"]["name"]] = "local"

    def _init_mcp(self, server_url: str):
        """Discover an approved catalog, sharing one transport across sessions."""
        try:
            # MCP is optional: local-only chat must not import its transport dependency.
            from app.mcp_client.pool import acquire_client, release_client

            self._mcp_client, discovered = acquire_client(server_url)
            mcp_tools = [d for d in discovered if d["function"]["name"] in _KNOWN_TOOLS and _schema_compatible(d, _LOCAL_DEFS[d["function"]["name"]])]
            if not mcp_tools:
                raise RuntimeError("MCP 工具目录不兼容")
            safe_print(f"🔗 [MCP] 已连接 {server_url}，发现 {len(mcp_tools)} 个工具")

            mcp_names = set()
            for td in mcp_tools:
                name = td["function"]["name"]
                mcp_names.add(name)
                self._tool_source[name] = "mcp"
            self._tool_defs = [_LOCAL_DEFS[name] for name in mcp_names]

            for td in LOCAL_TOOL_DEFINITIONS:
                name = td["function"]["name"]
                if name not in mcp_names and name not in PROTECTED_TOOLS:
                    self._tool_defs.append(td)
                    self._tool_source[name] = "local"

        except Exception as e:
            _logger.warning("MCP connection failed: %s", type(e).__name__)
            safe_print("⚠️  [MCP] 连接失败，受保护的业务工具暂不可用")
            if self._mcp_client:
                release_client(server_url, self._mcp_client)
                self._mcp_client = None
            self._init_local()
            self._hide_protected_when_unavailable()
            self._retry_at = time.monotonic() + settings.mcp_recovery_seconds

    def _hide_protected_when_unavailable(self) -> None:
        self._tool_defs = [d for d in self._tool_defs if d["function"]["name"] not in PROTECTED_TOOLS]
        self._tool_source = {name: source for name, source in self._tool_source.items() if name not in PROTECTED_TOOLS}

    def _filter_tools(self, allowed: set):
        """只保留白名单中的工具，用于子 Agent 工具隔离。"""
        self._tool_defs = [
            d for d in self._tool_defs
            if d["function"]["name"] in allowed
        ]
        self._tool_source = {
            k: v for k, v in self._tool_source.items()
            if k in allowed
        }

    def _exclude_tools(self, disabled: set):
        """移除因功能开关关闭而不可用的工具，避免模型调用必然失败的能力。"""
        self._tool_defs = [
            d for d in self._tool_defs
            if d["function"]["name"] not in disabled
        ]
        self._tool_source = {
            k: v for k, v in self._tool_source.items()
            if k not in disabled
        }

    @property
    def tool_definitions(self) -> list[dict]:
        return self._tool_defs

    def available_tool_names(self) -> set[str]:
        """MA-04：计划前刷新熔断后的 MCP 目录，按实际可用工具校验能力。"""
        self._maybe_reconnect()
        return {item["function"]["name"] for item in self._tool_defs}

    def execute_tool(self, name: str, arguments: dict) -> str:
        """根据工具来源分发调用。"""
        self._maybe_reconnect()
        if self._use_mcp and name in PROTECTED_TOOLS and name not in self._tool_source:
            return self._error("UNAVAILABLE", "订单或售后服务暂时不可用，请稍后重试")
        if not isinstance(arguments, dict) or name not in self._tool_source:
            return self._error("INVALID_ARGUMENT", "工具名称或参数无效")
        if "auth_token" in arguments or "context" in arguments:
            return self._error("INVALID_ARGUMENT", "工具参数包含保留字段")
        source = self._tool_source.get(name)
        definition = next((d["function"] for d in self._tool_defs if d["function"]["name"] == name), None)
        if definition is None or not self._valid_arguments(arguments, definition["parameters"]):
            return self._error("INVALID_ARGUMENT", "工具参数不符合契约")

        if self._use_mcp and self._failures >= settings.mcp_failure_threshold and time.monotonic() < self._retry_at:
            if settings.mcp_allow_local_read_fallback and name in _SAFE_LOCAL_FALLBACK:
                return local_execute_tool(name, arguments, context=self._context)
            return self._error("RESULT_UNKNOWN" if name == "apply_refund" else "UNAVAILABLE", "工具暂时不可用，请稍后核实")

        if source == "mcp" and self._mcp_client:
            if name in PROTECTED_TOOLS:
                if self._context is None:
                    return self._error("UNAUTHORIZED", "请先登录")
                try:
                    token = issue_context_token(self._context)
                except Exception:
                    _logger.error("MCP identity context unavailable")
                    return self._error("UNAVAILABLE", "身份校验服务暂时不可用")
                arguments = {**arguments, "auth_token": token}
            start = time.monotonic()
            result = self._mcp_client.call_tool(name, arguments)
            data = self._parse_result(result, name)
            error_code = data.get("error_code")
            if name != "apply_refund" and error_code in _RETRYABLE_READ_ERRORS:
                for _ in range(max(0, settings.mcp_read_retries)):
                    result = self._mcp_client.call_tool(name, arguments)
                    data = self._parse_result(result, name)
                    error_code = data.get("error_code")
                    if error_code not in _RETRYABLE_READ_ERRORS:
                        break
            if error_code in _TRANSIENT_ERRORS or error_code == "RESULT_UNKNOWN":
                self._failures += 1
                if self._failures >= settings.mcp_failure_threshold:
                    self._retry_at = time.monotonic() + settings.mcp_recovery_seconds
                if settings.mcp_allow_local_read_fallback and name in _SAFE_LOCAL_FALLBACK and error_code in _TRANSIENT_ERRORS:
                    result = local_execute_tool(name, {k: v for k, v in arguments.items() if k != "auth_token"})
                    # 2026-10-10: return the actual fallback, not the old remote error.
                    data = {**self._parse_result(result, name), "mcp_read_fallback": True}
            else:
                self._failures = 0
            user_hash = hashlib.sha256(self._context.user_id.encode()).hexdigest()[:12] if self._context else ""
            _logger.info(
                "mcp_tool request_id=%s user_hash=%s name=%s elapsed_ms=%d error_code=%s",
                request_id_var.get(), user_hash, name,
                int((time.monotonic() - start) * 1000), error_code or "",
            )
            return json.dumps(data, ensure_ascii=False) if error_code else result

        if self._use_mcp and name in PROTECTED_TOOLS:
            return self._error("UNAVAILABLE", "订单或售后服务暂时不可用，请稍后重试")

        if source == "local" and name in self._local_tool_overrides:
            try:
                if name == "load_skill":
                    result = self._local_tool_overrides[name](**arguments, available_tools=set(self._tool_source))
                else:
                    result = self._local_tool_overrides[name](**arguments)
            except Exception:
                result = {"success": False, "error_code": "TOOL_ERROR", "error": "工具暂时不可用"}
            return json.dumps(result, ensure_ascii=False)

        if source == "local":
            return local_execute_tool(name, arguments, context=self._context)

        return self._error("UNKNOWN_TOOL", "未知工具")

    @staticmethod
    def _error(code: str, message: str) -> str:
        return json.dumps({"success": False, "error_code": code, "error": message}, ensure_ascii=False)

    @staticmethod
    def _parse_result(raw: str, name: str) -> dict:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
            code = "RESULT_UNKNOWN" if name == "apply_refund" else "BAD_RESPONSE"
            return {"success": False, "error_code": code, "error": "工具返回格式异常"}
        if data["success"] is False:
            code = data.get("error_code")
            if code not in _BUSINESS_ERROR_CODES:
                return {"success": False, "error_code": "TOOL_ERROR", "error": "工具处理失败"}
            data["error"] = str(data.get("error", "工具处理失败"))[:200]
        return data

    @staticmethod
    def _valid_arguments(arguments: dict, schema: dict) -> bool:
        properties = schema.get("properties", {})
        if any(key not in properties for key in arguments):
            return False
        if any(key not in arguments for key in schema.get("required", [])):
            return False
        for key, value in arguments.items():
            expected = properties[key].get("type")
            if expected == "string" and (not isinstance(value, str) or len(value) > 500):
                return False
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return False
            if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                return False
            if key == "top_k" and not 1 <= value <= 5:
                return False
            if key == "limit" and not 1 <= value <= 20:
                return False
        return True

    def _maybe_reconnect(self) -> None:
        if not self._use_mcp or not self._server_url or time.monotonic() < self._retry_at:
            return
        if self._mcp_client and self._failures < settings.mcp_failure_threshold:
            return
        if self._mcp_client:
            from app.mcp_client.pool import invalidate_client

            invalidate_client(self._server_url, self._mcp_client)
        self._mcp_client = None
        self._init_mcp(self._server_url)
        if self._allowed_tools is not None:
            self._filter_tools(self._allowed_tools)
        if self._disabled_tools:
            self._exclude_tools(self._disabled_tools)
        self._failures = 0

    def close(self):
        """清理 MCP 连接。"""
        if self._mcp_client:
            from app.mcp_client.pool import release_client

            release_client(self._server_url, self._mcp_client)
            self._mcp_client = None
