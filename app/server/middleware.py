"""HTTP 中间件：请求追踪、并发限制和过载背压。"""

from __future__ import annotations

import asyncio
import contextvars
import re
import time
import uuid

from starlette.responses import JSONResponse

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


class RequestContextMiddleware:
    """为每个请求生成追踪 ID，并返回服务耗时。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("ascii", "ignore")
        request_id = (
            supplied[:128]
            if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", supplied)
            else uuid.uuid4().hex
        )
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode()))
                elapsed_ms = (time.perf_counter() - started) * 1000
                response_headers.append(
                    (b"x-process-time-ms", f"{elapsed_ms:.2f}".encode())
                )
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            request_id_var.reset(token)


class ConcurrencyLimitMiddleware:
    """限制同时执行的聊天请求；队列超时后快速返回 503。"""

    def __init__(self, app, max_concurrency: int, queue_timeout: float):
        self.app = app
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.queue_timeout = max(0.01, queue_timeout)
        self.in_flight = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/chat":
            await self.app(scope, receive, send)
            return

        try:
            await asyncio.wait_for(
                self.semaphore.acquire(), timeout=self.queue_timeout
            )
        except TimeoutError:
            response = JSONResponse(
                {
                    "detail": "服务繁忙，请稍后重试",
                    "request_id": request_id_var.get(),
                },
                status_code=503,
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        self.in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.in_flight -= 1
            self.semaphore.release()
