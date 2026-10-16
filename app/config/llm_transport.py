"""聊天模型的进程级背压；不记录密钥、提示词或上游响应正文。"""

import logging
import json
import threading
import time
from collections import deque

import httpx

from app.config.settings import settings
from app.server.middleware import request_id_var

logger = logging.getLogger(__name__)


class ModelGate:
    """跨会话共享配额。拒绝而不长时间排队，防止浏览器超时后继续积压。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.starts = deque()
        self.cooldown_until = 0.0

    def enter(self):
        with self.lock:
            now = time.monotonic()
            while self.starts and self.starts[0] <= now - 60:
                self.starts.popleft()
            if (now < self.cooldown_until or self.active >= settings.llm_max_inflight
                    or len(self.starts) >= settings.llm_requests_per_minute):
                return False
            self.active += 1
            self.starts.append(now)
            return True

    def leave(self):
        with self.lock:
            self.active -= 1

    def cool_down(self):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic()
                                      + settings.llm_rate_limit_cooldown_seconds)


model_gate = ModelGate()


class GuardedModelClient(httpx.Client):
    """在 HTTP 层覆盖路由、专家、摘要和记忆调用，包括 SDK 的 parse 路径。"""

    def __init__(self, *args, gate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate if gate is not None else model_gate

    def send(self, request, **kwargs):
        # Central policy also covers SDK parse calls and every specialist/reviewer.
        # Flash defaults to thinking too; disable it across online stages, including
        # review/repair, rather than only changing the specialist's request.
        if (request.url.host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
                and request.url.path.endswith("/chat/completions")):
            try:
                payload = json.loads(request.content)
            except (ValueError, httpx.RequestNotRead):
                payload = None
            rewrite = False
            if isinstance(payload, dict) and payload.get("model") == "qwen3.8-flash":
                payload["enable_thinking"] = False
                payload.pop("thinking_budget", None)
                rewrite = True
            elif isinstance(payload, dict) and payload.get("model") == "qwen3.8-2.4t-a95b":
                payload["enable_thinking"] = True
                configured = settings.llm_required_thinking_budget
                explicit = payload.get("thinking_budget")
                payload["thinking_budget"] = min(explicit, configured) if type(explicit) is int and explicit > 0 else configured
                rewrite = True
            if rewrite:
                headers = request.headers.copy()
                headers.pop("content-length", None)
                request = httpx.Request(request.method, request.url, headers=headers,
                                        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                        extensions=request.extensions)
        if not self.gate.enter():
            return httpx.Response(429, request=request, json={"error": {
                "message": "模型服务繁忙，请稍后重试", "type": "local_rate_limit"}})
        started = time.monotonic()
        status = "error"
        try:
            response = super().send(request, **kwargs)
            status = response.status_code
            # 非流式聊天需读完正文才释放并发名额，不能只保护响应头阶段。
            response.read()
            response.close()
            if status == 429:
                self.gate.cool_down()
            return response
        finally:
            self.gate.leave()
            logger.warning("llm_request request_id=%s status=%s elapsed_ms=%d", request_id_var.get(), status,
                           (time.monotonic() - started) * 1000)

