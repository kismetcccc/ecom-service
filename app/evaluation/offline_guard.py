"""Pytest plugin: reject real HTTP transports in trusted offline test workers.

This is an accidental-network guard, NOT a security boundary for hostile code.
In-memory MockTransport and ASGI test clients remain available.
"""
import importlib


def _deny(*args, **kwargs):
    raise RuntimeError("OFFLINE_EVALUATION: real HTTP transport is disabled")


async def _deny_async(*args, **kwargs):
    _deny()


def pytest_configure(config):
    # MCP SDK versions may use httpx2. Cover both and allow loopback test servers.
    def sync_guard(original):
        def wrapped(self, request):
            if request.url.host not in {"127.0.0.1", "::1", "localhost"}:
                _deny()
            return original(self, request)
        return wrapped
    def async_guard(original):
        async def wrapped(self, request):
            if request.url.host not in {"127.0.0.1", "::1", "localhost"}:
                _deny()
            return await original(self, request)
        return wrapped
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        module.HTTPTransport.handle_request = sync_guard(module.HTTPTransport.handle_request)
        module.AsyncHTTPTransport.handle_async_request = async_guard(module.AsyncHTTPTransport.handle_async_request)
