"""Supervise local MCP and API together; either child exiting stops both."""
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


def health():
    from app.mcp_client import MCPClient
    with urlopen("http://127.0.0.1:8000/health/ready", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("API not ready")
    client = MCPClient("http://127.0.0.1:9123/mcp")
    try:
        if not client.connect():
            raise RuntimeError("MCP tools unavailable")
    finally:
        client.close()


def main():
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "mcp_server/server.py"]))
        # Wait for the actual MCP handshake, not just a listening TCP socket.
        from app.mcp_client import MCPClient
        for _ in range(20):
            if stopping or children[0].poll() is not None:
                return 1
            client = MCPClient("http://127.0.0.1:9123/mcp")
            try:
                if client.connect():
                    break
            except Exception:
                time.sleep(1)
            finally:
                client.close()
        else:
            raise RuntimeError("MCP startup timed out")
        children.append(subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app.server.api:app",
            "--host", "0.0.0.0", "--port", "8000",
        ]))
        while not stopping:
            if any(child.poll() is not None for child in children):
                return 1
            time.sleep(.5)
        return 0
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    if "--health" in sys.argv:
        health()
    else:
        raise SystemExit(main())
