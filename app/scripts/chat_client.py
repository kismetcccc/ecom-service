"""多终端聊天客户端，每个终端用独立 user_id/session_id。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def request_json(url: str, method: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return json.loads(data) if data else None


def main() -> None:
    parser = argparse.ArgumentParser(description="小飞智能客服多用户终端")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.server.rstrip("/")

    print(f"已连接小飞：用户={args.user_id}，会话={args.session_id}")
    print("输入 reset 重置会话，输入 quit/exit 退出。")
    while True:
        try:
            message = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break

        try:
            if message.lower() == "reset":
                request_json(
                    f"{base}/v1/sessions/{args.user_id}/{args.session_id}/reset",
                    "POST",
                )
                print("会话已重置。")
                continue

            data = request_json(
                f"{base}/v1/chat",
                "POST",
                {
                    "user_id": args.user_id,
                    "session_id": args.session_id,
                    "message": message,
                },
            )
            result = data["result"]
            print(f"小飞: {result['reply']}")
            print(
                f"[{result['intent']} | 置信度 {result['confidence']:.0%} | "
                f"转人工 {result['requires_human']}]\n"
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            print(f"请求失败 ({e.code}): {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"无法连接服务: {e}")


if __name__ == "__main__":
    main()
