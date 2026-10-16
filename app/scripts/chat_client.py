"""多终端聊天客户端，每个终端用独立 user_id/session_id。"""

from __future__ import annotations

import argparse
import getpass
import json
import urllib.error
import urllib.parse
import urllib.request


def request_json(
    url: str,
    method: str,
    payload: dict | None = None,
    token: str = "",
):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        # MR-05：终端客户端同样使用登录令牌，不能只依赖可伪造的 user_id。
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return json.loads(data) if data else None


def print_skills(data: dict) -> None:
    if not data["enabled"]:
        print("技能系统未启用\n")
        return

    skills = data["skills"]
    print(f"\n--- 可供用户选择的技能（{len(skills)} 个）---")
    if skills:
        for skill in skills:
            print(f"  - {skill['name']}：{skill['description']}")
    else:
        print("  （暂无）")
    print()


def print_memory(data: dict) -> None:
    if not data["enabled"]:
        print("记忆功能未启用\n")
        return

    print("\n--- 短期记忆（本次对话）---")
    if data["short_term"]:
        for fact in data["short_term"]:
            # MR-06：CLI 同步适配结构化短期记忆响应。
            print(
                f"  - [{fact['category']}] {fact['content']} "
                f"(来源={fact['source']}, 置信度={fact['confidence']:.2f}, 状态={fact['status']})"
            )
    else:
        print("  （暂无）")

    print(f"\n--- 长期记忆（跨会话，用户: {data['user_id']}）---")
    if data["long_term"]:
        for fact in data["long_term"]:
            print(f"  - [{fact['category']}] {fact['content']}")
    else:
        print("  （暂无）")

    if data["recent_interactions"]:
        print("\n--- 最近交互 ---")
        for item in data["recent_interactions"]:
            print(f"  - {item['summary']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="小飞智能客服多用户终端")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--password", help="登录密码；省略时安全提示输入")
    parser.add_argument("--register", action="store_true", help="先注册再连接")
    parser.add_argument(
        "--enable-long-term-memory",
        action="store_true",
        help="登录后显式授权跨会话长期记忆",
    )
    args = parser.parse_args()
    base = args.server.rstrip("/")
    password = args.password or getpass.getpass("密码: ")
    auth_action = "register" if args.register else "login"
    auth_data = request_json(
        f"{base}/v1/auth/{auth_action}",
        "POST",
        {"user_id": args.user_id, "password": password},
    )
    token = auth_data["access_token"]
    if args.enable_long_term_memory:
        # MR-12：终端用户也必须主动授权，默认不会沉淀长期记忆。
        request_json(
            f"{base}/v1/users/{urllib.parse.quote(args.user_id, safe='')}/memory/consent",
            "PUT",
            {"granted": True},
            token=token,
        )
    user_id = urllib.parse.quote(args.user_id, safe="")
    session_id = urllib.parse.quote(args.session_id, safe="")
    session_url = f"{base}/v1/sessions/{user_id}/{session_id}"

    print("=" * 50)
    print("  并夕夕 · 智能客服「小飞」(ReAct + MCP + RAG)")
    print("  支持工具调用 + 政策检索 + 用户记忆 + 技能编排")
    print("  输入 quit/exit 退出, reset 重置, memory 查看记忆, skills 查看技能")
    print("  输入 memory on 授权长期记忆, memory off 撤回授权并删除长期记忆")
    print("=" * 50)
    print(f"已连接：用户={args.user_id}，会话={args.session_id}")
    print()

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
                request_json(f"{session_url}/reset", "POST", token=token)
                print("会话已重置。")
                continue

            if message.lower() == "skills":
                print_skills(request_json(f"{session_url}/skills", "GET", token=token))
                continue

            if message.lower() == "memory":
                print_memory(request_json(f"{session_url}/memory", "GET", token=token))
                continue

            if message.lower() in {"memory on", "memory off"}:
                granted = message.lower() == "memory on"
                request_json(
                    f"{base}/v1/users/{user_id}/memory/consent",
                    "PUT",
                    {"granted": granted},
                    token=token,
                )
                print("已授权长期记忆。" if granted else "已撤回授权并删除长期记忆。")
                continue

            data = request_json(
                f"{base}/v1/chat",
                "POST",
                {
                    "user_id": args.user_id,
                    "session_id": args.session_id,
                    "message": message,
                },
                token=token,
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
