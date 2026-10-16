"""对本地 /v1/chat 接口执行简单的真实并发压测。

运行前请先启动 API 服务：
    uvicorn app.server.api:app --host 0.0.0.0 --port 8000

然后在项目根目录执行：
    python tests/load_test.py

注意：每个请求都会真实调用模型并消耗 API 额度。
"""

import asyncio
import time
from collections import Counter

import httpx


URL = "http://127.0.0.1:8000/v1/chat"
TOTAL_REQUESTS = 100
CONCURRENCY =32


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int | str, float]:
    payload = {
        # 使用不同用户，避免同一用户的请求被服务端串行处理。
        "user_id": f"load-user-{index}",
        "session_id": "main",
        "message": "你好，请只回复收到",
    }

    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(URL, json=payload)
            elapsed = time.perf_counter() - started
            print(
                f"请求 {index:02d}: 状态码={response.status_code}, "
                f"耗时={elapsed:.2f}秒"
            )
            return response.status_code, elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"请求 {index:02d}: 失败={exc}")
            return "error", elapsed


async def main() -> None:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=60) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                send_request(client, semaphore, index)
                for index in range(TOTAL_REQUESTS)
            )
        )
        total_time = time.perf_counter() - started

    statuses = Counter(status for status, _ in results)
    latencies = sorted(elapsed for _, elapsed in results)

    print("\n===== 测试结果 =====")
    print(f"总请求数: {TOTAL_REQUESTS}")
    print(f"并发数: {CONCURRENCY}")
    print(f"状态码统计: {dict(statuses)}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"吞吐量: {TOTAL_REQUESTS / total_time:.2f} 请求/秒")
    print(f"平均延迟: {sum(latencies) / len(latencies):.2f} 秒")
    print(f"最大延迟: {max(latencies):.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
