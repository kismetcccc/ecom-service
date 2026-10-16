"""持续压测 /v1/chat，并统计延迟、错误率和服务资源占用。

示例（会真实调用模型并产生费用）：
    python tests/soak_test.py --duration 300 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

try:
    import psutil
except ImportError:  # pragma: no cover - 仅影响可选的资源监控
    psutil = None


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat"


@dataclass
class Results:
    latencies: list[float] = field(default_factory=list)
    statuses: Counter[int | str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)


@dataclass
class ResourceSamples:
    pid: int | None = None
    cpu_percent: list[float] = field(default_factory=list)
    memory_mb: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="持续压测小飞客服 API")
    parser.add_argument("--url", default=DEFAULT_URL, help="聊天接口地址")
    parser.add_argument(
        "--duration",
        type=float,
        default=300,
        help="持续时间（秒），默认 300 秒",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="并发客户端数，默认 32",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="单次请求超时（秒），默认 60 秒",
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Uvicorn 进程 PID；省略时根据 URL 端口自动寻找",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10,
        help="进度输出间隔（秒），默认 10 秒",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.concurrency <= 0:
        parser.error("--concurrency 必须大于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.progress_interval <= 0:
        parser.error("--progress-interval 必须大于 0")
    return args


def percentile(values: list[float], percent: float) -> float:
    """使用线性插值计算百分位。"""
    if not values:
        return 0.0
    position = (len(values) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def find_listening_pid(port: int) -> int | None:
    if psutil is None:
        return None
    try:
        for connection in psutil.net_connections(kind="inet"):
            if (
                connection.status == psutil.CONN_LISTEN
                and connection.laddr
                and connection.laddr.port == port
                and connection.pid
            ):
                return connection.pid
    except (psutil.AccessDenied, OSError):
        return None
    return None


def process_family(root: "psutil.Process") -> list["psutil.Process"]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


async def monitor_resources(
    samples: ResourceSamples,
    stop_event: asyncio.Event,
) -> None:
    if psutil is None or samples.pid is None:
        return

    try:
        root = psutil.Process(samples.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    tracked: dict[int, "psutil.Process"] = {}

    while not stop_event.is_set():
        family = process_family(root)
        for process in family:
            if process.pid not in tracked:
                tracked[process.pid] = process
                try:
                    process.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        cpu_total = 0.0
        memory_total = 0
        for process in list(tracked.values()):
            try:
                cpu_total += process.cpu_percent(interval=None)
                memory_total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                tracked.pop(process.pid, None)

        samples.cpu_percent.append(cpu_total)
        samples.memory_mb.append(memory_total / 1024 / 1024)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1)
        except TimeoutError:
            pass


async def worker(
    worker_id: int,
    client: httpx.AsyncClient,
    url: str,
    deadline: float,
    run_id: str,
    results: Results,
) -> None:
    request_number = 0
    while time.perf_counter() < deadline:
        payload = {
            "user_id": f"soak-{run_id}-{worker_id}",
            "session_id": "main",
            "message": "你好，请只回复收到",
        }
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
            elapsed = time.perf_counter() - started
            results.latencies.append(elapsed)
            results.statuses[response.status_code] += 1
        except Exception as exc:
            elapsed = time.perf_counter() - started
            results.latencies.append(elapsed)
            results.statuses["error"] += 1
            results.errors[type(exc).__name__] += 1
        request_number += 1


async def show_progress(
    results: Results,
    started: float,
    stop_event: asyncio.Event,
    interval: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except TimeoutError:
            elapsed = time.perf_counter() - started
            completed = sum(results.statuses.values())
            print(
                f"进度: {elapsed:.0f}秒, 完成={completed}, "
                f"实时平均吞吐量={completed / elapsed:.2f} 请求/秒"
            )


async def run(args: argparse.Namespace) -> None:
    port = urlparse(args.url).port or 80
    pid = args.pid or find_listening_pid(port)
    resources = ResourceSamples(pid=pid)
    results = Results()
    stop_event = asyncio.Event()
    run_id = uuid.uuid4().hex[:8]

    print("===== 持续压测配置 =====")
    print(f"接口: {args.url}")
    print(f"持续时间: {args.duration:.0f} 秒")
    print(f"并发数: {args.concurrency}")
    if pid is None:
        print("服务 PID: 未找到（本轮不采集 CPU/内存）")
    else:
        print(f"服务 PID: {pid}")

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    started = time.perf_counter()
    deadline = started + args.duration

    monitor_task = asyncio.create_task(monitor_resources(resources, stop_event))
    progress_task = asyncio.create_task(
        show_progress(results, started, stop_event, args.progress_interval)
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(
            *(
                worker(worker_id, client, args.url, deadline, run_id, results)
                for worker_id in range(args.concurrency)
            )
        )

    elapsed = time.perf_counter() - started
    stop_event.set()
    await asyncio.gather(monitor_task, progress_task)

    latencies = sorted(results.latencies)
    total = sum(results.statuses.values())
    successes = sum(
        count
        for status, count in results.statuses.items()
        if isinstance(status, int) and 200 <= status < 300
    )
    failures = total - successes
    error_rate = failures / total * 100 if total else 0.0

    print("\n===== 持续压测结果 =====")
    print(f"实际耗时: {elapsed:.2f} 秒")
    print(f"总请求数: {total}")
    print(f"成功请求数: {successes}")
    print(f"状态码统计: {dict(results.statuses)}")
    if results.errors:
        print(f"客户端异常: {dict(results.errors)}")
    print(f"错误率: {error_rate:.2f}%")
    print(f"吞吐量: {total / elapsed:.2f} 请求/秒")
    print(f"平均延迟: {sum(latencies) / len(latencies):.2f} 秒")
    print(f"P50 延迟: {percentile(latencies, 0.50):.2f} 秒")
    print(f"P95 延迟: {percentile(latencies, 0.95):.2f} 秒")
    print(f"P99 延迟: {percentile(latencies, 0.99):.2f} 秒")
    print(f"最大延迟: {max(latencies):.2f} 秒")

    if resources.cpu_percent and resources.memory_mb:
        # 第一个 CPU 样本仅用于预热 psutil，排除后统计更准确。
        cpu = resources.cpu_percent[1:] or resources.cpu_percent
        memory = resources.memory_mb
        print(f"服务 CPU 平均占用: {sum(cpu) / len(cpu):.1f}%")
        print(f"服务 CPU 峰值占用: {max(cpu):.1f}%")
        print(f"服务内存平均占用: {sum(memory) / len(memory):.1f} MB")
        print(f"服务内存峰值占用: {max(memory):.1f} MB")
    else:
        print("CPU/内存: 未采集；可使用 --pid 指定 Uvicorn 进程")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
