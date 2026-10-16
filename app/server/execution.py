"""同步任务与请求取消的共同生命周期。"""

import asyncio


async def run_blocking(function, *args):
    """CR-13：取消后等线程真正结束，外围锁/限流槽随后才可释放。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()
