"""小飞客服 FastAPI 入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Response
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator

from app.agent.context_budget import ContextWindowExceeded
from app.config.settings import settings
from app.schemas.response import CustomerServiceResponse
from app.server.middleware import (
    ConcurrencyLimitMiddleware,
    RequestContextMiddleware,
    request_id_var,
)
from app.server.pool import AgentPool, AgentPoolFull, InvalidSessionId

logger = logging.getLogger(__name__)

pool = AgentPool(
    session_dir=settings.concurrent_session_dir,
    max_sessions=settings.server_max_sessions,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await pool.close_all()


app = FastAPI(
    title="小飞智能客服 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    ConcurrencyLimitMiddleware,
    max_concurrency=settings.server_max_concurrency,
    queue_timeout=settings.server_queue_timeout,
)
app.add_middleware(RequestContextMiddleware)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=10000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    result: CustomerServiceResponse


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    return {
        "status": "ready",
        "active_sessions": pool.active_sessions,
        "max_sessions": settings.server_max_sessions,
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    try:
        result = await pool.chat(
            payload.user_id, payload.session_id, payload.message
        )
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except AgentPoolFull as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "1"},
        ) from e
    except ContextWindowExceeded as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except AuthenticationError as e:
        logger.warning("模型服务鉴权失败 request_id=%s", request_id_var.get())
        raise HTTPException(
            status_code=401,
            detail="模型服务鉴权失败，请检查 OPENAI_API_KEY",
        ) from e
    except RateLimitError as e:
        logger.warning("模型服务限流 request_id=%s", request_id_var.get())
        raise HTTPException(
            status_code=429,
            detail="模型服务当前限流，请稍后重试",
            headers={"Retry-After": "2"},
        ) from e
    except APITimeoutError as e:
        logger.warning("模型服务超时 request_id=%s", request_id_var.get())
        raise HTTPException(
            status_code=504,
            detail="模型服务响应超时，请稍后重试",
        ) from e
    except APIConnectionError as e:
        logger.warning("无法连接模型服务 request_id=%s", request_id_var.get())
        raise HTTPException(
            status_code=502,
            detail=(
                "无法连接模型服务，请检查 OPENAI_BASE_URL、网络和代理配置"
            ),
        ) from e
    except APIStatusError as e:
        logger.warning(
            "模型服务返回异常状态 request_id=%s upstream_status=%s",
            request_id_var.get(),
            e.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail=f"模型服务返回异常状态（{e.status_code}），请检查模型名称和服务配置",
        ) from e

    return ChatResponse(
        request_id=request_id_var.get(),
        user_id=payload.user_id,
        session_id=payload.session_id,
        result=result,
    )


@app.post("/v1/sessions/{user_id}/{session_id}/reset", status_code=204)
async def reset_session(user_id: str, session_id: str):
    try:
        await pool.reset(user_id, session_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return Response(status_code=204)


@app.delete("/v1/sessions/{user_id}/{session_id}", status_code=204)
async def close_session(user_id: str, session_id: str):
    try:
        await pool.close_session(user_id, session_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return Response(status_code=204)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.server.api:app",
        host=settings.server_host,
        port=settings.server_port,
    )


if __name__ == "__main__":
    run()
