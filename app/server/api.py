"""小飞客服 FastAPI 入口。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator

from app.agent.context_budget import ContextWindowExceeded
from app.agent.storage import SessionCorruptedError
from app.commands import is_control_command
from app.config.settings import settings
from app.schemas.response import CustomerServiceResponse
from app.server.middleware import (
    AuthenticationLimitMiddleware,
    ConcurrencyLimitMiddleware,
    RequestContextMiddleware,
    request_id_var,
)
from app.server.pool import AgentPool, AgentPoolFull, InvalidSessionId
from app.server.auth import AuthError, AuthService
from app.server.execution import run_blocking

logger = logging.getLogger(__name__)
FRONTEND_PATH = Path(__file__).with_name("static") / "index.html"

pool = AgentPool(
    session_dir=settings.concurrent_session_dir,
    max_sessions=settings.server_max_sessions,
)
auth_service = AuthService(
    database_path=settings.auth_database_path,
    secret_key=settings.auth_secret_key,
    token_ttl_seconds=settings.auth_token_ttl_seconds,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def maintain_sessions():
        while True:
            await asyncio.sleep(60)
            try:
                await pool.reap_idle(settings.server_session_idle_seconds)
            except Exception:
                logger.exception("空闲会话回收失败")

    maintenance = asyncio.create_task(maintain_sessions())
    try:
        yield
    finally:
        maintenance.cancel()
        await asyncio.gather(maintenance, return_exceptions=True)
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
app.add_middleware(
    AuthenticationLimitMiddleware,
    max_concurrency=settings.auth_max_concurrency,
    attempts_per_minute=settings.auth_attempts_per_minute,
)


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
        if is_control_command(value):
            raise ValueError(
                f"{value.casefold()} 是客户端控制命令，不能作为聊天消息提交"
            )
        return value


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    result: CustomerServiceResponse


class RefundPreviewRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)
    workflow_id: str = Field(default="", pattern=r"^(?:[a-f0-9]{32})?$", max_length=32)


class RefundConfirmRequest(BaseModel):
    preview_token: str = Field(min_length=1, max_length=256)


class MemoryOperationResponse(BaseModel):
    """MR-04：所有记忆清理操作明确返回实际作用范围。"""

    user_id: str
    session_id: str | None = None
    scope: str
    deleted: bool


class CredentialsRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class AuthResponse(BaseModel):
    user_id: str
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MemoryConsentRequest(BaseModel):
    granted: bool


class MemoryFactUpdateRequest(BaseModel):
    """MR-17：用户可纠正事实正文或分类，至少提供一项。"""

    content: str | None = Field(default=None, min_length=1, max_length=500)
    category: Literal[
        "identity", "preference", "intent", "emotion", "order",
        "behavior", "issue", "other",
    ] | None = None


def require_authenticated_user(
    authorization: str | None = Header(default=None),
) -> str:
    """MR-05：只信任签名令牌中的用户，不把请求中的 user_id 当作身份凭证。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    try:
        return auth_service.verify_token(authorization[7:].strip())
    except AuthError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


from app.server.shopping import shopping_router

app.include_router(shopping_router(require_authenticated_user))


def enforce_user_scope(authenticated_user: str, requested_user: str) -> None:
    """MR-05：阻止已登录用户枚举、读取或污染其他 user_id 的数据。"""
    if authenticated_user != requested_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户的数据")


@app.get("/v1/auth/me")
async def current_identity(authenticated_user: str = Depends(require_authenticated_user)):
    return {"user_id": authenticated_user}


@app.post("/v1/auth/logout", status_code=204)
async def logout(
    authenticated_user: str = Depends(require_authenticated_user),
    authorization: str = Header(),
):
    try:
        await asyncio.to_thread(auth_service.revoke_token, authorization[7:].strip())
    except AuthError:
        # 并发退出请求保持幂等；身份依赖已在入口处校验签名。
        pass
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(
        FRONTEND_PATH,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/quality", include_in_schema=False)
async def quality_dashboard():
    # 2026-10-10: static local-import UI; no server report or session file access.
    return FileResponse(FRONTEND_PATH.with_name("quality.html"), media_type="text/html", headers={
        "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'",
    })


@app.post("/v1/auth/register", response_model=AuthResponse, status_code=201)
async def register(payload: CredentialsRequest):
    try:
        token = await run_blocking(auth_service.register, payload.user_id, payload.password)
    except AuthError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return AuthResponse(
        user_id=payload.user_id,
        access_token=token,
        expires_in=settings.auth_token_ttl_seconds,
    )


@app.post("/v1/auth/login", response_model=AuthResponse)
async def login(payload: CredentialsRequest):
    try:
        token = await run_blocking(auth_service.login, payload.user_id, payload.password)
    except AuthError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    return AuthResponse(
        user_id=payload.user_id,
        access_token=token,
        expires_in=settings.auth_token_ttl_seconds,
    )


def _skill_snapshot(agent) -> dict:
    manager = getattr(agent, "skill_manager", None)
    enabled = bool(manager and manager.enabled)
    return {
        "enabled": enabled,
        "skills": manager.get_catalog() if enabled else [],
    }


def _memory_snapshot(agent) -> dict:
    manager = getattr(agent, "memory_manager", None)
    enabled = bool(manager and manager.memory_enabled)
    if not enabled:
        return {
            "enabled": False,
            "short_term": [],
            "user_id": None,
            "long_term": [],
            "recent_interactions": [],
            "long_term_consent": False,
        }

    # MR-08：管理页面读取前也刷新生命周期，使 UI 不会把过期事实显示为有效。
    getattr(manager.stm, "active_facts", None)
    refresh_lifecycle = getattr(manager.ltm, "refresh_lifecycle", None)
    if callable(refresh_lifecycle):
        refresh_lifecycle()
    return {
        "enabled": True,
        # MR-06：API 返回完整结构化事实，前端可展示来源、类别与状态。
        "short_term": list(manager.stm.to_dict()["facts"]),
        "user_id": manager.ltm.user_id,
        "long_term": [
            {
                "id": fact.id,
                "category": fact.category,
                "content": fact.content,
                "memory_key": fact.memory_key,
                "status": fact.status,
                "superseded_by": fact.superseded_by,
                # MR-11：管理界面可追溯每条长期事实的来源和证据。
                "source": fact.source,
                "source_session": fact.source_session,
                "source_message_id": fact.source_message_id,
                "evidence": fact.evidence,
                "extraction_model": fact.extraction_model,
                "confidence": fact.confidence,
                "last_verified_at": fact.last_verified_at,
            }
            for fact in manager.ltm.facts
        ] if manager.long_term_consent else [],
        "recent_interactions": (
            list(manager.ltm.interaction_summaries[-3:])
            if manager.long_term_consent else []
        ),
        "long_term_consent": manager.long_term_consent,
        # MR-17：导出与管理视图同时返回人工操作审计。
        "short_term_audit": list(getattr(manager.stm, "audit", [])),
        "long_term_audit": (
            list(getattr(manager.ltm, "memory_audit", []))
            if manager.long_term_consent else []
        ),
    }


def _summary_snapshot(agent) -> dict:
    """MR-02：返回当前摘要、版本历史及回滚/删除审计。"""
    versions = list(getattr(agent, "summary_versions", []))
    current_summary = getattr(agent, "summary", None)
    current_version_id = next(
        (
            item["version_id"] for item in reversed(versions)
            if item["summary"] == current_summary
        ),
        None,
    )
    return {
        "current_summary": current_summary,
        "current_version_id": current_version_id,
        "versions": versions,
        "audit": list(getattr(agent, "summary_audit", [])),
    }


def _memory_metrics_snapshot(agent) -> dict:
    """MR-18：只返回聚合指标与无正文事件，不暴露记忆内容。"""
    manager = getattr(agent, "memory_manager", None)
    if manager is None or not manager.memory_enabled:
        return {"enabled": False}
    return {"enabled": True, **manager.metrics_snapshot()}


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    return {
        "status": "ready",
        "dependency_checks": "not_performed",
        "active_sessions": pool.active_sessions,
        "max_sessions": settings.server_max_sessions,
        "capabilities": {"demo_refunds": settings.seed_demo_orders, "simulated_purchases": True},
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, payload.user_id)
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
    except SessionCorruptedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except AuthenticationError as e:
        logger.warning("模型服务鉴权失败 request_id=%s", request_id_var.get())
        raise HTTPException(
            status_code=502,
            detail="客服模型服务暂不可用，请稍后重试或联系管理员",
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
    except Exception as e:
        # Keep internal errors in server logs; give the UI a request ID for diagnosis.
        request_id = request_id_var.get()
        logger.exception("聊天请求异常 request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"客服服务暂时不可用，请稍后重试（请求编号：{request_id}）",
        ) from e

    return ChatResponse(
        request_id=request_id_var.get(),
        user_id=payload.user_id,
        session_id=payload.session_id,
        result=result,
    )


@app.post("/v1/refunds/preview")
async def refund_preview(
    payload: RefundPreviewRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MCP-03: preview is read-only and never submits a refund."""
    from app.agent.tools.order_store import get_order_store

    result = await asyncio.to_thread(get_order_store().preview_refund, authenticated_user, payload.order_id, payload.reason,
                                     origin_workflow_id=payload.workflow_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/v1/refunds/confirm")
async def refund_confirm(
    payload: RefundConfirmRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MCP-02/03: explicit UI confirmation, then one idempotent transaction."""
    from app.agent.tools.order_store import get_order_store

    store = get_order_store()
    existing = await asyncio.to_thread(store.refund_status, authenticated_user, "", payload.preview_token)
    if existing["success"]:
        return existing
    confirmed = await asyncio.to_thread(store.confirm_preview, authenticated_user, payload.preview_token)
    if not confirmed["success"]:
        raise HTTPException(status_code=409, detail=confirmed["error"])
    result = await asyncio.to_thread(store.apply_refund, authenticated_user, payload.preview_token, payload.preview_token)
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@app.post("/v1/refunds/resolve")
async def resolve_refund(
    payload: RefundConfirmRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """CR-19：丢失提交响应时按原凭据只读核对，不创建第二笔申请。"""
    from app.agent.tools.order_store import get_order_store
    result = await asyncio.to_thread(
        get_order_store().refund_status, authenticated_user, "", payload.preview_token,
    )
    if not result["success"]:
        raise HTTPException(status_code=404, detail="尚未查询到本次申请，请稍后核实或联系人工；请勿重复新建申请")
    return result


@app.get("/v1/refunds/{application_id}")
async def refund_status(
    application_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    from app.agent.tools.order_store import get_order_store

    result = await asyncio.to_thread(get_order_store().refund_status, authenticated_user, application_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/v1/sessions/{user_id}/{session_id}/skills")
async def get_skills(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    try:
        return await pool.inspect(user_id, session_id, _skill_snapshot)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except AgentPoolFull as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "1"},
        ) from e


@app.get("/v1/sessions/{user_id}/{session_id}/memory")
async def get_memory(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    try:
        return await pool.inspect(user_id, session_id, _memory_snapshot)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except AgentPoolFull as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "1"},
        ) from e


@app.get("/v1/sessions/{user_id}/{session_id}/memory/metrics")
async def get_memory_metrics(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-18：用户只能查看自己的 Memory 成本、质量和存储指标。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        return await pool.inspect(user_id, session_id, _memory_metrics_snapshot)
    except InvalidSessionId as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AgentPoolFull as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


async def _mutate_memory_fact(
    user_id: str,
    session_id: str,
    scope: str,
    fact_id: str,
    action: str,
    actor: str,
    payload: MemoryFactUpdateRequest | None = None,
) -> dict:
    """MR-17：共享鉴权后的单条事实变更错误处理。"""
    if scope not in {"short-term", "long-term"}:
        raise HTTPException(status_code=422, detail="scope 必须是 short-term 或 long-term")
    if action == "update" and payload is not None:
        if payload.content is None and payload.category is None:
            raise HTTPException(status_code=422, detail="content 和 category 至少提供一项")
        content = payload.content.strip() if payload.content is not None else None
        category = payload.category
    else:
        content = category = None
    try:
        fact = await pool.mutate_memory_fact(
            user_id, session_id, scope, action, fact_id, actor,
            content, category,
        )
    except InvalidSessionId as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        status = 404 if "不存在" in str(error) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    return {"updated": True, "scope": scope, "action": action, "fact": fact}


@app.patch("/v1/sessions/{user_id}/{session_id}/memory/{scope}/{fact_id}")
async def update_memory_fact(
    user_id: str,
    session_id: str,
    scope: str,
    fact_id: str,
    payload: MemoryFactUpdateRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    return await _mutate_memory_fact(
        user_id, session_id, scope, fact_id, "update",
        authenticated_user, payload,
    )


@app.delete("/v1/sessions/{user_id}/{session_id}/memory/{scope}/{fact_id}")
async def delete_memory_fact(
    user_id: str,
    session_id: str,
    scope: str,
    fact_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    return await _mutate_memory_fact(
        user_id, session_id, scope, fact_id, "delete", authenticated_user,
    )


@app.post("/v1/sessions/{user_id}/{session_id}/memory/{scope}/{fact_id}/confirm")
async def confirm_memory_fact(
    user_id: str,
    session_id: str,
    scope: str,
    fact_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    return await _mutate_memory_fact(
        user_id, session_id, scope, fact_id, "confirm", authenticated_user,
    )


@app.get("/v1/sessions/{user_id}/{session_id}/summaries")
async def get_summary_versions(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-02：查看当前用户会话的摘要版本和审计记录。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        return await pool.inspect(user_id, session_id, _summary_snapshot)
    except (InvalidSessionId, AgentPoolFull) as error:
        status = 422 if isinstance(error, InvalidSessionId) else 503
        raise HTTPException(status_code=status, detail=str(error)) from error


@app.post(
    "/v1/sessions/{user_id}/{session_id}/summaries/{version_id}/rollback"
)
async def rollback_summary_version(
    user_id: str,
    session_id: str,
    version_id: int,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-02：回滚摘要并将登录用户写入审计。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        version = await pool.rollback_summary(
            user_id, session_id, version_id, authenticated_user
        )
    except (InvalidSessionId, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"rolled_back": True, "version": version}


@app.delete(
    "/v1/sessions/{user_id}/{session_id}/summaries/{version_id}"
)
async def delete_summary_version(
    user_id: str,
    session_id: str,
    version_id: int,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-02：删除非当前摘要版本并留下审计记录。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        await pool.delete_summary_version(
            user_id, session_id, version_id, authenticated_user
        )
    except (InvalidSessionId, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"deleted": True, "version_id": version_id}


@app.post(
    "/v1/sessions/{user_id}/{session_id}/reset",
    response_model=MemoryOperationResponse,
)
async def reset_session(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
    try:
        deleted = await pool.reset(user_id, session_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return MemoryOperationResponse(
        user_id=user_id,
        session_id=session_id,
        scope="session_context_and_short_term",
        deleted=deleted,
    )


@app.delete(
    "/v1/sessions/{user_id}/{session_id}/memory/short-term",
    response_model=MemoryOperationResponse,
)
async def clear_short_term_memory(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-04：短期记忆使用独立接口，避免和长期记忆删除混淆。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        deleted = await pool.clear_short_term(user_id, session_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return MemoryOperationResponse(
        user_id=user_id,
        session_id=session_id,
        scope="short_term_memory",
        deleted=deleted,
    )


@app.put("/v1/users/{user_id}/memory/consent")
async def set_memory_consent(
    user_id: str,
    payload: MemoryConsentRequest,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-12：长期记忆必须由当前登录用户显式授权并可随时撤回。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        await pool.set_long_term_consent(user_id, payload.granted)
    except InvalidSessionId as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "user_id": user_id,
        "long_term_consent": payload.granted,
        "long_term_deleted": not payload.granted,
    }


@app.delete(
    "/v1/users/{user_id}/memory",
    response_model=MemoryOperationResponse,
)
async def delete_long_term_memory(
    user_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    """MR-04：长期记忆删除是用户级操作，必须与会话重置分离。"""
    enforce_user_scope(authenticated_user, user_id)
    try:
        deleted = await pool.delete_long_term(user_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return MemoryOperationResponse(
        user_id=user_id,
        scope="long_term_memory",
        deleted=deleted,
    )


@app.delete("/v1/sessions/{user_id}/{session_id}", status_code=204)
async def close_session(
    user_id: str,
    session_id: str,
    authenticated_user: str = Depends(require_authenticated_user),
):
    enforce_user_scope(authenticated_user, user_id)
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
