"""统一创建 OpenAI 兼容客户端。"""

from openai import OpenAI

from app.config.settings import settings
from app.config.llm_transport import GuardedModelClient


def create_openai_client() -> OpenAI:
    """创建聊天客户端，并显式控制是否继承系统代理变量。"""
    # 使用 HTTPX 公共发送接口保留代理行为，同时覆盖 SDK 的所有聊天调用。
    http_client = GuardedModelClient(trust_env=settings.openai_trust_env)
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        http_client=http_client,
        max_retries=0,
        timeout=settings.llm_timeout_seconds,
    )
