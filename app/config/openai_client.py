"""统一创建 OpenAI 兼容客户端。"""

import httpx
from openai import OpenAI

from app.config.settings import settings


def create_openai_client() -> OpenAI:
    """创建聊天客户端，并显式控制是否继承系统代理变量。"""
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        http_client=httpx.Client(trust_env=settings.openai_trust_env),
    )
