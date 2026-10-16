from typing import Literal
import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    # 默认不继承 HTTP_PROXY/HTTPS_PROXY，避免无效系统代理导致全部请求失败。
    # 确实需要企业代理时可在 .env 中设为 true。
    openai_trust_env: bool = False
    # 免费接口保护：所有聊天模型阶段共享进程内额度，不隐式重试上游请求。
    llm_max_inflight: int = Field(default=1, ge=1)
    llm_requests_per_minute: int = Field(default=20, ge=1)
    llm_timeout_seconds: float = Field(default=20.0, gt=0)
    # 当前 Qwen3.8 开源模型不能关闭思考；客服链路限制每次调用的思考开销。
    llm_required_thinking_budget: int = Field(default=128, ge=1, le=8192)
    llm_rate_limit_cooldown_seconds: float = Field(default=60.0, gt=0)

    # SQLite 商品数据库，无需独立数据库服务。
    database_url: str = "sqlite:///app/sessions/ecom.db"

    # ReAct 循环
    max_react_steps: int = 5

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"
    mcp_connect_timeout: float = 5.0
    mcp_call_timeout: float = 8.0
    mcp_max_inflight: int = 32
    mcp_read_retries: int = 1
    mcp_failure_threshold: int = 3
    mcp_recovery_seconds: float = 15.0
    mcp_allow_local_read_fallback: bool = False
    order_database_path: str = "app/sessions/orders.db"
    # Seeded rows are explicitly demo data. Disable in deployments with a real order feed.
    seed_demo_orders: bool = True

    # RAG 配置（第5期）
    # Leave blank to reuse OPENAI_API_KEY / OPENAI_BASE_URL.
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    kb_dir: str = "app/agent/rag/knowledge"
    # 独立知识路由；RAG 数据库和索引配置保持原样。
    knowledge_mode: Literal["rag", "wiki", "wiki_first"] = "rag"
    wiki_store_dir: str = "app/sessions/llm_wiki"
    # RAG 后端：numpy/chroma 保留用于本地对照；qdrant 用独立服务承载并发检索。
    rag_backend: str = "numpy"
    # 主向量服务不可用时的只读降级；空字符串可关闭。索引模型不匹配仍会拒绝。
    rag_fallback_backend: str = "numpy"
    # NumpyBackend 的 JSON 索引路径
    kb_index_path: str = "app/sessions/kb_index.json"
    # ChromaBackend 的持久化目录与 collection 名
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "ecom_kb"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "ecom_kb"
    qdrant_api_key: str = ""
    qdrant_timeout_seconds: float = 5.0
    # 演示政策只能在明确允许的环境参与检索；正式部署应设为 false。
    rag_allow_demo_policies: bool = True
    # 阈值必须由离线评估校准；0 表示尚未启用相似度门禁。
    rag_min_score: float = 0.0
    rag_retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    rag_candidate_k: int = Field(default=40, ge=5, le=200)
    rag_rrf_k: int = Field(default=60, ge=1, le=1000)
    rag_min_rrf_score: float = Field(default=0.0, ge=0, le=1)
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "BAAI/bge-reranker-base"
    rag_rerank_device: str = "cpu"
    rag_rerank_top_n: int = Field(default=20, ge=5, le=200)
    rag_rerank_batch_size: int = Field(default=8, ge=1, le=64)
    rag_rerank_max_length: int = Field(default=512, ge=32, le=8192)
    rag_rerank_local_files_only: bool = False
    rag_rerank_failure_mode: Literal["rrf", "error"] = "rrf"
    rag_min_rerank_score: float = Field(default=0.0, ge=0, le=1)

    # Multi-Agent 配置（第6期）
    multi_agent_enabled: bool = True
    # V2：退换货使用有依赖、只读核验的工作流；关闭可恢复旧专家编排。
    multi_agent_workflows_enabled: bool = True
    # 独立质检：off / shadow（观测）/ enforce（阻断）；最多一次无工具修订。
    multi_agent_review_mode: Literal["off", "shadow", "enforce"] = "enforce"
    multi_agent_review_model: str = ""
    # 对已确认支持的百炼 DeepSeek 混合模型关闭质检深度思考；不影响业务专家。
    multi_agent_review_fast_mode: bool = True
    multi_agent_review_max_repairs: int = Field(default=1, ge=0, le=1)
    multi_agent_review_timeout_seconds: float = Field(default=25, gt=0, le=60)
    multi_agent_review_budget_seconds: float = Field(default=60, gt=0, le=120)
    # MA-02/03：路由不确定时澄清；规则只覆盖高精度、只读或纯投诉场景。
    multi_agent_rules_enabled: bool = True
    multi_agent_route_min_confidence: float = 0.75
    # MA-05/07：一次复合请求最多两个子任务；并行仅用于不同专家的独立只读任务。
    multi_agent_max_tasks: int = 2
    multi_agent_parallel_reads: bool = True
    multi_agent_parallel_requests: int = 8
    multi_agent_model_timeout_seconds: float = 30.0
    multi_agent_minimal_context: bool = True
    multi_agent_context_recent_messages: int = 8

    # Memory 配置（第7期）
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50
    # MR-09：限制交互摘要总量和保留时间，避免长期运行后无限增长。
    max_ltm_interaction_summaries: int = 100
    ltm_interaction_retention_days: int = 180
    # MR-12：长期记忆需用户授权，持久化内容使用认证加密。
    memory_require_consent: bool = True
    memory_consent_db_path: str = "app/sessions/memory_control.db"
    memory_encryption_key: str = ""
    # MR-08：不同类型记忆使用不同生命周期，可通过环境变量独立调整。
    stm_identity_ttl_hours: int = 2160
    stm_preference_ttl_hours: int = 720
    stm_intent_ttl_hours: int = 24
    stm_emotion_ttl_hours: int = 6
    stm_order_ttl_hours: int = 72
    stm_other_ttl_hours: int = 24
    ltm_identity_ttl_days: int = 180
    ltm_preference_ttl_days: int = 365
    ltm_behavior_ttl_days: int = 180
    ltm_issue_ttl_days: int = 30
    ltm_other_ttl_days: int = 90
    ltm_confirmation_days: int = 180
    # MR-10：每轮最多注入的长期事实数量。
    memory_retrieval_top_k: int = 5
    # MR-15：可单独指定更轻量的记忆提取模型；留空时复用主模型。
    memory_extraction_model: str = ""

    # Skill 配置（第8期）
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（第9期，离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.6  # 单维度通过阈值（judge 归一化到 0-1 后比较）

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    # Legacy message-count compaction remains as a secondary cleanup trigger.
    history_threshold: int = 10
    history_keep_recent: int = 3
    # Primary preflight budget: effective input = window - reserved output.
    context_window_tokens: int = 28000
    context_reserved_output_tokens: int = 4000
    context_keep_recent_messages: int = 6
    # MR-02：每个会话保留的滚动摘要历史版本数。
    summary_version_limit: int = 10

    # 并发 HTTP 服务
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_max_concurrency: int = 32
    server_queue_timeout: float = 10.0
    server_max_sessions: int = 1000
    server_session_idle_seconds: float = Field(default=1800, gt=0)
    concurrent_session_dir: str = "app/sessions/concurrent"

    # MR-05：HTTP 用户身份由注册登录令牌确定，禁止仅信任客户端 user_id。
    auth_database_path: str = "app/sessions/auth.db"
    auth_secret_key: str = ""
    auth_token_ttl_seconds: int = 43200
    # CR-04/11：正文日志默认关闭；认证入口有独立的资源限制。
    log_model_content: bool = False
    session_encryption_key: str = ""
    auth_max_concurrency: int = Field(default=4, ge=1, le=64)
    auth_attempts_per_minute: int = Field(default=20, ge=1)

    model_config = {
        # 2026-10-10: isolated offline workers never ingest development secrets.
        "env_file": None if os.environ.get("ECOM_EVAL_WORKER") == "1" else ".env",
        "protected_namespaces": ("settings_",),
    }

    @model_validator(mode="after")
    def validate_context_budget(self):
        if self.context_window_tokens <= 0:
            raise ValueError("CONTEXT_WINDOW_TOKENS 必须大于 0")
        if self.context_reserved_output_tokens <= 0:
            raise ValueError("CONTEXT_RESERVED_OUTPUT_TOKENS 必须大于 0")
        if self.context_reserved_output_tokens >= self.context_window_tokens:
            raise ValueError(
                "CONTEXT_RESERVED_OUTPUT_TOKENS 必须小于 CONTEXT_WINDOW_TOKENS"
            )
        if self.context_keep_recent_messages <= 0:
            raise ValueError("CONTEXT_KEEP_RECENT_MESSAGES 必须大于 0")
        if self.summary_version_limit < 10:
            raise ValueError("SUMMARY_VERSION_LIMIT 不得小于 10")
        if self.auth_secret_key and len(self.auth_secret_key) < 32:
            raise ValueError("AUTH_SECRET_KEY 配置后不得少于 32 个字符")
        if self.auth_token_ttl_seconds <= 0:
            raise ValueError("AUTH_TOKEN_TTL_SECONDS 必须大于 0")
        if self.mcp_connect_timeout <= 0 or self.mcp_call_timeout <= 0:
            raise ValueError("MCP 超时配置必须大于 0")
        if self.mcp_failure_threshold <= 0 or self.mcp_recovery_seconds <= 0:
            raise ValueError("MCP 熔断配置必须大于 0")
        if self.mcp_max_inflight <= 0 or self.mcp_read_retries < 0:
            raise ValueError("MCP 并发或重试配置无效")
        if not -1.0 <= self.rag_min_score <= 1.0:
            raise ValueError("RAG_MIN_SCORE 必须在 -1 到 1 之间")
        if self.rag_backend not in {"numpy", "chroma", "qdrant"}:
            raise ValueError("RAG_BACKEND 必须是 numpy、chroma 或 qdrant")
        if self.rag_fallback_backend not in {"", "numpy", "chroma"}:
            raise ValueError("RAG_FALLBACK_BACKEND 必须为空、numpy 或 chroma")
        if not 0.0 <= self.multi_agent_route_min_confidence <= 1.0:
            raise ValueError("MULTI_AGENT_ROUTE_MIN_CONFIDENCE 必须在 0 到 1 之间")
        if not 1 <= self.multi_agent_max_tasks <= 2:
            raise ValueError("MULTI_AGENT_MAX_TASKS 必须在 1 到 2 之间")
        if self.multi_agent_parallel_requests <= 0:
            raise ValueError("MULTI_AGENT_PARALLEL_REQUESTS 必须大于 0")
        if self.multi_agent_model_timeout_seconds <= 0:
            raise ValueError("MULTI_AGENT_MODEL_TIMEOUT_SECONDS 必须大于 0")
        if self.multi_agent_context_recent_messages <= 0:
            raise ValueError("MULTI_AGENT_CONTEXT_RECENT_MESSAGES 必须大于 0")
        if self.max_ltm_interaction_summaries <= 0:
            raise ValueError("MAX_LTM_INTERACTION_SUMMARIES 必须大于 0")
        if self.ltm_interaction_retention_days <= 0:
            raise ValueError("LTM_INTERACTION_RETENTION_DAYS 必须大于 0")
        ttl_values = [
            self.stm_identity_ttl_hours, self.stm_preference_ttl_hours,
            self.stm_intent_ttl_hours, self.stm_emotion_ttl_hours,
            self.stm_order_ttl_hours, self.stm_other_ttl_hours,
            self.ltm_identity_ttl_days, self.ltm_preference_ttl_days,
            self.ltm_behavior_ttl_days, self.ltm_issue_ttl_days,
            self.ltm_other_ttl_days, self.ltm_confirmation_days,
            self.memory_retrieval_top_k,
        ]
        if any(value <= 0 for value in ttl_values):
            raise ValueError("Memory TTL 配置必须大于 0")
        return self


settings = Settings()
