from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """用户意图类型"""

    ORDER_QUERY = "order_query"  # 订单查询
    DELIVERY_EXCEPTION = "delivery_exception"  # 履约异常交叉诊断
    RETURN_REQUEST = "return_request"  # 退换货
    PRODUCT_CONSULT = "product_consult"  # 商品咨询
    COMPLAINT = "complaint"  # 投诉
    AFTER_SALE = "after_sale"  # 售后服务
    PROMOTION = "promotion"  # 优惠活动
    ACCOUNT = "account"  # 账户问题
    GREETING = "greeting"  # 打招呼/闲聊
    OTHER = "other"  # 其他


class PolicySource(BaseModel):
    """来源由本轮检索工具填充，不能让模型自行编造。"""

    chunk_id: str
    doc: str
    section: str
    version: str = ""
    effective_from: str = ""
    source_url: str = ""
    is_demo: bool = False


class TaskOutcome(BaseModel):
    """MA-08/13：只描述子任务是否得到答复，不虚构业务操作已完成。"""

    label: str
    status: Literal["answered", "blocked", "verified", "skipped", "waiting_confirmation"]
    key: str = ""
    agent: str = ""
    depends_on: list[str] = Field(default_factory=list)
    reason_code: str = ""


class QualityReview(BaseModel):
    """Operational outcome only; model critique and private evidence stay internal."""

    status: Literal["approved", "repaired", "blocked", "unavailable", "skipped", "disabled"]
    mode: Literal["off", "shadow", "enforce"]
    reason_code: str = ""
    model_calls: int = 0
    repair_count: int = 0
    facts_only: bool = False
    evidence_count: int = 0
    elapsed_ms: int = 0


class SkillCheck(BaseModel):
    code: str
    status: Literal["verified", "attention", "missing"]
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class SkillAudit(BaseModel):
    skill: str
    version: str
    rule_version: str
    status: Literal["incomplete", "review_needed"] = "incomplete"
    checks: list[SkillCheck] = Field(default_factory=list)


class SkillExecution(BaseModel):
    name: str
    run_id: str
    version: str = ""
    bundle_version: str = ""
    runtime_version: str
    state: Literal["needs_input", "blocked", "waiting_confirmation", "previewed", "confirmed", "submitted", "disabled"]
    policy_scope: str
    resources: list[str] = Field(default_factory=list)
    reason_code: str = ""


class RefundAction(BaseModel):
    order_id: str
    workflow_id: str


class CustomerServiceResponse(BaseModel):
    """客服回复的结构化输出"""

    intent: IntentType = Field(description="识别到的用户意图类型")
    confidence: float = Field(
        description="意图识别的置信度，0.0-1.0", ge=0.0, le=1.0
    )
    reply: str = Field(description="给用户的回复内容")
    requires_human: bool = Field(
        description="是否需要转接人工客服", default=False
    )
    follow_up_question: Optional[str] = Field(
        description="需要向用户进一步确认的问题，如果不需要则为 null",
        default=None,
    )
    sources: list[PolicySource] = Field(default_factory=list)
    knowledge_status: Optional[str] = None
    knowledge_providers: list[Literal["rag", "wiki"]] = Field(default_factory=list)
    knowledge_fallback: bool = False
    task_status: Literal["single", "answered", "partial", "blocked"] = "single"
    task_outcomes: list[TaskOutcome] = Field(default_factory=list)
    workflow_id: Optional[str] = None
    workflow_status: Optional[Literal["blocked", "needs_input", "waiting_confirmation"]] = None
    quality_review: Optional[QualityReview] = None
    skill_audit: Optional[SkillAudit] = None
    skill_execution: Optional[SkillExecution] = None
    refund_action: Optional[RefundAction] = None
