"""Prompt 工程的静态契约测试，不调用模型 API。"""

from types import SimpleNamespace

from app.prompts.agents import (
    COMPLAINT_PROMPT,
    POSTSALE_PROMPT,
    PRESALE_PROMPT,
    ROUTER_PROMPT,
)
from app.prompts.customer_service import SYSTEM_PROMPT
from app.prompts.evaluation import (
    ANSWER_QUALITY_PROMPT,
    HALLUCINATION_PROMPT,
    PROCESS_SOUNDNESS_PROMPT,
)
from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT
from app.prompts.response_extraction import STRUCTURED_RESPONSE_PROMPT
from app.prompts.summarizer import SUMMARY_PROMPT
from app.multi_agent.router import Router


ALL_SYSTEM_PROMPTS = {
    "customer_service": SYSTEM_PROMPT,
    "router": ROUTER_PROMPT,
    "presale": PRESALE_PROMPT,
    "postsale": POSTSALE_PROMPT,
    "complaint": COMPLAINT_PROMPT,
    "short_term_memory": STM_EXTRACTION_PROMPT,
    "long_term_memory": LTM_EXTRACTION_PROMPT,
    "summary": SUMMARY_PROMPT,
    "answer_quality_judge": ANSWER_QUALITY_PROMPT,
    "hallucination_judge": HALLUCINATION_PROMPT,
    "process_judge": PROCESS_SOUNDNESS_PROMPT,
    "response_extraction": STRUCTURED_RESPONSE_PROMPT,
}


def test_every_system_prompt_uses_five_part_contract():
    for name, prompt in ALL_SYSTEM_PROMPTS.items():
        for section in range(1, 6):
            assert f"## {section}." in prompt, (
                f"{name} 缺少第 {section} 部分的 Prompt 契约"
            )


def test_sensitive_write_requires_confirmation():
    assert "二次确认" in SYSTEM_PROMPT
    assert "明确确认后" in SYSTEM_PROMPT
    assert "二次确认" in POSTSALE_PROMPT
    assert "明确确认后" in POSTSALE_PROMPT


def test_subagent_permission_boundaries_are_explicit():
    assert "没有修改订单" in PRESALE_PROMPT
    assert "apply_refund" in POSTSALE_PROMPT
    assert "没有退款、赔偿" in COMPLAINT_PROMPT
    assert "不得声称已完成转接" in PRESALE_PROMPT


def test_extraction_prompts_remain_formattable():
    STM_EXTRACTION_PROMPT.format(existing_facts="（暂无）")
    LTM_EXTRACTION_PROMPT.format(existing_ltm="（暂无）")


def test_router_uses_system_role_and_does_not_duplicate_current_input():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="postsale")
                    )
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    router = Router(client, "fake-model")
    current = "帮我查订单"

    assert router.route(current, history=[{"role": "user", "content": current}]) == "postsale"
    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": ROUTER_PROMPT}
    assert messages[1]["content"].count(current) == 1
