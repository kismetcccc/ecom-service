"""Multi-Agent 编排器：协调 Router 和子 Agent 完成用户请求。

流程：Router 分类意图 → 选择子 Agent → ReAct 执行 → 结构化提取 → 持久化。
"""

from typing import Optional

from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import summarize
from app.config.openai_client import create_openai_client
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.router import Router
from app.schemas.response import CustomerServiceResponse, IntentType
from app.agent.tools.manager import ToolManager


class MultiAgentOrchestrator:
    """多 Agent 编排器，对外接口与 EcomAgent 一致。"""

    def __init__(
        self,
        session_path: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.client = create_openai_client()
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=user_id or settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )

        from app.agent.tools.memory_tool import recall_memory
        local_tool_overrides = {
            "recall_user_memory": (
                lambda query="": recall_memory(self.memory_manager, query)
            ),
            "load_skill": self.skill_manager.load_skill,
        }

        self.router = Router(self.client, self.model)
        self.agents: dict[str, SubAgent] = {}
        for key, cfg in AGENT_CONFIGS.items():
            tm = ToolManager(
                use_mcp=settings.mcp_enabled,
                mcp_server_url=settings.mcp_server_url,
                allowed_tools=cfg["tools"],
                disabled_tools={
                    name for name, disabled in (
                        ("recall_user_memory", not settings.memory_enabled),
                        ("load_skill", not settings.skills_enabled),
                    )
                    if disabled
                },
                local_tool_overrides=local_tool_overrides,
            )
            self.agents[key] = SubAgent(
                name=cfg["name"],
                system_prompt=cfg["prompt"],
                tool_manager=tm,
                client=self.client,
                model=self.model,
                temperature=self.temperature,
            )

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """路由 → 子 Agent 执行 → 结构化提取 → 返回结果。"""
        self.raw_messages.append({"role": "user", "content": user_input})

        agent_key = self.router.route(user_input, self.raw_messages)
        agent = self.agents[agent_key]
        print(f"\n🔀 [路由] → {agent.name}")

        messages = self._build_messages(agent)
        final_text, new_messages = agent.handle(
            messages, max_steps=self.max_react_steps,
        )
        self.raw_messages.extend(new_messages)

        result = self._extract_structured_response(user_input, final_text)

        self.memory_manager.update_short_term(self.raw_messages[-6:])

        self.raw_messages.append(
            {"role": "assistant", "content": result.model_dump_json()}
        )

        if len(self.raw_messages) > self.history_threshold:
            self._compress_history()

        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )
        return result

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    def close(self):
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        for agent in self.agents.values():
            agent.tool_manager.close()

    def _build_messages(self, agent: SubAgent) -> list[dict]:
        """用子 Agent 的 system prompt 构建消息列表。"""
        system_content = agent.system_prompt
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
            })
        messages.extend(self.raw_messages)
        return messages

    def _extract_structured_response(
        self, user_input: str, text: str,
    ) -> CustomerServiceResponse:
        intent_guide = (
            "请主要根据用户原始输入判断意图，客服回复只作为辅助证据。\n"
            "意图映射：order_query=订单/物流查询；return_request=退货/退款/换货/退换货政策；"
            "product_consult=商品信息/库存/推荐；complaint=明确投诉或威胁向消协、媒体、"
            "监管机构举报；after_sale=未升级投诉的维修或质量售后；promotion=优惠促销；"
            "account=账户问题；greeting=问候；other=其他。\n"
            "明确投诉升级、威胁监管/法律/媒体曝光、涉及账户支付安全或超出客服权限时，"
            "requires_human=true；普通订单查询为 false。"
        )
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{intent_guide}\n"
                            "基于用户原始输入和客服回复提取结构化信息。"
                            "reply 字段直接使用原文，不要修改或缩减。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"用户原始输入：\n{user_input}\n\n客服回复：\n{text}",
                    },
                ],
                temperature=0.0,
                response_format=CustomerServiceResponse,
            )
            result = response.choices[0].message.parsed
            return self._enforce_structured_invariants(result, user_input, text)
        except Exception:
            return self._extract_structured_fallback(user_input, text, intent_guide)

    def _extract_structured_fallback(
        self, user_input: str, text: str, intent_guide: str,
    ) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{intent_guide}\n\n"
                        "基于用户原始输入和客服回复提取结构化信息并输出 JSON。\n"
                        "reply 字段直接使用原文，不要修改或缩减。\n\n"
                        "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                        "{\n"
                        f'  "intent": <从以下选择: {intent_values}>,\n'
                        '  "confidence": <0.0到1.0的浮点数>,\n'
                        '  "reply": <原文回复内容>,\n'
                        '  "requires_human": <true或false>,\n'
                        '  "follow_up_question": <追问问题或null>\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户原始输入：\n{user_input}\n\n客服回复：\n{text}",
                },
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = CustomerServiceResponse.model_validate_json(raw)
        return self._enforce_structured_invariants(result, user_input, text)

    @staticmethod
    def _enforce_structured_invariants(
        result: CustomerServiceResponse, user_input: str, reply: str,
    ) -> CustomerServiceResponse:
        """保护回复正文，并对明确投诉实施确定性的升级规则。"""
        result.reply = reply
        escalation_markers = ("投诉", "消协", "举报", "曝光", "起诉", "监管")
        if any(marker in user_input for marker in escalation_markers):
            result.intent = IntentType.COMPLAINT
            result.requires_human = True
            result.confidence = max(result.confidence, 0.95)
        return result

    def _compress_history(self) -> None:
        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]

        new_summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        print(
            f"\n💾 [已压缩 {len(old_messages)} 条老消息 → summary "
            f"({len(new_summary)} 字)]\n"
        )
