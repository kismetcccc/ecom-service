# 小飞：面向真实业务的电商智能客服系统
---

## 快速开始（Quick Start）

跑起来只需要一个 OpenAI API Key，5 分钟即可看到「小飞」上线对话。

```bash
# 1. 进入项目并创建虚拟环境（Python 3.11+）
cd ecom-service-agent
python3.11 -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，至少填入：
#   OPENAI_API_KEY=sk-你的key
#   （可选）OPENAI_BASE_URL=https://... 如用中转/代理
#   （可选）MODEL_NAME=gpt-4o-mini
#   （可选）OPENAI_TRUST_ENV=false  # 默认忽略系统代理；使用有效代理时改为 true

# 4. 构建知识库索引（RAG 检索需要，首次运行一次即可）
python -m app.scripts.build_kb_index

# 5. 启动对话
python main.py
```

启动后直接输入问题即可，试试这些：

- `我的订单还没发货，怎么回事？` —— 触发订单 + 物流查询
- `有没有宽松透气的裤子推荐？` —— 触发商品推荐技能
- `这件衣服质量有问题，我要退货` —— 触发退货退款流程

每条回复底部会显示 `[意图 | 置信度 | 是否转人工]`。对话中还支持这些命令：

| 命令 | 作用 |
|------|------|
| `skills` | 查看已加载的技能模块 |
| `memory` | 查看短期 / 长期记忆 |
| `reset` | 清空当前会话 |
| `quit` / `exit` | 退出 |

**开启进阶能力**（可选，改 `.env` 后重启即可）：

- `MULTI_AGENT_ENABLED=true` —— 多 Agent 协作（售前/售后/投诉分流）
- `MCP_ENABLED=true` —— 通过 MCP 协议调用工具（需另起 `python mcp_server/server.py`）
- `RAG_BACKEND=chroma` —— 换用 Chroma 向量数据库（需 `pip install chromadb`）

**上下文预算**（可选，默认输入预算为 `28000 - 4000 = 24000`）：

```env
# 模型总上下文窗口预算（输入 + 预留输出）
CONTEXT_WINDOW_TOKENS=28000
# 为本轮模型回答预留的 token
CONTEXT_RESERVED_OUTPUT_TOKENS=4000
# 自动压缩时至少保留的最近原始消息数
CONTEXT_KEEP_RECENT_MESSAGES=6
```

系统会在每次主要模型调用前保守估算 System Prompt、记忆、摘要、历史消息和
工具 Schema 的总 Token。超过输入预算时，会先压缩最早的历史并重新校验；如果
System Prompt、最近消息或单条输入本身仍无法放入预算，则拒绝请求，HTTP API 返回
`413`。估算器面向不同 OpenAI-compatible 提供商，采用偏保守的通用算法，因此该值
是安全预算而不是供应商账单中的精确 Token 数。

**跑评估 & 测试**：

```bash
python -m app.scripts.run_eval        # 离线评估（沙箱重跑黄金测试集 + LLM judge）
pytest                                # 运行全部单元测试
```

### 多用户并发服务

启动异步 HTTP 服务（模型调用会在线程池中执行，不阻塞事件循环）：

```bash
uvicorn app.server.api:app --host 0.0.0.0 --port 8000
```

然后可以同时打开多个终端，每个终端使用独立用户 ID：

```bash
# 终端 A
python -m app.scripts.chat_client --user-id user-a

# 终端 B
python -m app.scripts.chat_client --user-id user-b

# 同一用户也可以显式区分不同会话
python -m app.scripts.chat_client --user-id user-a --session-id after-sale
```

服务层提供以下并发保障：

- 每个 `user_id/session_id` 使用独立会话文件，聊天上下文不会串线。
- 同一用户的请求通过异步锁顺序处理，避免历史和长期记忆并发写入。
- 不同用户请求并行执行；同步模型 SDK 被放入工作线程，不阻塞 HTTP 事件循环。
- 中间件限制单进程同时执行的请求数；排队超过 `SERVER_QUEUE_TIMEOUT` 时返回 `503` 和 `Retry-After`，形成过载背压。
- `X-Request-ID` 和 `X-Process-Time-Ms` 响应头可用于日志追踪和延迟观测。

也可以直接调用接口：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-a","session_id":"main","message":"我的订单到哪了？"}'
```

并发相关配置位于 `.env.example`。容器镜像由根目录 `Dockerfile` 构建，
Kubernetes 的 3 副本 Deployment、Service 和 HPA 示例位于 `deploy/k8s.yaml`。
示例使用 `ClientIP` 会话亲和与 Pod 本地会话目录；生产环境若要求跨 Pod、跨设备保持同一会话，
应将会话和分布式锁迁移到 Redis 等共享存储后再取消会话亲和。

---

## 项目介绍

“小飞”是一个可直接通过命令行或 HTTP API 使用的电商智能客服系统。它不是只根据提示词生成回答的聊天机器人，而是由大模型负责理解与决策，再通过业务工具、知识库、记忆和标准操作流程完成订单查询、物流跟踪、商品推荐、退换货及投诉分流等任务。

系统面向多用户场景设计。每个用户和会话拥有独立的上下文与持久化文件；同一用户的请求按顺序处理，不同用户可以并行调用模型。HTTP 中间件提供并发限制、排队超时、请求追踪和过载保护，Docker 与 Kubernetes 配置则用于容器化部署和横向扩容。

### 核心能力

| 能力 | 说明 |
|------|------|
| ReAct 与工具调用 | 模型根据问题选择订单、物流、商品、退款、知识检索等业务工具，并结合结果生成回答 |
| 结构化输出 | 每次回复都包含意图、置信度、是否转人工和可选追问，便于上层系统消费 |
| RAG 知识检索 | 从退换货政策、配送说明、会员权益和 FAQ 中检索依据，降低规则类回答的幻觉 |
| Multi-Agent | 可将售前、售后和投诉请求路由给具有独立提示词及工具权限的专家 Agent |
| 用户记忆 | 保存会话内事实和跨会话长期偏好，并按用户隔离 |
| Skill 工作流 | 按需加载退货、订单跟踪和商品推荐等标准操作流程，避免系统提示无限膨胀 |
| MCP 集成 | 可连接独立 MCP 工具服务，也能在连接失败时回退到本地工具 |
| 多用户并发 | FastAPI 异步入口、工作线程执行、用户级锁、会话池、并发限制和 503 背压 |
| 上下文预算 | 模型调用前估算 Token，超限时保持工具消息完整并自动压缩旧历史 |
| 评估体系 | 支持黄金测试集、过程轨迹、规则指标以及可选的 LLM-as-Judge |
| 容器部署 | 提供 Dockerfile，以及带健康检查、Service 和 HPA 的 Kubernetes 示例 |

### 请求处理流程

```text
CLI / HTTP 客户端
        │
        ▼
请求追踪与并发限制中间件
        │
        ▼
按 user_id / session_id 获取独立 Agent
        │
        ├── 单 Agent ReAct
        └── Multi-Agent 路由（可选）
                │
                ▼
     Memory + Skill + RAG + 业务工具
                │
                ▼
结构化客服回复 + 会话持久化
```

### 架构设计原则

- **业务能力与模型解耦**：订单、物流、退款等确定性操作由工具执行，大模型负责理解、规划和组织语言。
- **用户状态隔离**：会话文件、短期记忆、长期记忆和工具上下文均按用户绑定，避免并发请求串数据。
- **渐进式加载上下文**：知识通过 RAG 检索，流程通过 Skill 按需加载，减少无关提示内容和 token 消耗。
- **有界上下文**：调用模型前统一计算输入预算，优先摘要旧历史，并为模型输出保留固定空间。
- **同步能力异步接入**：现有同步 Agent 通过工作线程执行，避免阻塞 FastAPI 事件循环。
- **过载时快速失败**：达到并发上限后请求进入有限等待，超时返回 `503 Retry-After`，防止服务被无限请求拖垮。
- **部署方式可扩展**：本地可运行单进程服务，容器环境可通过 Kubernetes 副本和 HPA 扩展吞吐量。

---

## 项目架构

### 当前架构

```
ecom-service-agent/
├── main.py                        # CLI 入口（支持单 Agent / Multi-Agent 模式切换 + memory/skills 命令）
├── requirements.txt
├── .env.example
│
├── app/                           # 主 Bot 全部代码 + 数据
│   ├── config/
│   │   ├── settings.py            # 环境配置与并发参数
│   │   └── openai_client.py       # 统一创建模型客户端并控制系统代理
│   ├── prompts/
│   │   ├── customer_service.py    # 电商客服 system prompt（含工具使用指南 + 记忆能力）
│   │   ├── summarizer.py          # 历史摘要 prompt
│   │   ├── agents.py              # Multi-Agent 子 Agent prompt（售前/售后/投诉 + Router）
│   │   ├── memory.py              # 记忆提取 prompt（短期 STM / 长期 LTM 事实抽取）
│   │   └── evaluation.py          # LLM-as-judge prompt（回答质量 / 幻觉 / 过程合理性）
│   ├── schemas/
│   │   └── response.py            # 结构化输出 schema（Pydantic）
│   ├── agent/                     # Agent 核心、工具、知识库、记忆与技能
│   │   ├── chat.py                # 核心 ReAct 循环（集成 MemoryManager + SkillManager）
│   │   ├── context_budget.py      # Provider-neutral Token 估算、预算校验与安全分段
│   │   ├── summarizer.py          # LLM 自我压缩老对话（支持工具消息）
│   │   ├── storage.py             # 会话 JSON 持久化（含短期记忆）
│   │   ├── memory/                # 短期与长期记忆系统
│   │   │   ├── __init__.py        # 导出 MemoryManager / ShortTermMemory / LongTermMemory
│   │   │   ├── manager.py         # MemoryManager：统一管理短期 + 长期记忆
│   │   │   ├── short_term.py      # 短期记忆：会话内事实提取
│   │   │   ├── long_term.py       # 长期记忆：跨会话持久化（JSON per user）
│   │   │   └── extraction.py      # LLM 事实提取（共用模块）
│   │   ├── skills/                # Skill 发现、目录与按需加载
│   │   │   ├── __init__.py        # 导出 SkillManager / SkillMeta
│   │   │   ├── loader.py          # SkillManager：扫描、发现、加载 SKILL.md（渐进式披露）
│   │   │   └── definitions/       # 技能内容（遵循 Agent Skills 开放标准，每个一个 SKILL.md）
│   │   │       ├── process-return/
│   │   │       │   └── SKILL.md   # 退货退款处理技能（确认订单→校验资格→退款→告知进度）
│   │   │       ├── track-order/
│   │   │       │   └── SKILL.md   # 订单物流跟踪技能（查单→查物流→综合建议）
│   │   │       └── product-recommend/
│   │   │           └── SKILL.md   # 商品推荐技能（了解需求→查偏好→搜索→推荐）
│   │   ├── strategies/            # Agent 执行策略扩展目录
│   │   ├── tools/                 # 电商工具集（Function Calling）
│   │   │   ├── mock_data.py       # Mock 数据：订单、商品、物流
│   │   │   ├── registry.py        # 本地工具注册表 + OpenAI schema + 分发执行
│   │   │   ├── manager.py         # ToolManager：统一管理本地 + MCP 工具（支持 allowed_tools 过滤）
│   │   │   ├── order.py           # 查询订单详情
│   │   │   ├── product.py         # 搜索商品信息
│   │   │   ├── logistics.py       # 查询物流轨迹
│   │   │   ├── refund.py          # 申请退款
│   │   │   ├── knowledge.py       # search_knowledge：RAG 政策/FAQ 检索
│   │   │   ├── memory_tool.py     # recall_user_memory：查询用户记忆
│   │   │   └── skill_tool.py      # load_skill：按需加载技能指令
│   │   └── rag/                   # RAG 模块
│   │       ├── chunker.py         # Markdown → Chunk（按二级标题切分）
│   │       ├── embedder.py        # OpenAI Embeddings 封装
│   │       ├── retriever.py       # KnowledgeRetriever：query → 向量检索
│   │       ├── backends/          # 向量后端（可切换）
│   │       │   ├── base.py        # VectorBackend 抽象接口
│   │       │   ├── numpy_backend.py   # 手写余弦 + JSON（教学透明，零依赖）
│   │       │   └── chroma_backend.py  # Chroma 嵌入式向量数据库（生产代表）
│   │       └── knowledge/         # 知识库源文档（markdown，RAG 数据源）
│   │           ├── 退换货政策.md
│   │           ├── 配送说明.md
│   │           ├── 会员权益.md
│   │           └── 常见问题FAQ.md
│   ├── mcp_client/                # MCP Client（同步封装）
│   │   ├── client.py              # MCPClient：后台线程管理异步连接
│   │   └── converter.py           # MCP Tool schema → OpenAI function calling 格式
│   ├── evaluation/                # Agent 离线评估体系
│   │   ├── __init__.py            # 导出 EvalCase / Sandbox / Evaluator / RunTrace 等
│   │   ├── dataset.py             # EvalCase 数据结构 + load_dataset
│   │   ├── trace.py               # RunTrace：沙箱采集的过程+结果载体
│   │   ├── sandbox.py             # Sandbox：隔离环境 + 共享 client 插桩 + 采集
│   │   ├── metrics.py             # 过程/结果双层指标（代码规则 + LLM judge）
│   │   ├── evaluator.py           # Evaluator：跑用例 → 双层评分 → 聚合报告
│   │   └── cases.json             # 黄金测试集（~10 条，引用 mock 数据）
│   ├── multi_agent/               # 售前、售后、投诉多 Agent 协作
│   │   ├── router.py              # 意图路由器（LLM 分类 → 子 Agent）
│   │   ├── agents.py              # SubAgent 子 Agent 类 + 配置
│   │   └── orchestrator.py        # 编排器：路由 → 执行 → 结构化提取（集成 MemoryManager + SkillManager）
│   ├── scripts/
│   │   ├── build_kb_index.py      # 离线构建知识库索引（--backend numpy/chroma）
│   │   ├── run_eval.py            # 离线运行评估（--mode single/multi · --judge/--no-judge · --output）
│   │   └── chat_client.py         # 支持 user_id/session_id 的多终端客户端
│   ├── server/
│   │   ├── api.py                 # FastAPI 路由、健康检查与上游异常映射
│   │   ├── middleware.py          # 请求追踪、并发限制和过载背压
│   │   └── pool.py                # Agent 会话池、用户锁和状态隔离
│   └── sessions/                  # 运行时生成，已 .gitignore
│       ├── session.json           # 当前会话快照
│       ├── concurrent/            # HTTP 多用户会话，按 user_id/session_id 保存
│       ├── kb_index.json          # NumpyBackend 索引
│       ├── chroma/                # ChromaBackend 持久化目录
│       └── memory/                # 长期记忆存储（按 user_id 分文件）
│           └── {user_id}.json
│
├── mcp_server/                    # MCP Server（独立微服务）
│   └── server.py                  # FastMCP + Streamable HTTP，暴露电商工具
├── deploy/
│   └── k8s.yaml                   # Deployment、Service、健康检查与 HPA
├── Dockerfile                     # HTTP 服务容器镜像
│
└── tests/                         # 全部测试
    ├── test_agent.py              # 结构化输出 + 多轮 + reset
    ├── test_conversation_management.py  # 多轮对话管理
    ├── test_react_agent.py        # ReAct Agent + Function Calling
    ├── test_mcp.py                # MCP 集成
    ├── test_rag.py                # RAG 知识库检索
    ├── test_multi_agent.py        # Multi-Agent 协作
    ├── test_memory.py             # Memory 短期记忆 & 长期记忆
    ├── test_skills.py             # Skill 可复用能力模块
    ├── test_evaluation.py         # Agent 评估体系（沙箱 + 双层测评）
    └── test_concurrency.py        # 用户隔离、并行执行、容量竞态和过载保护
```
