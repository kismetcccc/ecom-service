# 并夕夕 · 小飞智能客服

## RAG 最终检索链路：BM25 + Dense + RRF + Cross-encoder

RAG 默认采用双路召回：BM25 使用中文单字/双字组合及完整英文、数字、SKU 词元；Dense 复用原有 Embedding 与向量后端。两路各取 40 个候选，按 chunk_id 去重，以 `sum(1 / (60 + rank))` 做 RRF 融合，再对前 20 个候选使用 `BAAI/bge-reranker-base` 联合编码问题和片段重排，返回工具要求的 Top-K。政策有效性在融合前和工具输出前检查，原证据引用逻辑保留。

安装与模型预下载（Windows 使用 `.venv\Scripts\python.exe` 替换 `python`）：

```bash
python -m pip install -r requirements.txt
python -m app.scripts.warm_reranker
```

预下载成功后重启 API/MCP 进程。模型默认在 CPU 运行；首次下载需要联网，预下载只缓存权重，每个服务进程首次使用仍需加载模型。离线部署将 `RAG_RERANK_MODEL` 指向已下载的本地目录，设置 `RAG_RERANK_LOCAL_FILES_ONLY=true`。Docker 镜像安装 CPU 版 PyTorch；本地 Compose 用 `reranker-cache` 命名卷持久化 `/root/.cache/huggingface`，容器重建后仍可复用权重。首次需要在容器内预下载或复制已下载模型缓存，宿主机缓存不会自动进入镜像。

`.env.example` 列出全部参数。默认 `RAG_RETRIEVAL_MODE=hybrid`、`RAG_RERANK_ENABLED=true`；设置 `RAG_RERANK_ENABLED=false` 可对照 RRF，设置 `RAG_RETRIEVAL_MODE=dense` 可对照原纯向量链路。重排失败时默认返回 RRF 排序，并明确输出 `rerank_status=unavailable`；要求重排必须成功时设置 `RAG_RERANK_FAILURE_MODE=error`。

返回结果保留 `score`，新增 `score_type`、`dense_score`、`bm25_score`、`rrf_score`、`rerank_score` 和 `rerank_status`。`RAG_MIN_SCORE` 仅作用于纯 Dense，`RAG_MIN_RRF_SCORE` 作用于 RRF，`RAG_MIN_RERANK_SCORE` 作用于 Cross-encoder。重排分数是 sigmoid 转换后的排序分数，不是经过校准的正确率；阈值默认 0，需用标注集校准。

三个向量后端均支持混合检索，不需要重建现有向量。BM25 从已发布索引读取片段，不直接读取可能尚未发布的 Markdown；Qdrant 固定同一物理集合完成两路检索，发布和回滚后下一次检索刷新语料。当前本地 BM25 面向项目规模的知识库，每次检查语料快照，未变化时复用词项索引；大规模语料应将稀疏召回迁到检索服务。Chroma 全量重建期间仍可能暂时不可用。

知识来源路由保持独立：只有 `rag` 或 `wiki_first` 回退到 RAG 时运行本链路；需要纯 RAG 演示时使用 `python -m app.scripts.manage_knowledge switch rag`。知识对比评测复用线上配置并记录参数；旧报告不会自动成为新链路的效果证明。`quantify retrieval` 仍是明确标注的原始 Dense 基线。

2026-10-13：质量评测台升级为[量化指标评估](docs/量化指标评估.md)，支持性能、质量、规模、效率与成本、稳定性五类指标。访问 `/quality`，导入 `reports/quant-overview.json` 查看本次实测与样本口径；重跑入口为 `python -m app.scripts.quantify --help`。

2026-10-13：完成 [退款 Skill 工程化链路](docs/10.13更新-退款Skill完整链路与工程化.md)：按需资源、包版本校验、退款预览与申请快照、三步确认界面和结果恢复；保留原售后 SOP，交易权限继续由服务端控制。

2026-10-13：新增 [履约证据核对 Skill](docs/10.13更新-履约证据Skill与简历说明.md)，以“文字 SOP + Python 验收”生成带版本和证据编号的核对清单；文档包含实现边界、简历写法和面试问答。

2026-10-13：Skills 新增专家能力过滤、加载时权限校验、版本追踪与文件变更检测，见 [Skills 治理与优化](docs/10.13更新-Skills治理与优化.md)。

本机一键运行：打开 Docker Desktop，双击根目录 `start-local.cmd`，访问 `http://127.0.0.1:18000/`。操作与数据说明见 [启动 README](README-启动.md)。

首次启动或重启服务，请先看 [本地启动说明（2026-10-11）](README-启动.md)：按顺序启动向量服务（按需）、MCP 和聊天后端。

## 独立 LLM Wiki 与热切换

现支持 `rag`、`wiki`、`wiki_first` 三种知识来源模式。默认仍走 RAG，现有向量数据库与索引无需迁移或重建。Wiki 使用独立文件版本，支持草稿审核发布、Markdown 目录与关联页面、来源变化校验及版本回切。

首次部署新代码需重启 API/MCP；之后使用 `python -m app.scripts.manage_knowledge switch rag` 或 `switch wiki` / `switch wiki_first` 即可切换后续检索，无需再次重启。启用 Wiki 前须先发布审核稿。完整操作见 [LLM Wiki 接入与切换](docs/10.10更新-LLMWiki接入与切换.md)。

## 五角色协作：履约诊断与独立质检

Multi-Agent 现有四个业务专家（售前、售后、投诉、履约诊断）与一个内部回复质检 Agent。物流停滞、签收争议等进入履约诊断，必须取得同一订单的订单/物流证据；生成式回复经过独立质检，必要时只修订一次文字，再次审核后返回。质检无法执行业务工具。架构、使用样例、成本开关和评估方式见 [多 Agent 工程化升级](docs/10.10更新-多Agent工程化升级.md)。

## 2026-10-09：业务工作流与 RAG Wiki 草稿

Multi-Agent 模式新增退换货自动核验：订单与政策检索可并行，两个前置步骤通过后才检查售后资格；失败时显示阻塞和跳过状态，成功后等待页面预览与用户确认。不会由模型文字直接提交退款。

设置 `MULTI_AGENT_ENABLED=true` 后，`MULTI_AGENT_WORKFLOWS_ENABLED` 默认开启；设为 `false` 可恢复旧编排。示例输入：`申请退货 ORD-20240115-001`。涉及商品推荐或物流查询的复合诉求仍使用原有专家编排，详情见 [升级说明](docs/10.10更新-工作流与Wiki升级.md)。

LLM Wiki 入口复用 `KB_DIR` 中的 RAG 原始资料，生成供审阅的流程知识 JSON 草稿。`python -m app.scripts.build_workflow_wiki` 仅统计来源；增加 `--generate` 才调用模型。草稿通过独立的 Wiki 发布流程接入查询，不写回 RAG 索引，也不直接控制业务工具。

这是一个可在浏览器和命令行使用的电商客服演示项目。它通过 ReAct 调用订单、物流、商品、退款和知识检索工具；可选 MCP、多 Agent、短期/长期记忆及 Skill。订单、商品、物流和政策资料目前主要是**演示数据**，退款申请不会触发真实资金流转。

## 开发环境与配置确认

下表是开发配置示例。运行后端、索引数量和服务状态以本机配置与实际探测为准，不将历史验收时的状态作为启动前提。

| 项目 | 当前状态 |
|---|---|
| 项目目录 | `D:\project_sum\agent_project\ecom-service-agent-main` |
| Python | 项目 `.venv` 中的 Python 3.12.7 |
| 对话 / 向量模型 | `qwen-plus` / `text-embedding-v4`，调用 OpenAI-compatible API |
| 可选开关 | `MCP_ENABLED`、`MULTI_AGENT_ENABLED` 按需设置；`RAG_BACKEND` 选择实际已构建的后端 |
| 本地数据 | `app/sessions/` 下的 SQLite、加密会话及配置的向量索引 |
| Qdrant | 使用 Qdrant 后端时必须启动服务并构建对应索引；NumPy 后端不依赖它 |

`.env` 中的密钥只在本地配置，不要提交到 Git。下文的命令均从项目根目录执行，明确调用 `.venv\Scripts\python.exe`，避免 PowerShell 同时激活 Anaconda/base 时误用全局 Python。

## 本机启动：浏览器 + MCP（当前配置）

首次克隆时先准备环境；已有 `.venv` 和 `.env` 的机器**不要覆盖**它们：

```powershell
cd D:\project_sum\agent_project\ecom-service-agent-main
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，至少填写模型服务的 `OPENAI_API_KEY`；本项目示例默认使用阿里云百炼兼容接口，还应确认 `OPENAI_BASE_URL`、`MODEL_NAME`、`EMBEDDING_MODEL` 与你的账号能力匹配。当前 `.env` 已存在并配置了模型密钥，无需重新复制。`AUTH_SECRET_KEY` 应设置为至少 32 个字符的随机值；多进程或多实例必须使用同一密钥。

在首次使用 RAG、修改政策文件或更换 Embedding 模型后构建索引。此命令会调用向量模型 API，可能产生费用；当前已有 NumPy 索引，不必为了普通启动重复执行：

```powershell
.\.venv\Scripts\python.exe -m app.scripts.build_kb_index --backend numpy
```

当 `MCP_ENABLED=true` 时，需要两个终端分别保持运行，**先启动 MCP，再启动 HTTP 服务**：

```powershell
# 终端 A：MCP 工具服务，默认监听 127.0.0.1:9123
.\.venv\Scripts\python.exe mcp_server/server.py
```

```powershell
# 终端 B：HTTP API 与内置前端，开发机仅绑定本机
.\.venv\Scripts\python.exe -m uvicorn app.server.api:app --host 127.0.0.1 --port 8000
```

打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)，在页面左侧注册或登录（密码至少 8 位），再连接会话。注册 `user-a` 可试用其演示订单；`user-b` 的订单与其隔离。浏览器聊天框按 Enter 发送、Shift+Enter 换行。首次登录后可按需要授权长期记忆；未授权不应沉淀跨会话记忆。

可以用浏览器访问 [http://127.0.0.1:8000/health/ready](http://127.0.0.1:8000/health/ready) 检查 API 就绪。此检查**不保证**上游模型、MCP 或 Qdrant 都可用；如聊天失败，还要看两个服务终端中的报错。

如果暂时不想启动 MCP，可在 `.env` 设置 `MCP_ENABLED=false` 并重启 API。本地工具仍可做演示，但这不等于验证了 MCP 链路。`MULTI_AGENT_ENABLED=false` 则切换到单 Agent；修改 `.env` 后均需重启对应进程。

### 前端验证工具调用

登录 `user-a` 后可依次问：

1. “查看我的所有订单”——应调用 `list_user_orders`，只返回当前用户可见的订单。
2. “查询订单 ORD-20240115-001 的物流”——应在 API 终端看到路由、`query_logistics` 调用和工具结果；仅看到售后 Agent 回复，不等于工具真的被调用。
3. “七天无理由退货有什么条件？”——应调用 `search_knowledge`。当前文档标记为演示资料，页面参考来源会显示演示标记，不能当作正式平台承诺。

API 终端出现 `🔗 [MCP] 已连接` 说明发现工具；出现 `🔧 [调用工具] ...` 或 `🔧 [小飞-…·调用工具] ...` 才说明该轮实际发起了工具调用。MCP 终端也应有 `/mcp` 请求记录。模型可根据问题选择不调用工具，业务事实必须以实际工具结果为准。

## 命令行使用

不走 HTTP 登录的本地演示入口：

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` 使用本地默认用户/会话，不适合拿来验证网页多用户身份隔离。支持 `quit`/`exit`、`reset`、`memory`、`memory on`、`memory off`、`skills` 等本地命令；这些控制命令不会作为普通聊天内容发送给 Agent。

需要在多个终端模拟独立用户时，先保持 HTTP 服务运行：

```powershell
# 第一次注册，按提示输入密码
.\.venv\Scripts\python.exe -m app.scripts.chat_client --user-id user-a --register
.\.venv\Scripts\python.exe -m app.scripts.chat_client --user-id user-b --register

# 后续登录；同一用户可用不同 session-id 隔离对话
.\.venv\Scripts\python.exe -m app.scripts.chat_client --user-id user-a --session-id after-sale
```

用户 ID 是账户标识，不是并发开关；不同用户/会话的上下文分开保存。终端客户端用登录令牌访问 API，服务端仍会核对身份。需要跨会话长期记忆时使用页面授权，或在终端命令增加 `--enable-long-term-memory`。

## 数据与知识库

| 数据 | 默认位置 | 说明 |
|---|---|---|
| 商品 | `app/sessions/ecom.db` | SQLite；可运行下方初始化命令导入 6 条演示商品 |
| 订单、物流、退款申请 | `app/sessions/orders.db` | SQLite 演示业务仓储，按登录用户限定可见范围 |
| 账户 | `app/sessions/auth.db` | 用户名和密码哈希；不要公开数据库或备份 |
| 会话与记忆 | `app/sessions/concurrent/`、`app/sessions/memory/` 等 | 包含用户数据，默认不纳入 Git |
| Markdown 政策 | `app/agent/rag/knowledge/` | `policies.json` 标注状态、版本和生效期；现有四份均为 `demo` |
| 默认向量索引 | `app/sessions/kb_index.json` | NumPy/JSON，适合本地演示；其他后端要分别构建 |

商品库可手动初始化：

```powershell
.\.venv\Scripts\python.exe -m app.scripts.init_database
```

RAG 构建会复用未变化片段的向量缓存，但修改 Markdown、政策元数据或 Embedding 模型后仍应重新发布索引。草稿和已退休政策不会进入新索引；检索也会过滤无效期或未发布结果。`RAG_ALLOW_DEMO_POLICIES=true` 仅用于演示；正式环境应设为 `false`，并先录入完成审核且有 HTTPS 来源的正式政策。`RAG_MIN_SCORE` 默认为 `0`，尚未经过黄金集校准，**不构成可靠的生产拒答阈值**。改进状态见 [RAG 需求改进](docs/rag需求改进.md)。

### 可选：切换到 Qdrant

先启动 Docker Desktop，确认 `docker info` 能连接引擎，再执行：

```powershell
docker compose -f compose.qdrant.yml up -d
.\.venv\Scripts\python.exe -m app.scripts.build_kb_index --backend qdrant
.\.venv\Scripts\python.exe -m app.scripts.manage_kb_index list
```

确认构建成功后，把 `.env` 的 `RAG_BACKEND` 改成 `qdrant`，并确保 `QDRANT_URL=http://127.0.0.1:6333`，然后**重启 API 和 MCP 服务**。Qdrant 开发编排仅绑定本机 6333 端口，数据使用 Docker 命名卷，镜像固定为 1.19.1。出错时可使用 `manage_kb_index rollback --to <list 输出的物理集合名>` 人工回滚别名；回滚会校验当前 Embedding 模型。不要删除仍由别名引用的集合。

## 常见问题

- **网页显示 500：** 先看运行 API 的终端 traceback，而不是 VS Code 的 Python 扩展日志。确认服务由 `.\.venv\Scripts\python.exe -m uvicorn ...` 启动、`.env` 模型密钥有效、模型与向量服务可达；必要时用 `/health/ready` 区分 API 未启动和聊天链路报错。
- **已连接 MCP，但 Agent 没调用工具：** “发现工具”和“本轮调用工具”是两件事。检查 API 终端有没有 `🔧` 调用日志；用上面的明确订单号问题测试。若 `MCP_ENABLED=true`，MCP 服务需单独保持运行。
- **知识库索引缺失或模型不匹配：** 使用当前 `.env` 的 `EMBEDDING_MODEL` 重新运行 `build_kb_index --backend <当前后端>`；不要把旧模型向量与新模型混用。
- **切换配置后仍是旧行为：** `.env` 在进程启动时读取；分别重启受影响的 API、MCP 或 `main.py`。生产部署一般用受控滚动发布，不靠人工每次重启整个平台。
- **找不到别人的演示订单：** 这是预期的用户隔离；`user-a` 和 `user-b` 各自只能查自己的订单。演示订单/退款不会连接真实电商平台。

## 测试与部署边界

不调用真实模型的相关回归测试可运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_qdrant_backend.py tests/test_rag_policy_evidence.py tests/test_frontend.py tests/test_mcp_requirements.py
```

`tests/test_rag.py`、`tests/test_agent.py` 等部分端到端测试可能调用真实模型/Embedding API，不应把 `pytest` 全量运行视为零成本离线检查。离线 Agent 评估入口是 `.\.venv\Scripts\python.exe -m app.scripts.run_eval --help`；其沙箱与 Judge 配置应按实际成本选择。

根目录 `Dockerfile` 和 `deploy/k8s.yaml` 是部署**示例**，不是可直接上线的完整生产方案。Kubernetes 示例现采用单副本、Recreate 更新策略和持久卷，已移除自动扩容；当前必须保持单 API worker，分布式状态协调完成前不要增加副本。需要集群提供可用的 StorageClass；Recreate 更新期间有短暂不可用。生产还必须配置共享密钥、TLS、正式业务数据源、Qdrant 鉴权与备份、政策审核和质量评估。整改记录见 [综合审查](docs/综合审查.md)。

### 综合审查后的运行变化（2026-10-04）

- 安装测试依赖使用 `python -m pip install -r requirements-dev.txt`。CI 采用 Python 3.12，离线检查覆盖认证、记忆、MCP、RAG、Multi-Agent 和接口契约。
- 用户名大小写变体不允许注册为两个账户，Windows 保留名称和结尾点被拒绝。历史账户如存在路径冲突，服务拒绝启动并提示人工核对，不会自动合并或删除数据。
- 会话文件保留 `.json` 扩展名，但新保存内容已加密；通过 `load_session` 读取。旧明文文件在下次保存时迁移。生产请注入独立的 `SESSION_ENCRYPTION_KEY`（有效 Fernet 密钥）；未配置时开发密钥保存在会话父目录的 `.keys/.memory.key`。备份必须保留正确密钥，不要随意删除或直接换密钥。
- `LOG_MODEL_CONTENT=false` 默认关闭模型正文日志；工具执行摘要继续保留。关闭长期记忆授权不等于删除原始会话。
- 登录/注册默认每来源 IP 每分钟最多 20 次、同时最多 4 个请求，通过 `AUTH_ATTEMPTS_PER_MINUTE`、`AUTH_MAX_CONCURRENCY` 调整；代理可信来源需在部署层正确配置。
- 页面支持退出撤销当前令牌，刷新后通过服务端校验身份；退款响应丢失时，可在“查询退款进度”中留空申请编号，核实本页最近一次提交。
- 空闲 Agent 默认 30 分钟后回收，已保存会话继续保留；配置项为 `SERVER_SESSION_IDLE_SECONDS`。健康显示“API 已连接”只代表 API 可达，不证明所有上游可用。
## 2026-10-10 全链路评估升级

新增分模块工程基线、严格指标口径、隔离工具评测和前端质量评测台。完整变更及隔离边界见 [2026-10-10 全链路评估与升级](docs/10.10更新-2026-10-10全链路评估与升级.md)。

```bash
python -m app.evaluation.benchmark --output reports/engineering-latest.json
```

Windows / Anaconda 用户请使用项目虚拟环境执行，避免误用缺少依赖的 `(base)` Python：

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.benchmark
```

2026-10-11 修复：依赖版本元数据缺失时提前警告，报告将该版本记录为 `null` 并保留测试结果；实际测试失败仍返回非零退出码。报告同时记录 `python_executable` 和 `missing_dependency_metadata`，便于排查解释器差异。

启动服务后访问 `/quality`，本地导入报告查看分项指标与基线差异。工程测试通过率不等于真实模型准确率。真实模型评测现在需要显式 `--live --tool-fixtures PATH`；容器离线评测配置见 `Dockerfile.evaluation`。
