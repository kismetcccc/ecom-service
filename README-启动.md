# 本地启动说明

更新日期：2026-10-11。适用于 Windows PowerShell。

## 推荐：本机 Docker 一键启动

1. 打开 **Docker Desktop**，等待引擎运行。
2. 保留根目录已配置好的 `.env`。停止之前手动启动的聊天后端和 MCP，避免两套应用同时写相同会话。
3. 双击根目录 **`start-local.cmd`**。首次会下载 Python 镜像并安装依赖，耗时较长；后续使用构建缓存。
4. 启动检查成功后自动打开 **http://127.0.0.1:18000/**，使用原账号登录。
5. 结束使用时双击 **`stop-local.cmd`**；它只停止应用，保留原有 Qdrant 服务。

应用容器内部同时运行 MCP 和聊天后端；Qdrant 使用独立容器，启动脚本统一管理。只对本机开放网页端口，不需要部署服务器，也不需要本机 Python 环境。网页端口与下面的手动启动方式不同。

- 数据：直接挂载已有 `app/sessions`，保留订单、账号、会话、密钥和 Wiki；知识源只读挂载。Qdrant 复用现有 `ecom-service-agent-main_qdrant_data` 数据卷。
- 配置：容器读取 `.env`，覆盖 MCP 地址与 Qdrant 地址以适配容器网络，并开启多 Agent/MCP。自定义路径需使用容器可访问的项目相对路径，不能使用 Windows 盘符。
- 更新：修改代码或 `.env` 后重新双击启动脚本，会重新构建并更新容器。不要删除数据卷来解决启动问题。
- 重启电脑后：Docker Desktop 启动时，未手动停止的容器可按重启策略恢复；也可再次双击脚本。

查看状态及日志（项目根目录执行）：

```powershell
docker compose -f compose.qdrant.yml -f compose.local.yml ps
docker compose -f compose.qdrant.yml -f compose.local.yml logs --tail 100 app
# 需要停止包括 Qdrant 在内的全部服务时执行；保留数据
docker compose -f compose.qdrant.yml -f compose.local.yml stop
```

健康检查验证后端就绪和 MCP 工具目录，不调用付费模型；模型、订单和 RAG 是否可用仍以实际查询结果为准。以下内容为不使用 Docker 时的手动启动方式。

2026-10-11 本机验收：镜像构建成功，应用健康检查通过，网页返回 HTTP 200；容器内真实 MCP 订单、物流查询成功，RAG 返回 3 条结果，模型最小请求返回有效文本。此检查不是所有业务流程的全面回归。首次下载官方 Python 镜像曾网络超时，重试成功；遇到相同问题可先执行 `docker pull python:3.12-slim` 再重新双击启动脚本。

**启动顺序：Qdrant（配置需要时）→ MCP 工具服务 → 聊天后端 → 浏览器。**
MCP 与聊天后端是两个独立服务，两个终端都要保持运行。前端由聊天后端提供，无需另开前端服务。

## 1. 首次准备（已有环境可跳过）

```powershell
cd D:\project_sum\agent_project\ecom-service-agent-main

# 仅在虚拟环境不存在时创建，需要已安装 Python 3.12
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 仅复制缺失的配置，保留已有密钥和数据配置
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在 `.env` 中确认模型密钥、接口地址和模型名称已配置；使用 MCP 多 Agent 时确认以下配置：

```dotenv
MCP_ENABLED=true
MCP_SERVER_URL=http://127.0.0.1:9123/mcp
MULTI_AGENT_ENABLED=true
MULTI_AGENT_REVIEW_MODE=enforce
MULTI_AGENT_REVIEW_FAST_MODE=true
```

`MULTI_AGENT_REVIEW_FAST_MODE` 对已适配的百炼 DeepSeek 模型关闭质检深度思考，仍执行审核。其他配置保留现有值。所有命令使用项目 `.venv`，避免误用 Anaconda 的 `base` 环境。

## 2. 启动向量服务（仅 RAG_BACKEND=qdrant 时）

先打开 Docker Desktop，等待引擎启动，再执行：

```powershell
cd D:\project_sum\agent_project\ecom-service-agent-main
docker compose -f compose.qdrant.yml up -d
docker compose -f compose.qdrant.yml ps
```

Qdrant 在后台运行，默认端口 `6333`。使用其他后端时跳过本步。日常启动不需要重建索引或初始化数据库；首次部署知识库请参考主 README 的“数据与知识库”。

## 3. 终端 A：启动 MCP

```powershell
cd D:\project_sum\agent_project\ecom-service-agent-main
.\.venv\Scripts\python.exe mcp_server/server.py
```

看到服务监听 `127.0.0.1:9123` 后，保持这个终端运行。MCP 提供订单、物流、知识检索等工具。

## 4. 终端 B：启动聊天后端

新开一个 PowerShell 终端：

```powershell
cd D:\project_sum\agent_project\ecom-service-agent-main
.\.venv\Scripts\python.exe -m uvicorn app.server.api:app --host 127.0.0.1 --port 8000
```

看到 `Application startup complete` 后，保持这个终端运行。后端负责网页、登录、Agent 调度和模型调用。`python main.py` 是命令行聊天入口，不代替网页后端。

## 5. 浏览器使用

1. 打开 <http://127.0.0.1:8000/>。
2. 注册或登录，再连接会话。演示订单 `ORD-20240115-001` 属于 `user-a`；已有该账号请直接登录，其他账号不能查询它。
3. 先发送“查看我的所有订单”，确认工具查询成功。
4. 再发送“订单 ORD-20240115-001 的物流一直不动”，验证履约诊断。

订单状态以数据库当前记录为准；之前提交过退款的演示订单可能已处于退款处理中。

## 6. 快速检查与常见问题

可在第三个终端执行：

```powershell
# 只检查 MCP 端口是否能连接，不代表业务工具调用成功
Test-NetConnection 127.0.0.1 -Port 9123

# 只检查聊天后端就绪，不代表模型、MCP、RAG 都正常
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

| 现象 | 检查方法 |
|---|---|
| 网页打不开 | 检查终端 B 是否启动，访问地址是否为 8000 端口 |
| `query_order` 返回 `UNAVAILABLE` | 检查终端 A 的报错；若 MCP 刚重启，再重启终端 B 以重新建立连接 |
| 工具显示 `UNAUTHORIZED` / `NOT_FOUND` | 确认登录账号及订单归属，不要通过切换到其他身份绕过校验 |
| 政策检索失败 | 查看 `search_knowledge` 的错误码；使用 Qdrant 时检查 Docker 和索引是否可用 |
| 模型或质检超时 | 查看终端 B 中对应 `request_id` 的日志；服务启动不代表模型接口一定可用 |
| 端口已被占用 | 检查是否已经启动了相同服务，不要重复启动第二份 |

## 7. 修改代码后的重启顺序

1. 在终端 B 按 `Ctrl+C` 停止聊天后端。
2. 修改了 MCP、工具、共享配置或 `.env` 时，在终端 A 按 `Ctrl+C`，重新执行第 3 步。
3. 等 MCP 启动完成后，重新执行第 4 步。
4. 刷新浏览器；前端内容没更新时按 `Ctrl+F5`。

上述命令未启用自动重载，保存 Python 文件不会自动更新运行中的服务。日常关停先停聊天后端，再停 MCP；Qdrant 按需要用 `docker compose -f compose.qdrant.yml stop` 停止，保留数据卷。
