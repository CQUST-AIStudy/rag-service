# CQUST RAG Service

`rag-service` 是 CQUST AIStudy 平台的独立 RAG（Retrieval-Augmented Generation，检索增强生成）服务。它基于 **uv + FastAPI + LangChain + Chroma + SQLite** 构建，为教师端课程知识库提供课程空间管理、资料入库、向量检索、流式问答、引用来源、切片标注和 RAG 分析能力。

服务默认监听 `8001`，业务接口统一挂载在 `/rag` 前缀下；健康检查接口为 `/health`。

## 快速启动

### 1. 准备环境

- Python `>= 3.11`
- `uv`
- DashScope API Key（文档入库、检索重排和问答需要）
- 与 Java 后端一致的 JWT 配置（需要联调前端登录态时）

### 2. 安装依赖

```powershell
cd F:\WorkSpace\Coding\CQUST-AIStudy\rag-service
uv sync
```

### 3. 配置 `.env`

```powershell
Copy-Item .env.example .env
```

至少确认这些配置：

```env
DASHSCOPE_API_KEY=你的 DashScope Key
RAG_JWT_SECRET=与 Java 后端 JWT_SECRET 保持一致
RAG_JWT_ISSUER=tap
RAG_DATA_DIR=./data
```

本地开发如果未配置 `RAG_JWT_SECRET`，服务会使用开发身份放行；生产环境必须配置，否则鉴权会拒绝请求。

### 4. 启动服务

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

启动后检查：

- 健康检查：`http://127.0.0.1:8001/health`
- Swagger：`http://127.0.0.1:8001/docs`
- 课程空间列表：`GET http://127.0.0.1:8001/rag/knowledge-base/list`

## Docker 一键部署

### 1. 准备 Docker 环境变量

```powershell
Copy-Item .env.docker.example .env.docker
```

至少修改以下配置：

```env
DASHSCOPE_API_KEY=你的 DashScope Key
RAG_JWT_SECRET=与 Java 后端 JWT_SECRET 保持一致
RAG_JWT_ISSUER=tap
```

Docker 部署默认使用生产模式：`RAG_ENV=production`。生产模式下必须配置 `RAG_JWT_SECRET`，否则鉴权接口会拒绝请求。

### 2. 构建并启动

```powershell
docker compose up -d --build
```

查看日志：

```powershell
docker compose logs -f rag-service
```

启动后检查：

- 健康检查：`http://127.0.0.1:8001/health`
- Swagger：`http://127.0.0.1:8001/docs`

停止服务：

```powershell
docker compose down
```

如需同时删除运行时数据卷：

```powershell
docker compose down -v
```

### 3. 数据持久化

Docker Compose 会创建 `rag_data` 数据卷并挂载到容器内 `/app/data`。SQLite、Chroma 和上传文件都会写入该目录，容器重建不会丢失数据。

容器内固定监听 `8001`。如果要更换宿主机对外端口，可在执行 compose 前设置 `RAG_HOST_PORT`，例如：

```powershell
$env:RAG_HOST_PORT = "18001"
docker compose up -d --build
```

此时访问地址为 `http://127.0.0.1:18001/health`。

### 4. 外部服务联动

当前 Compose 只部署 RAG 服务。与外部 Java 后端和前端联动时，需要确认：

- `RAG_JWT_SECRET` / `RAG_JWT_ISSUER` 与 Java 后端运行时配置一致。
- `RAG_ALLOWED_ORIGINS` 包含前端实际访问源。
- 前端代理 `/rag` 指向 Docker 暴露的 RAG 地址，例如 `http://127.0.0.1:8001`。

## 前端联调

前端默认通过同源 `/rag` 访问 RAG 服务。开发环境需要确保 `frontend-repo/vue.config.js` 中的代理指向：

```js
'^/rag(?:/|$)': {
  target: 'http://127.0.0.1:8001',
  changeOrigin: true,
}
```

联调链路：

1. 启动 Java 后端 `8081`。
2. 启动 RAG 服务 `8001`。
3. 启动前端 `8080`。
4. 登录教师账号。
5. 打开“课程知识库”，确认 `/rag/knowledge-base/list` 返回 `200 success`。

如果前端已登录但 RAG 返回 `Missing Bearer token`，通常是前端没有拿到 `tap_token`；重新登录或确认 `/api/auth/session` 是否能换票。

如果 RAG 返回 `Invalid Bearer token`，优先检查 `RAG_JWT_SECRET` / `RAG_JWT_ISSUER` 是否和后端运行时配置一致，并重启 RAG 服务。当前 RAG 支持 `HS256`、`HS384`、`HS512` 三种 HMAC JWT 算法，以兼容 Java `jjwt` 根据密钥长度选择签名算法的行为。

## 功能概览

| 能力 | 说明 |
|------|------|
| 课程空间 | 创建、更新、删除课程知识库，可按 `courseId` 查询 |
| 文档入库 | 支持 PDF、DOCX、TXT、Markdown 上传与处理 |
| 向量检索 | Chroma 持久化向量，结合 SQLite 维护业务数据 |
| 混合召回 | 语义检索 + BM25 稀疏召回 |
| 重排 | 可调用 DashScope rerank 提升引用质量 |
| 问答 | 支持普通问答和 SSE 流式问答 |
| 引用 | 返回来源文档、切片、页码/章节等元数据 |
| 分析 | 热门问题、命中率、引用覆盖、Web 触发率、反馈统计 |
| Web 兜底 | `open` 模式下可使用 Tavily 进行外部资料补充 |

## 目录结构

```text
rag-service/
├── app/
│   ├── api/            # FastAPI 路由：knowledge_base / document / chat / compat / health
│   ├── core/           # 配置、鉴权、统一响应
│   ├── schemas/        # Pydantic 请求和响应模型
│   └── services/       # 数据库、向量库、DashScope、入库、问答链路
├── tests/              # pytest 测试
├── docs/               # 设计文档
├── data/               # 运行时数据：SQLite / Chroma / uploads
├── Dockerfile          # Docker 镜像构建文件
├── docker-compose.yml  # Docker Compose 单服务部署
├── pyproject.toml      # 依赖和工具配置
├── uv.lock             # uv 锁文件
└── .env                # 本地环境变量，不应提交真实密钥
```

## 环境变量

### 基础配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_SERVICE_NAME` | FastAPI 服务标题 | `CQUST RAG Service` |
| `RAG_ENV` | 运行环境，`local` / `production` | `local` |
| `RAG_HOST` | 监听地址 | `0.0.0.0` |
| `RAG_PORT` | 监听端口 | `8001` |
| `RAG_DATA_DIR` | 数据根目录 | `./data` |
| `RAG_ALLOWED_ORIGINS` | CORS 白名单，逗号分隔 | `http://localhost:8080,http://127.0.0.1:8080` |
| `RAG_MAX_UPLOAD_MB` | 单文件上传大小上限 | `50` |

### 鉴权配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_JWT_SECRET` / `JWT_SECRET` | JWT 密钥，必须与 Java 后端运行时配置一致 | 空 |
| `RAG_JWT_ISSUER` / `JWT_ISSUER` | JWT 签发方 | `tap` |

RAG 会从 `Authorization: Bearer <token>` 中解析用户身份，支持 payload 中的 `uid` 和 `sub`。支持的 JWT 算法：`HS256`、`HS384`、`HS512`。

### DashScope 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API Key | 空 |
| `DASHSCOPE_COMPAT_BASE_URL` | OpenAI 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_RERANK_URL` | DashScope rerank 接口地址 | 见 `app/core/config.py` |
| `RAG_EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `RAG_EMBEDDING_DIMENSIONS` | 嵌入维度 | `1024` |
| `RAG_RERANK_MODEL` | 重排模型 | `qwen3-vl-rerank` |
| `RAG_CHAT_MODEL` | 问答模型 | `qwen-plus` |

### 检索配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_DEFAULT_TOP_K` | 向量召回条数 | `30` |
| `RAG_DEFAULT_RERANK_TOP_N` | 重排后保留条数 | `5` |
| `RAG_DEFAULT_SCORE_THRESHOLD` | 最低相似度阈值 | `0.0` |
| `RAG_COVERAGE_THRESHOLD` | 触发 Web 兜底的覆盖率阈值 | `0.4` |

### Web 兜底配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_WEB_FALLBACK_ENABLED` | 是否启用 Web 兜底 | `true` |
| `RAG_WEB_MAX_RESULTS` | Web 兜底最大结果数 | `5` |
| `TAVILY_API_KEY` | Tavily API Key，未配置时不能执行 Web 兜底 | 空 |
| `TAVILY_SEARCH_URL` | Tavily 搜索接口 | `https://api.tavily.com/search` |

## 文档入库流程

```text
上传文件
  -> 文本抽取
  -> 按知识库配置切片
  -> SQLite 保存文档和切片
  -> DashScope 批量 embedding
  -> Chroma 写入向量
  -> 文档状态变为 completed
```

支持格式：`.pdf`、`.docx`、`.txt`、`.md`。

| 状态 | 含义 |
|------|------|
| `processing` | 已上传，正在抽取、切片或写入向量 |
| `completed` | SQLite 切片和 Chroma 向量均就绪，可检索 |
| `failed` | 入库失败，服务会清理残留切片和向量，并记录错误原因 |

`text-embedding-v4` 有批量限制，服务会按批次调用 DashScope 并保持结果顺序。

## 检索与问答

检索流程：

1. 根据用户问题进行 Chroma 向量召回。
2. 可选：使用 BM25 补充稀疏召回。
3. 可选：调用 DashScope rerank 重排。
4. 按分数阈值过滤。
5. 计算覆盖率；`open` 模式下覆盖不足时可触发 Web 兜底。
6. 生成回答并返回引用来源。

问答模式：

| 模式 | 行为 |
|------|------|
| `strict` | 只基于课程资料回答，资料不足时明确说明 |
| `open` | 允许结合模型知识和 Web 兜底补充回答 |

SSE 流式事件：

| 事件 | 内容 |
|------|------|
| `retrieval` | 检索结果、来源、覆盖率 |
| `delta` | 模型增量文本 |
| `done` | 会话 ID、用量信息 |
| `error` | 错误信息 |

## API 概览

### 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/knowledge-base` | 创建知识库 |
| `GET` | `/rag/knowledge-base/list` | 查询知识库列表，支持 `courseId` |
| `PUT` | `/rag/knowledge-base/{id}` | 更新知识库 |
| `DELETE` | `/rag/knowledge-base/{id}` | 删除知识库及关联资源 |

### 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/document/upload` | 上传文档 |
| `POST` | `/rag/knowledge-base/{id}/documents` | 兼容上传接口 |
| `GET` | `/rag/document/{id}/status` | 查询文档状态 |
| `DELETE` | `/rag/document/{id}` | 删除文档和向量 |
| `GET` | `/rag/knowledge-base/{id}/documents` | 查询知识库文档 |
| `GET` | `/rag/knowledge-base/{id}/documents/status-summary` | 查询文档状态汇总 |
| `POST` | `/rag/knowledge-base/{id}/documents/reprocess` | 重处理全部文档 |
| `POST` | `/rag/knowledge-base/{id}/documents/{doc_id}/reprocess` | 重处理单个文档 |

### 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/retrieve` | 仅检索，不生成回答 |
| `POST` | `/rag/chat` | 非流式问答 |
| `POST` | `/rag/chat/stream` | SSE 流式问答 |
| `POST` | `/rag/chat/legacy-stream` | 兼容旧格式流式问答 |
| `GET` | `/rag/conversation/{id}/history` | 查询会话历史 |
| `DELETE` | `/rag/conversation/{id}` | 删除会话 |

### 标注与分析

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/rag/knowledge-base/{id}/chunks` | 查询切片 |
| `POST` | `/rag/knowledge-base/{id}/annotations` | 创建切片标注 |
| `GET` | `/rag/knowledge-base/{id}/annotations` | 查询标注 |
| `DELETE` | `/rag/annotations/{id}` | 删除标注 |
| `POST` | `/rag/feedback` | 提交问答反馈 |
| `GET` | `/rag/knowledge-base/{id}/analytics` | 分析总览 |
| `GET` | `/rag/knowledge-base/{id}/analytics/hot-questions` | 热门问题 |
| `GET` | `/rag/knowledge-base/{id}/analytics/hit-rate` | 命中率 |
| `GET` | `/rag/knowledge-base/{id}/analytics/citation-coverage` | 引用覆盖 |
| `GET` | `/rag/knowledge-base/{id}/analytics/web-trigger-rate` | Web 触发率 |
| `GET` | `/rag/knowledge-base/{id}/analytics/feedback-stats` | 反馈统计 |
| `GET` | `/rag/knowledge-base/{id}/analytics/resource-gaps` | 资源缺口 |

## 响应格式

标准响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

错误响应：

```json
{
  "code": 401,
  "message": "Invalid Bearer token",
  "data": null
}
```

## 开发与验证

常用命令：

```powershell
# 运行全部测试
uv run pytest

# 运行鉴权相关测试
uv run pytest tests/test_auth.py

# 静态检查
uv run ruff check .

# 自动修复可安全修复的问题
uv run ruff check . --fix
```

提交前建议至少执行：

```powershell
uv run pytest
uv run ruff check .
```

测试用例应显式隔离本地 `.env`，避免私有配置影响测试期望。真实 API Key、JWT Secret、Token 不应写入测试、文档或提交记录。

## 常见问题

### 前端提示“加载课程知识库失败”

排查顺序：

1. `http://127.0.0.1:8001/health` 是否返回 `200 success`。
2. 前端代理 `/rag` 是否指向 `http://127.0.0.1:8001`。
3. 浏览器请求 `/rag/knowledge-base/list` 是否带有 `Authorization`。
4. 如果返回 `Missing Bearer token`，检查前端是否拿到 `tap_token`，以及 `/api/auth/session` 是否能换票。
5. 如果返回 `Invalid Bearer token`，检查 RAG 与 Java 后端的 `JWT_SECRET` / `JWT_ISSUER` 是否一致，并重启 RAG。

### DashScope 返回 HTTP 400

排查顺序：

1. `DASHSCOPE_API_KEY` 是否有效。
2. embedding 模型和 `RAG_EMBEDDING_DIMENSIONS` 是否匹配。
3. 输入文本是否超过模型限制。
4. 查看文档处理状态中的错误详情、DashScope `code/message/request_id`。

### 上传失败但切片里有内容

这是旧版本可能留下的半成功数据。当前版本失败时会清理切片和向量，并写入错误原因。历史遗留数据可以通过“重处理”修复。

### 问答没有命中课程资料

排查顺序：

1. 文档状态是否为 `completed`。
2. 请求的 `knowledgeBaseIds` 是否包含目标知识库。
3. `scoreThreshold` 是否设置过高。
4. `RAG_DATA_DIR` 是否指向同一套 `rag.sqlite3` 和 `chroma` 数据。
5. 当前用户是否有目标课程空间访问权限。

### Web 兜底未触发

确认：

- `TAVILY_API_KEY` 已配置。
- `RAG_WEB_FALLBACK_ENABLED=true`。
- 问答模式为 `open`。
- 检索覆盖率低于 `RAG_COVERAGE_THRESHOLD`。

## 运行边界

- 删除知识库会同步删除 SQLite 中的知识库、文档、切片记录，并清理 Chroma 中对应向量。
- 重处理单个文档时，路径中的知识库 ID 必须与文档真实归属一致。
- 创建切片标注时，`chunkId` 必须存在且属于当前知识库。
- `data/` 是本地运行时目录，不应作为源码修改提交。

## License

Internal project for CQUST AIStudy platform.
