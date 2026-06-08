# CQUST RAG Service

`rag-service` 是 CQUST AIStudy 平台的独立 RAG（检索增强生成）服务，基于 **uv + FastAPI + LangChain** 构建。

负责：知识库管理、多格式文档入库、向量检索与重排、课程问答（流式/非流式）、会话历史、切片标注和 RAG 分析。

所有业务接口统一挂载在 `/rag` 前缀下，默认监听 **8001** 端口。

## 目录结构

```
rag-service/
├── app/
│   ├── api/            # 路由层（knowledge_base / document / chat / compat / health）
│   ├── core/           # 配置、鉴权、统一响应
│   ├── schemas/        # Pydantic 请求/响应模型
│   └── services/       # 业务逻辑（database / vector_store / dashscope / document_ingest / rag_chain / repository）
├── tests/              # pytest 测试
├── docs/               # 文档
├── data/               # 运行时数据（SQLite / Chroma / uploads）
├── pyproject.toml      # 项目依赖与工具配置
└── .env                # 环境变量（不入库）
```

## 技术栈

| 组件 | 用途 |
|------|------|
| FastAPI | HTTP API、依赖注入、CORS、异常处理 |
| LangChain | 模型调用、Prompt 编排、文档处理 |
| Chroma | 本地向量存储（持久化到 `./data/chroma`） |
| SQLite | 知识库、文档、切片、会话、反馈和分析日志 |
| DashScope | embedding（text-embedding-v4）、rerank（qwen3-vl-rerank）、chat（qwen-plus） |
| BM25 | 稀疏检索，混合召回补充语义检索 |
| Tavily | 开放模式下的可选 Web 兜底搜索 |
| jieba | 中文分词（BM25 需要） |

## 快速启动

### 前置条件

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器
- DashScope API Key（阿里云百炼）

### 安装与运行

```bash
# 安装依赖
uv sync

# 创建 .env（至少配置 DASHSCOPE_API_KEY）
cp .env.example .env  # 或手动新建

# 启动服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

启动后可访问：

- 健康检查：`GET http://127.0.0.1:8001/health`
- Swagger 文档：`http://127.0.0.1:8001/docs`
- 前端联调：确保前端代理将 `/rag` 转发到 `http://127.0.0.1:8001`

## 环境变量

服务读取项目根目录下的 `.env` 文件，也支持系统环境变量。

### 基础配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_SERVICE_NAME` | FastAPI 服务标题 | `CQUST RAG Service` |
| `RAG_ENV` | 运行环境（`local` / `production`） | `local` |
| `RAG_HOST` | 监听地址 | `0.0.0.0` |
| `RAG_PORT` | 监听端口 | `8001` |
| `RAG_DATA_DIR` | 数据根目录（SQLite / Chroma / uploads） | `./data` |
| `RAG_ALLOWED_ORIGINS` | CORS 白名单，逗号分隔 | `http://localhost:8080,http://127.0.0.1:8080` |
| `RAG_MAX_UPLOAD_MB` | 单文件最大上传大小（MB） | `50` |

### 鉴权配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_JWT_SECRET` / `JWT_SECRET` | JWT 密钥，与 `backend` 保持一致 | 空 |
| `RAG_JWT_ISSUER` / `JWT_ISSUER` | JWT 签发方 | `tap` |

### 模型配置（DashScope）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API Key（必须） | 空 |
| `DASHSCOPE_COMPAT_BASE_URL` | OpenAI 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_RERANK_URL` | 重排接口地址 | 见 `app/core/config.py` |
| `RAG_EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `RAG_EMBEDDING_DIMENSIONS` | 嵌入维度 | `1024` |
| `RAG_RERANK_MODEL` | 重排模型 | `qwen3-vl-rerank` |
| `RAG_CHAT_MODEL` | 对话模型 | `qwen-plus` |

### 检索配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_DEFAULT_TOP_K` | 向量召回条数 | `30` |
| `RAG_DEFAULT_RERANK_TOP_N` | 重排后保留条数 | `5` |
| `RAG_DEFAULT_SCORE_THRESHOLD` | 最低相似度阈值 | `0.0` |
| `RAG_COVERAGE_THRESHOLD` | 触发 Web 兜底的覆盖率阈值 | `0.4` |

### Web 兜底（Tavily）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TAVILY_API_KEY` | Tavily API Key，未配置时 Web 兜底不可用 | 空 |
| `RAG_WEB_FALLBACK_ENABLED` | 是否启用 Web 兜底 | `true` |
| `RAG_WEB_MAX_RESULTS` | Web 兜底最大结果数 | `5` |

## 鉴权说明

- 请求携带 `Authorization: Bearer <token>`，服务解析 JWT 获取用户身份。
- **本地开发**：未配置 `RAG_JWT_SECRET` 时，服务自动使用开发身份放行。
- **生产环境**：必须配置 `RAG_JWT_SECRET`，否则接口拒绝访问。
- 支持 JWT payload 中的 `uid` 和 `sub` 两种 claim 格式。

## 文档入库流程

```
上传文件 → 文本抽取 → 按配置切片 → SQLite 持久化 → DashScope 批量 Embedding → Chroma 写入
```

支持格式：PDF、DOCX、TXT、Markdown。

### 状态流转

| 状态 | 含义 |
|------|------|
| `processing` | 已上传，正在抽取 / 切片 / 写入向量 |
| `completed` | SQLite 切片 + Chroma 向量均已就绪，可用于检索 |
| `failed` | 入库失败，已自动清理残留切片和向量，错误原因记录在文档状态中 |

> `text-embedding-v4` 单次请求有批量限制，服务自动按 10 条一批调用 DashScope 并保持顺序。

## 检索与问答

### 检索流程

1. 用户提问 → 向量相似度召回（Chroma，top_k=30）
2. 可选：BM25 混合召回补充
3. 可选：DashScope rerank 重排取 top_n
4. 分数阈值过滤
5. 覆盖率评估 → 低于阈值时触发 Web 兜底（Tavily）

### 问答模式

| 模式 | 行为 |
|------|------|
| `strict` | 仅基于检索结果回答，无资料时明确告知 |
| `open` | 允许结合模型自身知识 + Web 兜底补充回答 |

### 流式响应（SSE）

流式问答使用 `text/event-stream`，事件类型：

| 事件 | 内容 |
|------|------|
| `retrieval` | 检索来源、模式、覆盖率 |
| `delta` | 模型增量文本 |
| `done` | 会话 ID、用量信息 |
| `error` | 错误信息 |

## API 概览

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/knowledge-base` | 创建知识库 |
| `GET` | `/rag/knowledge-base/list` | 查询知识库列表（支持 courseId 筛选） |
| `PUT` | `/rag/knowledge-base/{id}` | 更新知识库 |
| `DELETE` | `/rag/knowledge-base/{id}` | 删除知识库及关联资源 |

### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/document/upload` | 上传文档到知识库 |
| `GET` | `/rag/document/{id}/status` | 查询文档处理状态 |
| `DELETE` | `/rag/document/{id}` | 删除文档及向量 |
| `GET` | `/rag/knowledge-base/{id}/documents` | 查询知识库文档列表 |
| `GET` | `/rag/knowledge-base/{id}/documents/status-summary` | 文档状态汇总 |
| `POST` | `/rag/knowledge-base/{id}/documents/reprocess` | 重处理全部文档 |
| `POST` | `/rag/knowledge-base/{id}/documents/{doc_id}/reprocess` | 重处理单个文档 |

### 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/retrieve` | 仅检索（不生成回答） |
| `POST` | `/rag/chat` | 非流式问答 |
| `POST` | `/rag/chat/stream` | SSE 流式问答 |
| `POST` | `/rag/chat/legacy-stream` | 兼容旧格式的流式问答 |
| `GET` | `/rag/conversation/{id}/history` | 查询会话历史 |
| `DELETE` | `/rag/conversation/{id}` | 删除会话 |

### 标注与分析

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/rag/knowledge-base/{id}/chunks` | 查询切片列表 |
| `POST` | `/rag/knowledge-base/{id}/annotations` | 创建切片标注 |
| `GET` | `/rag/knowledge-base/{id}/annotations` | 查询标注列表 |
| `DELETE` | `/rag/annotations/{id}` | 删除标注 |
| `POST` | `/rag/feedback` | 提交问答反馈 |
| `GET` | `/rag/knowledge-base/{id}/analytics` | 分析总览 |
| `GET` | `/rag/knowledge-base/{id}/analytics/hot-questions` | 热门问题 |
| `GET` | `/rag/knowledge-base/{id}/analytics/hit-rate` | 命中率 |
| `GET` | `/rag/knowledge-base/{id}/analytics/citation-coverage` | 引用覆盖 |
| `GET` | `/rag/knowledge-base/{id}/analytics/web-trigger-rate` | Web 触发率 |
| `GET` | `/rag/knowledge-base/{id}/analytics/feedback-stats` | 反馈统计 |
| `GET` | `/rag/knowledge-base/{id}/analytics/resource-gaps` | 资源缺口 |

### 兼容接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rag/knowledge-base/{id}/documents` | 兼容文档上传 |
| `POST` | `/rag/knowledge-base/{id}/rebuild-bm25` | 兼容 BM25 重建 |

## 响应格式

### 标准响应

```json
{ "code": 200, "message": "success", "data": { ... } }
```

### 错误响应

```json
{ "code": 400, "message": "错误描述", "data": null }
```

## 开发指南

### 常用命令

```bash
# 运行测试
uv run pytest

# 代码检查（只检查，不自动改文件）
uv run ruff check .

# 自动修复
uv run ruff check . --fix
```

### 验证与测试配置

- 提交前至少执行 `uv run pytest` 和 `uv run ruff check .`，确保测试与静态检查均通过。
- 服务运行时会读取项目根目录 `.env`；测试用例应显式隔离本地 `.env`，避免私有配置（如 `RAG_DEFAULT_RERANK_TOP_N`）改变测试期望。
- 测试和文档示例不得依赖真实密钥，`DASHSCOPE_API_KEY`、`TAVILY_API_KEY` 等敏感配置只应写入本地 `.env` 或系统环境变量。

### 本地联调步骤

1. 配置 `.env`（至少填入 `DASHSCOPE_API_KEY`）
2. 启动服务：`uv run uvicorn app.main:app --port 8001 --reload`
3. 前端课程知识库页面上传文档
4. 等待文档状态变为 `completed`
5. 使用"知识问答"功能提问验证

### 接口边界说明

- 删除知识库会同步删除 SQLite 中的知识库、文档和切片记录，并清理该知识库在 Chroma 中的向量数据。
- 重处理单个文档时，路径中的知识库 ID 必须与文档真实归属一致，不一致会被拒绝，避免跨知识库误操作。
- 创建切片标注时，`chunkId` 必须存在且属于当前知识库，不允许引用其他知识库的切片。

### 代码风格

- Ruff 检查：`line-length = 120`，Python 3.11+ 特性
- Lint 规则：`E`（pycodestyle）、`F`（pyflakes）、`I`（isort）、`UP`（pyupgrade）、`B`（bugbear）

## 常见问题

### 上传显示失败，但分块里有内容

旧版本的半成功状态。当前版本失败时会清理切片和向量，并写入错误原因。历史遗留数据需点击"重处理"修复。

### DashScope 返回 HTTP 400

排查顺序：
1. `DASHSCOPE_API_KEY` 是否配置正确
2. embedding 模型与维度是否匹配
3. 输入文本是否超过模型限制
4. 查看文档错误信息中的 `code/message/request_id`

### 问答没有命中课程资料

排查顺序：
1. 文档状态是否为 `completed`
2. 请求的 `knowledgeBaseIds` 是否包含目标知识库
3. 分数阈值是否设置过高
4. `data/chroma` 与 `data/rag.sqlite3` 是否属于同一数据目录

### Web 兜底未触发

- 确认 `TAVILY_API_KEY` 已配置
- 确认 `RAG_WEB_FALLBACK_ENABLED` 为 `true`
- 确认使用的是 `open` 模式
- 确认覆盖率低于 `RAG_COVERAGE_THRESHOLD`

## License

Internal project for CQUST AIStudy platform.
