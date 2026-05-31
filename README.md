# CQUST RAG Service

`rag-service` 是 CQUST AIStudy 的独立 RAG 服务，基于 `uv + FastAPI + LangChain` 构建，负责课程知识库、文档入库、向量检索、重排、问答、会话历史、切片标注和 RAG 分析接口。

服务所有业务接口统一挂载在 `/rag` 前缀下，默认监听 `8001` 端口，可通过前端工程的 `/rag` 代理访问。

## 功能概览

- 知识库管理：创建、更新、删除和按课程查询知识库。
- 文档入库：支持 `PDF`、`DOCX`、`TXT`、`Markdown`，上传后异步抽取文本、切片并写入向量库。
- RAG 检索：使用 DashScope embedding + Chroma 做向量召回，支持可选重排。
- 课程问答：支持非流式和 SSE 流式回答，返回引用来源。
- 标注与分析：支持切片标注、问答反馈、命中率、引用覆盖、资源缺口等分析。
- 兼容接口：保留前端和旧调用方使用的课程空间、旧流式问答等兼容路径。

## 技术栈

- `FastAPI`：HTTP API、依赖注入、CORS 和异常处理。
- `LangChain`：模型调用、聊天链路和文档处理能力。
- `Chroma`：本地向量存储，默认写入 `./data/chroma`。
- `SQLite`：知识库、文档、切片、会话、反馈和分析日志存储。
- `DashScope`：embedding、rerank 和 chat 模型服务。
- `Tavily`：开放模式下的可选 Web 兜底搜索。

## 快速启动

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

启动后可访问：

- 健康检查：`GET http://127.0.0.1:8001/health`
- Swagger 文档：`http://127.0.0.1:8001/docs`

如需从前端联调，请确认前端代理将 `/rag` 转发到 `http://127.0.0.1:8001`。

## 环境变量

服务会读取项目根目录下的 `.env`，也可以直接使用系统环境变量。

| 变量 | 说明 | 默认值 |
|---|---|---|
| `RAG_SERVICE_NAME` | FastAPI 服务名称 | `CQUST RAG Service` |
| `RAG_ENV` | 运行环境，生产环境建议设为 `production` | `local` |
| `RAG_HOST` | 服务监听地址 | `0.0.0.0` |
| `RAG_PORT` | 服务端口 | `8001` |
| `RAG_DATA_DIR` | SQLite、Chroma、上传文件根目录 | `./data` |
| `RAG_ALLOWED_ORIGINS` | CORS 白名单，多个地址用英文逗号分隔 | `http://localhost:8080,http://127.0.0.1:8080` |
| `RAG_JWT_SECRET` / `JWT_SECRET` | JWT 密钥，与 `backend-go` 保持一致 | 空 |
| `RAG_JWT_ISSUER` / `JWT_ISSUER` | JWT 签发方，与 `backend-go` 保持一致 | `tap` |
| `DASHSCOPE_API_KEY` | DashScope API Key，用于 embedding、rerank 和 chat | 空 |
| `DASHSCOPE_COMPAT_BASE_URL` | DashScope OpenAI 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_RERANK_URL` | DashScope 重排接口地址 | 见 `app/core/config.py` |
| `RAG_EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `RAG_EMBEDDING_DIMENSIONS` | 嵌入维度 | `1024` |
| `RAG_RERANK_MODEL` | 重排模型 | `qwen3-vl-rerank` |
| `RAG_CHAT_MODEL` | 对话模型 | `qwen-plus` |
| `RAG_DEFAULT_TOP_K` | 默认召回条数 | `10` |
| `RAG_DEFAULT_RERANK_TOP_N` | 默认重排保留条数 | `3` |
| `RAG_DEFAULT_SCORE_THRESHOLD` | 默认分数阈值 | `0.0` |
| `RAG_COVERAGE_THRESHOLD` | 开放模式触发 Web 兜底的覆盖率阈值 | `0.4` |
| `RAG_MAX_UPLOAD_MB` | 单文件最大上传大小 | `50` |
| `TAVILY_API_KEY` | Tavily API Key，未配置时 Web 兜底不可用 | 空 |
| `RAG_WEB_FALLBACK_ENABLED` | 是否启用 Web 兜底 | `true` |
| `RAG_WEB_MAX_RESULTS` | Web 兜底最大结果数 | `5` |

## 鉴权说明

- 正常请求应携带 `Authorization: Bearer <token>`。
- 本地开发环境下，如果没有配置 `RAG_JWT_SECRET`，服务会使用开发身份放行。
- 生产环境必须配置 `RAG_JWT_SECRET`，否则相关接口会拒绝访问。
- `RAG_JWT_SECRET/RAG_JWT_ISSUER` 也支持读取后端通用变量 `JWT_SECRET/JWT_ISSUER`。

## 文档入库流程

1. 创建知识库。
2. 上传文档到知识库。
3. 后台抽取文本并按知识库配置切片。
4. 写入 SQLite 切片表。
5. 调用 DashScope embedding 并写入 Chroma。
6. 全部成功后文档状态才会变为 `completed`。
7. 问答接口从 Chroma 召回，再回表补全文档和切片信息。

状态语义：

- `processing`：文档已上传，正在抽取、切片或写入向量。
- `completed`：SQLite 分块和 Chroma 向量都已写入，可用于问答检索。
- `failed`：入库失败；服务会清理该文档残留分块和向量，避免“分块可见但问答不可用”的半成功状态。

说明：`text-embedding-v4` 单次 embedding 请求有批量限制，服务会自动按 10 条一批请求 DashScope，并保持返回向量顺序。

## 响应格式

成功响应：

```json
{ "code": 200, "message": "success", "data": {} }
```

错误响应：

```json
{ "code": 400, "message": "错误信息", "data": null }
```

流式问答接口使用 `text/event-stream`，主要事件包括：

- `retrieval`：返回检索来源、模式和覆盖率。
- `delta`：返回模型增量文本。
- `done`：返回会话 ID、用量占位信息和最终状态。
- `error`：返回错误信息。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/rag/knowledge-base` | 创建知识库 |
| `GET` | `/rag/knowledge-base/list` | 查询知识库列表 |
| `PUT` | `/rag/knowledge-base/{knowledge_base_id}` | 更新知识库 |
| `DELETE` | `/rag/knowledge-base/{knowledge_base_id}` | 删除知识库 |
| `POST` | `/rag/document/upload` | 标准文档上传接口 |
| `GET` | `/rag/document/{document_id}/status` | 查询文档处理状态 |
| `DELETE` | `/rag/document/{document_id}` | 删除文档 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/documents` | 查询知识库文档 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/documents/status-summary` | 查询文档状态汇总 |
| `POST` | `/rag/knowledge-base/{knowledge_base_id}/documents/reprocess` | 重处理知识库全部文档 |
| `POST` | `/rag/knowledge-base/{knowledge_base_id}/documents/{document_id}/reprocess` | 重处理单个文档 |
| `POST` | `/rag/knowledge-base/{knowledge_base_id}/documents` | 兼容文档上传接口 |
| `POST` | `/rag/knowledge-base/{knowledge_base_id}/rebuild-bm25` | 兼容 BM25 重建接口 |
| `POST` | `/rag/retrieve` | RAG 检索 |
| `POST` | `/rag/chat` | 非流式问答 |
| `POST` | `/rag/chat/stream` | SSE 流式问答 |
| `POST` | `/rag/chat/legacy-stream` | 兼容旧请求格式的 SSE 流式问答 |
| `GET` | `/rag/conversation/{conversation_id}/history` | 查询会话历史 |
| `DELETE` | `/rag/conversation/{conversation_id}` | 删除会话 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/chunks` | 查询知识库切片 |
| `POST` | `/rag/knowledge-base/{knowledge_base_id}/annotations` | 创建切片标注 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/annotations` | 查询切片标注 |
| `DELETE` | `/rag/annotations/{annotation_id}` | 删除切片标注 |
| `POST` | `/rag/feedback` | 提交问答反馈 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics` | 查询知识库分析总览 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/hot-questions` | 查询热门问题 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/hit-rate` | 查询命中率 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/citation-coverage` | 查询引用覆盖 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/web-trigger-rate` | 查询 Web 触发率 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/feedback-stats` | 查询反馈统计 |
| `GET` | `/rag/knowledge-base/{knowledge_base_id}/analytics/resource-gaps` | 查询资源缺口 |

## 开发与验证

常用命令：

```bash
uv run pytest
uv run ruff check .
```

建议修改后至少运行：

- 文档入库、检索或问答逻辑：`uv run pytest`
- 格式和静态检查：`uv run ruff check .`
- 本地联调：启动服务后在前端课程知识库页面上传文档、等待 `completed`，再使用“知识问答”提问。

## 常见问题

### 上传显示失败，但分块里有内容

这通常说明旧版本服务在 SQLite 已写入切片后，向量写入失败。当前版本会在失败时清理该文档的切片和向量，并把错误原因写入文档状态。历史失败文档需要点击“重处理”或“全部重处理”重新入库。

### DashScope 返回 HTTP 400

请优先检查：

- `DASHSCOPE_API_KEY` 是否配置正确。
- embedding 模型和维度是否匹配。
- 输入文本是否过长。
- DashScope 返回体中的 `code/message/request_id`，服务会尽量写入文档错误信息。

### 问答没有命中课程资料

请确认：

- 文档状态是 `completed`，不是 `processing` 或 `failed`。
- 当前提问的 `knowledgeBaseIds` 是否包含目标知识库。
- 分数阈值是否过高。
- 本地 `data/chroma` 是否与 `data/rag.sqlite3` 属于同一份数据目录。
