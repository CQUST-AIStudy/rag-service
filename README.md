# CQUST RAG Service

`rag-service` 是一个独立的 `uv + FastAPI + LangChain` RAG 服务，提供知识库管理、文档入库、向量检索、重排、聊天问答、会话历史和分析接口。它不依赖旧的 Java RAG 实现，所有接口统一挂在 `/rag` 前缀下。

## 技术栈

- `FastAPI`：HTTP API 与依赖注入
- `LangChain`：RAG 检索与大模型调用
- `Chroma`：向量存储
- `SQLite`：知识库、文档、切片、会话和日志存储
- `DashScope`：嵌入模型、重排模型和对话模型

## 快速开始

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

服务启动后默认访问：

- 健康检查：`GET /health`
- API 文档：`http://127.0.0.1:8001/docs`

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DASHSCOPE_API_KEY` | DashScope 访问密钥，用于嵌入、重排和生成 | 空 |
| `DASHSCOPE_COMPAT_BASE_URL` | OpenAI 兼容模式基础地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_RERANK_URL` | 重排接口地址 | `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank` |
| `RAG_SERVICE_NAME` | 服务名称 | `CQUST RAG Service` |
| `RAG_ENV` | 运行环境 | `local` |
| `RAG_HOST` | 服务监听地址 | `0.0.0.0` |
| `RAG_PORT` | 服务端口 | `8001` |
| `RAG_DATA_DIR` | SQLite、Chroma、上传文件根目录 | `./data` |
| `RAG_ALLOWED_ORIGINS` | CORS 白名单 | `http://localhost:8080,http://127.0.0.1:8080` |
| `RAG_JWT_SECRET` | JWT 密钥，与 Java 后端保持一致 | 空 |
| `RAG_JWT_ISSUER` | JWT 签发方 | `tap-backend` |
| `RAG_EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `RAG_EMBEDDING_DIMENSIONS` | 嵌入维度 | `1024` |
| `RAG_RERANK_MODEL` | 重排模型 | `qwen3-vl-rerank` |
| `RAG_CHAT_MODEL` | 对话模型 | `qwen-plus` |
| `RAG_DEFAULT_TOP_K` | 默认检索条数 | `10` |
| `RAG_DEFAULT_RERANK_TOP_N` | 默认重排保留条数 | `3` |
| `RAG_DEFAULT_SCORE_THRESHOLD` | 默认分数阈值 | `0.0` |
| `RAG_MAX_UPLOAD_MB` | 单文件最大上传大小 | `50` |

## 鉴权说明

- 请求通过 `Authorization: Bearer <token>` 传递 JWT。
- 当 `RAG_JWT_SECRET` 未配置且当前环境不是生产环境时，服务会使用本地开发身份放行。
- 当 `RAG_ENV=production` 时，必须配置 `RAG_JWT_SECRET`，否则相关接口会拒绝访问。

## RAG 使用流程

1. 创建知识库。
2. 上传文档到知识库。
3. 等待后台处理完成，状态变为 `completed`。
4. 使用 `/rag/retrieve` 或 `/rag/chat` 发起检索与问答。
5. 通过会话历史、反馈和分析接口查看效果。

## 接口约定

成功响应：

```json
{ "code": 200, "message": "success", "data": {} }
```

错误响应：

```json
{ "code": 400, "message": "错误信息", "data": null }
```

流式问答接口使用 `text/event-stream`。

## API 表格

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/rag/knowledge-base` | 创建知识库 |
| GET | `/rag/knowledge-base/list` | 查询知识库列表 |
| PUT | `/rag/knowledge-base/{knowledge_base_id}` | 更新知识库 |
| DELETE | `/rag/knowledge-base/{knowledge_base_id}` | 删除知识库 |
| POST | `/rag/document/upload` | 上传文档 |
| GET | `/rag/document/{document_id}/status` | 查询文档处理状态 |
| DELETE | `/rag/document/{document_id}` | 删除文档 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/documents` | 查询知识库文档列表 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/documents/status-summary` | 查询文档状态汇总 |
| POST | `/rag/knowledge-base/{knowledge_base_id}/documents/reprocess` | 重新处理知识库全部文档 |
| POST | `/rag/knowledge-base/{knowledge_base_id}/documents/{document_id}/reprocess` | 重新处理单个文档 |
| POST | `/rag/knowledge-base/{knowledge_base_id}/rebuild-bm25` | 兼容接口：重建 BM25 索引标记 |
| POST | `/rag/knowledge-base/{knowledge_base_id}/documents` | 兼容接口：上传文档 |
| POST | `/rag/retrieve` | RAG 检索 |
| POST | `/rag/chat` | RAG 非流式问答 |
| POST | `/rag/chat/stream` | RAG SSE 流式问答 |
| POST | `/rag/chat/legacy-stream` | 兼容旧请求格式的 SSE 流式问答 |
| GET | `/rag/conversation/{conversation_id}/history` | 查询会话历史 |
| DELETE | `/rag/conversation/{conversation_id}` | 删除会话 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/chunks` | 查询知识库切片 |
| POST | `/rag/knowledge-base/{knowledge_base_id}/annotations` | 创建切片标注 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/annotations` | 查询切片标注 |
| DELETE | `/rag/annotations/{annotation_id}` | 删除切片标注 |
| POST | `/rag/feedback` | 提交问答反馈 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics` | 查询知识库分析总览 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/hot-questions` | 查询热门问题 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/hit-rate` | 查询命中率 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/citation-coverage` | 查询引用覆盖 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/web-trigger-rate` | 查询 Web 触发率 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/feedback-stats` | 查询反馈统计 |
| GET | `/rag/knowledge-base/{knowledge_base_id}/analytics/resource-gaps` | 查询资源缺口 |

## 本地验证

已在当前仓库中完成以下验证：

- `uv run pytest -q`：`5 passed`
- `uv run ruff check .`：通过
- 离线 RAG 冒烟：通过，覆盖入库、检索、聊天、历史和分析日志
- 真实 DashScope 冒烟：通过，覆盖真实嵌入、重排和生成链路

说明：真实冒烟脚本结束时，Windows 临时目录清理 Chroma 文件可能会遇到文件锁，但不影响 RAG 功能验证结果。
