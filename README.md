# CQUST RAG Service

独立的 `uv + FastAPI + LangChain` RAG 服务，接口前缀为 `/rag`，不依赖旧 Java RAG 实现。

## 启动

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## 关键环境变量

- `DASHSCOPE_API_KEY`：阿里云 DashScope Key。
- `RAG_JWT_SECRET` / `RAG_JWT_ISSUER`：与 Java 后端 JWT 配置保持一致。
- `RAG_DATA_DIR`：SQLite、Chroma、上传文件存储目录，默认 `./data`。

未配置 `RAG_JWT_SECRET` 时仅本地开发放行；生产环境应显式配置。
