# AGENTS.md

## 语言

- 所有说明、计划、总结、测试结果默认使用中文。
- 代码标识符、库名、接口字段名、命令参数可保留英文。

## 项目概况

- 本项目是独立的 FastAPI RAG 服务，运行入口为 `app.main:app`。
- 依赖管理使用 `uv`，依赖锁定文件为 `uv.lock`。
- 服务默认监听 `8001`，健康检查接口为 `/health`，业务接口前缀为 `/rag`。
- 运行时数据默认放在 `data/`，包括 SQLite、Chroma 和上传文件；该目录不应提交。

## 工作原则

- 修改前先阅读相关代码、配置、类型定义与调用链。
- 优先最小必要改动，复用现有实现和项目约定。
- 不修改与当前任务无关的文件，不做无必要重构。
- 不硬编码 API Key、Token、JWT Secret、模型路径或私有配置。

## Python / FastAPI

- Python 代码必须使用类型标注。
- FastAPI 路由保持轻量，业务逻辑放在 Service 层或等价业务层。
- 请求和响应优先使用 Pydantic。
- 错误必须显式处理，不吞错、不静默失败。

## Docker 部署

- Docker 镜像内服务运行目录为 `/app`。
- 容器内运行时数据目录固定为 `/app/data`，通过 Docker volume 持久化。
- Docker 部署环境使用 `.env.docker`，仓库只提交 `.env.docker.example`。
- 生产部署必须配置 `RAG_JWT_SECRET` 和 `DASHSCOPE_API_KEY`。

## 验证

- 完成修改后尽可能执行最相关验证。
- Python 常用验证命令：

```powershell
uv run pytest
uv run ruff check .
```

- Docker 配置常用验证命令：

```powershell
docker compose config
```

- 未验证不得声称已修复；无法验证时必须说明原因。
