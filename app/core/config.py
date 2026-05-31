from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = Field("CQUST RAG Service", alias="RAG_SERVICE_NAME")
    env: str = Field("local", alias="RAG_ENV")
    host: str = Field("0.0.0.0", alias="RAG_HOST")
    port: int = Field(8001, alias="RAG_PORT")
    data_dir: Path = Field(Path("./data"), alias="RAG_DATA_DIR")
    allowed_origins: str = Field(
        "http://localhost:8080,http://127.0.0.1:8080",
        alias="RAG_ALLOWED_ORIGINS",
    )

    jwt_secret: str = Field(
        "",
        validation_alias=AliasChoices("RAG_JWT_SECRET", "JWT_SECRET"),
    )
    jwt_issuer: str = Field(
        "tap",
        validation_alias=AliasChoices("RAG_JWT_ISSUER", "JWT_ISSUER"),
    )

    dashscope_api_key: str = Field("", alias="DASHSCOPE_API_KEY")
    dashscope_compat_base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_COMPAT_BASE_URL",
    )
    dashscope_rerank_url: str = Field(
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        alias="DASHSCOPE_RERANK_URL",
    )
    embedding_model: str = Field("text-embedding-v4", alias="RAG_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1024, alias="RAG_EMBEDDING_DIMENSIONS")
    rerank_model: str = Field("qwen3-vl-rerank", alias="RAG_RERANK_MODEL")
    chat_model: str = Field("qwen-plus", alias="RAG_CHAT_MODEL")

    default_top_k: int = Field(10, alias="RAG_DEFAULT_TOP_K")
    default_rerank_top_n: int = Field(3, alias="RAG_DEFAULT_RERANK_TOP_N")
    default_score_threshold: float = Field(0.0, alias="RAG_DEFAULT_SCORE_THRESHOLD")
    coverage_threshold: float = Field(0.4, alias="RAG_COVERAGE_THRESHOLD")
    max_upload_mb: int = Field(50, alias="RAG_MAX_UPLOAD_MB")

    tavily_api_key: str = Field("", alias="TAVILY_API_KEY")
    tavily_search_url: str = Field("https://api.tavily.com/search", alias="TAVILY_SEARCH_URL")
    web_fallback_enabled: bool = Field(True, alias="RAG_WEB_FALLBACK_ENABLED")
    web_max_results: int = Field(5, alias="RAG_WEB_MAX_RESULTS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "rag.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
