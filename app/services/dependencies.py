from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.dashscope import DashScopeReranker
from app.services.database import Database
from app.services.document_ingest import DocumentIngestService
from app.services.rag_chain import RagChainService
from app.services.repository import RagRepository
from app.services.vector_store import VectorStore
from app.services.web_fallback import TavilyWebFallbackService


@lru_cache
def get_database() -> Database:
    return Database(get_settings())


@lru_cache
def get_repository() -> RagRepository:
    return RagRepository(get_database())


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())


@lru_cache
def get_reranker() -> DashScopeReranker:
    return DashScopeReranker(get_settings())


@lru_cache
def get_web_fallback_service() -> TavilyWebFallbackService:
    return TavilyWebFallbackService(get_settings())


@lru_cache
def get_document_ingest_service() -> DocumentIngestService:
    return DocumentIngestService(get_repository(), get_vector_store())


@lru_cache
def get_rag_chain_service() -> RagChainService:
    settings: Settings = get_settings()
    return RagChainService(settings, get_repository(), get_vector_store(), get_reranker(), get_web_fallback_service())
