from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.auth import Principal, current_principal
from app.core.config import Settings, get_settings
from app.core.responses import api_success
from app.schemas.rag import KnowledgeBaseCreate
from app.services.dependencies import get_repository, get_vector_store
from app.services.repository import RagRepository
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/rag/knowledge-base", tags=["knowledge-base"])


@router.post("")
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    principal: Annotated[Principal, Depends(current_principal)],
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    data = payload.model_dump()
    kb = repository.create_knowledge_base(principal, data, settings.embedding_dimensions)
    return api_success(
        {
            "id": kb["id"],
            "name": kb["name"],
            "documentCount": kb["documentCount"],
            "createdAt": kb["createdAt"],
            **kb,
        }
    )


@router.get("/list")
def list_knowledge_bases(
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    course_id: str | None = Query(default=None, alias="courseId"),
) -> dict:
    return api_success(repository.list_knowledge_bases(principal, course_id))


@router.put("/{knowledge_base_id}")
def update_knowledge_base(
    knowledge_base_id: str,
    payload: KnowledgeBaseCreate,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.update_knowledge_base(knowledge_base_id, principal, payload.model_dump()))


@router.delete("/{knowledge_base_id}")
def delete_knowledge_base(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> dict:
    repository.require_knowledge_base(knowledge_base_id, principal, write=True)
    vector_store.delete_by_knowledge_base(knowledge_base_id)
    repository.delete_knowledge_base(knowledge_base_id, principal)
    return api_success(None)
