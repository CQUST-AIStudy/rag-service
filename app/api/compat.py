from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.auth import Principal, current_principal
from app.core.responses import api_success
from app.schemas.rag import AnnotationCreate, FeedbackRequest
from app.services.dependencies import get_repository
from app.services.repository import RagRepository

router = APIRouter(prefix="/rag", tags=["rag-compat"])


@router.get("/knowledge-base/{knowledge_base_id}/chunks")
def list_chunks(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.list_chunks(knowledge_base_id, principal))


@router.post("/knowledge-base/{knowledge_base_id}/annotations")
def create_annotation(
    knowledge_base_id: str,
    payload: AnnotationCreate,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    annotation = repository.create_annotation(
        knowledge_base_id,
        payload.chunkId,
        payload.annotationType,
        payload.note,
        principal,
    )
    return api_success(annotation)


@router.get("/knowledge-base/{knowledge_base_id}/annotations")
def list_annotations(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.list_annotations(knowledge_base_id, principal))


@router.delete("/annotations/{annotation_id}")
def delete_annotation(
    annotation_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    repository.delete_annotation(annotation_id, principal)
    return api_success(None)


@router.post("/feedback")
def submit_feedback(
    payload: FeedbackRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    repository.set_feedback(payload.qaLogId, payload.feedback, principal)
    return api_success({"success": True})


@router.get("/knowledge-base/{knowledge_base_id}/analytics")
def analytics(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    coverage_threshold: float = Query(default=0.4, ge=0, le=1, alias="coverageThreshold"),
    min_frequency: int = Query(default=3, ge=1, alias="minFrequency"),
) -> dict:
    return api_success(repository.analytics(knowledge_base_id, principal, coverage_threshold, min_frequency))


@router.get("/knowledge-base/{knowledge_base_id}/analytics/hot-questions")
def hot_questions(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    top: int = Query(default=20, ge=1, le=100),
) -> dict:
    data = repository.analytics(knowledge_base_id, principal)["hotQuestions"][:top]
    return api_success(data)


@router.get("/knowledge-base/{knowledge_base_id}/analytics/hit-rate")
def hit_rate(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    threshold: float = Query(default=0.4),
) -> dict:
    logs = repository.analytics(knowledge_base_id, principal, threshold)["logs"]
    if not logs:
        return api_success({"hitRate": 0})
    hits = sum(1 for item in logs if float(item.get("coverage_score") or 0) >= threshold)
    return api_success({"hitRate": hits / len(logs)})


@router.get("/knowledge-base/{knowledge_base_id}/analytics/citation-coverage")
def citation_coverage(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.analytics(knowledge_base_id, principal)["citationCoverage"])


@router.get("/knowledge-base/{knowledge_base_id}/analytics/web-trigger-rate")
def web_trigger_rate(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success({"webTriggerRate": repository.analytics(knowledge_base_id, principal)["webTriggerRate"]})


@router.get("/knowledge-base/{knowledge_base_id}/analytics/feedback-stats")
def feedback_stats(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.analytics(knowledge_base_id, principal)["feedbackStats"])


@router.get("/knowledge-base/{knowledge_base_id}/analytics/resource-gaps")
def resource_gaps(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    coverage_threshold: float = Query(default=0.4, ge=0, le=1, alias="coverageThreshold"),
    min_frequency: int = Query(default=3, ge=1, alias="minFrequency"),
) -> dict:
    return api_success(repository.resource_gaps(knowledge_base_id, principal, coverage_threshold, min_frequency))
