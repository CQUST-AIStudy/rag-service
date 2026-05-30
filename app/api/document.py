import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.core.auth import Principal, current_principal
from app.core.config import Settings, get_settings
from app.core.responses import ApiError, api_success
from app.services.dependencies import get_document_ingest_service, get_repository, get_vector_store
from app.services.document_ingest import DocumentIngestService, parse_metadata
from app.services.repository import RagRepository, make_id
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/rag/document", tags=["document"])
compat_router = APIRouter(prefix="/rag/knowledge-base", tags=["knowledge-base-compat"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(current_principal)],
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    ingest_service: Annotated[DocumentIngestService, Depends(get_document_ingest_service)],
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(..., alias="knowledgeBaseId"),
    metadata: str | None = Form(default=None),
) -> dict:
    ingest_service.validate_extension(file.filename or "")
    meta = parse_metadata(metadata)
    upload_id = make_id("upload")
    stored_path = settings.upload_dir / knowledge_base_id / f"{upload_id}_{Path(file.filename or 'file').name}"
    stored_path.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with stored_path.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                target.close()
                stored_path.unlink(missing_ok=True)
                raise ApiError(413, f"文件过大，最大支持 {settings.max_upload_mb}MB")
            target.write(chunk)

    document = repository.create_document(
        principal,
        knowledge_base_id,
        file.filename or stored_path.name,
        str(stored_path),
        file.content_type or "",
        meta,
    )
    background_tasks.add_task(ingest_service.process_document, document["documentId"])
    return api_success(
        {
            "documentId": document["documentId"],
            "fileName": document["fileName"],
            "status": document["status"],
            "chunkCount": document["chunkCount"],
        }
    )


@router.get("/{document_id}/status")
def document_status(
    document_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    document = repository.require_document(document_id, principal)
    return api_success(document)


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> dict:
    stored_path = _stored_path(repository, document_id)
    document = repository.delete_document(document_id, principal)
    vector_store.delete_by_document(document["documentId"])
    if stored_path:
        Path(stored_path).unlink(missing_ok=True)
    return api_success(None)


@compat_router.get("/{knowledge_base_id}/documents")
def list_documents(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.list_documents(knowledge_base_id, principal))


@compat_router.get("/{knowledge_base_id}/documents/status-summary")
def document_status_summary(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    documents = repository.list_documents(knowledge_base_id, principal)
    counts = {
        "total": len(documents),
        "ready": sum(1 for item in documents if item["status"] == "completed"),
        "processing": sum(1 for item in documents if item["status"] == "processing"),
        "pending": 0,
        "failed": sum(1 for item in documents if item["status"] == "failed"),
        "totalChunks": sum(int(item["chunkCount"] or 0) for item in documents),
    }
    return api_success({"knowledgeBaseId": knowledge_base_id, "counts": counts, "documents": documents})


@compat_router.post("/{knowledge_base_id}/documents/reprocess")
def reprocess_all_documents(
    knowledge_base_id: str,
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    ingest_service: Annotated[DocumentIngestService, Depends(get_document_ingest_service)],
) -> dict:
    documents = repository.queue_all_documents_reprocess(knowledge_base_id, principal)
    for document in documents:
        background_tasks.add_task(ingest_service.process_document, document["documentId"])
    return api_success(
        {
            "knowledgeBaseId": knowledge_base_id,
            "requestedCount": len(documents),
            "queuedDocumentIds": [item["documentId"] for item in documents],
        }
    )


@compat_router.post("/{knowledge_base_id}/documents/{document_id}/reprocess")
def reprocess_document(
    knowledge_base_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    ingest_service: Annotated[DocumentIngestService, Depends(get_document_ingest_service)],
) -> dict:
    document = repository.queue_document_reprocess(document_id, principal)
    background_tasks.add_task(ingest_service.process_document, document["documentId"])
    return api_success(
        {
            "knowledgeBaseId": knowledge_base_id,
            "documentId": document["documentId"],
            "queued": True,
            "status": document["status"],
        }
    )


@compat_router.post("/{knowledge_base_id}/rebuild-bm25")
def rebuild_index(
    knowledge_base_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    repository.require_knowledge_base(knowledge_base_id, principal, write=True)
    return api_success({"knowledgeBaseId": knowledge_base_id, "rebuilt": True})


@compat_router.post("/{knowledge_base_id}/documents")
async def upload_document_compat(
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(current_principal)],
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[RagRepository, Depends(get_repository)],
    ingest_service: Annotated[DocumentIngestService, Depends(get_document_ingest_service)],
    file: UploadFile = File(...),
    doc_type: str = Form(default="textbook", alias="docType"),
    knowledge_base_id: str = "",
) -> dict:
    metadata = {"docType": doc_type}
    ingest_service.validate_extension(file.filename or "")
    upload_id = make_id("upload")
    stored_path = settings.upload_dir / knowledge_base_id / f"{upload_id}_{Path(file.filename or 'file').name}"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    with stored_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    document = repository.create_document(
        principal,
        knowledge_base_id,
        file.filename or stored_path.name,
        str(stored_path),
        file.content_type or "",
        metadata,
    )
    background_tasks.add_task(ingest_service.process_document, document["documentId"])
    return api_success(document)


def _stored_path(repository: RagRepository, document_id: str) -> str | None:
    with repository.db.connect() as conn:
        row = conn.execute("SELECT stored_path FROM document WHERE id = ?", (document_id,)).fetchone()
    return row["stored_path"] if row else None
