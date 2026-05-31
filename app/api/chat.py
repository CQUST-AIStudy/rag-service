from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.auth import Principal, current_principal
from app.core.responses import api_success
from app.schemas.rag import ChatRequest, LegacyChatRequest, RetrieveRequest
from app.services.dependencies import get_rag_chain_service, get_repository
from app.services.rag_chain import RagChainService
from app.services.repository import RagRepository

router = APIRouter(prefix="/rag", tags=["rag-chat"])


@router.post("/retrieve")
def retrieve(
    request: RetrieveRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    rag_chain: Annotated[RagChainService, Depends(get_rag_chain_service)],
) -> dict:
    return api_success({"results": rag_chain.retrieve(request, principal)})


@router.post("/chat")
async def chat(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    rag_chain: Annotated[RagChainService, Depends(get_rag_chain_service)],
) -> dict:
    return api_success(await rag_chain.chat(request, principal))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    rag_chain: Annotated[RagChainService, Depends(get_rag_chain_service)],
) -> StreamingResponse:
    return StreamingResponse(
        rag_chain.stream_chat(request, principal),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/legacy-stream")
async def legacy_chat_stream(
    request: LegacyChatRequest,
    principal: Annotated[Principal, Depends(current_principal)],
    rag_chain: Annotated[RagChainService, Depends(get_rag_chain_service)],
) -> StreamingResponse:
    kb_id = request.knowledgeBaseId or request.courseSpaceId
    normalized = ChatRequest(
        query=request.query,
        knowledgeBaseIds=[str(kb_id)] if kb_id else [],
        mode=request.mode,
        options=request.options or {},
    )
    return StreamingResponse(
        rag_chain.stream_chat(normalized, principal),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversation/{conversation_id}/history")
def conversation_history(
    conversation_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    return api_success(repository.get_conversation_history(conversation_id, principal))


@router.delete("/conversation/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    principal: Annotated[Principal, Depends(current_principal)],
    repository: Annotated[RagRepository, Depends(get_repository)],
) -> dict:
    repository.delete_conversation(conversation_id, principal)
    return api_success(None)
