from typing import Any

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    courseId: str = ""
    courseName: str = ""
    term: str = ""
    embeddingModel: str = "text-embedding-v4"
    chunkSize: int = Field(1500, ge=128, le=3000)
    chunkOverlap: int = Field(300, ge=0, le=1024)
    docVisibility: str = "public"
    classIds: list[int | str] = Field(default_factory=list)
    defaultMode: str = "strict"
    allowWebSearch: bool = False
    requireCitation: bool = True


class RagOptions(BaseModel):
    topK: int = Field(30, ge=1, le=100)
    rerankTopN: int = Field(5, ge=1, le=20)
    scoreThreshold: float = Field(0.0, ge=0.0, le=1.0)
    enableRerank: bool = True
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    maxTokens: int = Field(2048, ge=64, le=8192)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    knowledgeBaseIds: list[str] = Field(default_factory=list)
    conversationId: str | None = None
    mode: str | None = None
    options: RagOptions = Field(default_factory=RagOptions)


class AssistantHistoryMessage(BaseModel):
    role: str
    content: str = Field("", max_length=8000)


class AssistantRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    mode: str = Field("ai", pattern="^(ai|rag|web)$")
    enableWebSearch: bool = False
    knowledgeBaseIds: list[str] = Field(default_factory=list)
    conversationId: str | None = None
    history: list[AssistantHistoryMessage] = Field(default_factory=list)
    options: RagOptions = Field(default_factory=RagOptions)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    knowledgeBaseIds: list[str] = Field(default_factory=list)
    topK: int = Field(30, ge=1, le=100)
    enableRerank: bool = True
    rerankTopN: int = Field(5, ge=1, le=20)
    scoreThreshold: float = Field(0.0, ge=0.0, le=1.0)


class AnnotationCreate(BaseModel):
    chunkId: str
    annotationType: str
    note: str = ""


class FeedbackRequest(BaseModel):
    qaLogId: str
    feedback: int = Field(..., ge=-1, le=1)


class LegacyChatRequest(BaseModel):
    courseSpaceId: str | int | None = None
    knowledgeBaseId: str | int | None = None
    query: str
    mode: str = "strict"
    classId: str | int | None = None
    className: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
