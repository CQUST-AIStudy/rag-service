from app.core.auth import Principal
from app.core.config import Settings
from app.schemas.rag import ChatRequest, RagOptions
from app.services.database import Database
from app.services.rag_chain import RagChainService
from app.services.repository import RagRepository


class FakeVectorStore:
    def __init__(self, chunk_id: str, score: float):
        self.chunk_id = chunk_id
        self.score = score

    def similarity_search(self, query, knowledge_base_ids, top_k):
        return [{"chunkId": self.chunk_id, "score": self.score}]


class FakeReranker:
    def rerank(self, query, chunks, top_n):
        return chunks[:top_n]


class FakeWebFallback:
    async def search(self, query, max_results=None):
        return [
            {
                "documentId": "web:https://example.test",
                "chunkId": "web:1",
                "fileName": "Example Web Result",
                "content": "Fresh external context.",
                "chunkContent": "Fresh external context.",
                "score": 0.8,
                "rerankScore": 0.8,
                "source": "web",
                "metadata": {
                    "url": "https://example.test",
                    "title": "Example Web Result",
                    "source": "web",
                },
            }
        ]


def make_repository(tmp_path):
    settings = Settings(data_dir=tmp_path, coverage_threshold=0.4, web_max_results=2)
    db = Database(settings)
    db.initialize()
    return settings, RagRepository(db)


async def test_open_mode_uses_web_fallback_when_coverage_is_low(tmp_path):
    settings, repository = make_repository(tmp_path)
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
            "defaultMode": "open",
            "allowWebSearch": True,
        },
        embedding_dimensions=1024,
    )
    doc = repository.create_document(
        principal,
        kb["id"],
        "guide.md",
        str(tmp_path / "guide.md"),
        "text/markdown",
        {},
    )
    chunk = repository.replace_chunks(
        kb["id"],
        doc["documentId"],
        [{"content": "Local context", "metadata": {}}],
    )[0]

    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.1),
        FakeReranker(),
        FakeWebFallback(),
    )
    request = ChatRequest(
        query="What is missing?",
        knowledgeBaseIds=[kb["id"]],
        options=RagOptions(enableRerank=False),
    )

    prepared = await service._prepare_chat(request, principal)

    assert prepared.effective_mode == "open"
    assert prepared.used_web is True
    assert prepared.coverage_score == 0.1
    assert [item["source"] for item in prepared.sources if item.get("source") == "web"] == ["web"]


async def test_strict_mode_does_not_use_web_fallback(tmp_path):
    settings, repository = make_repository(tmp_path)
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
            "defaultMode": "open",
            "allowWebSearch": True,
        },
        embedding_dimensions=1024,
    )
    doc = repository.create_document(
        principal,
        kb["id"],
        "guide.md",
        str(tmp_path / "guide.md"),
        "text/markdown",
        {},
    )
    chunk = repository.replace_chunks(
        kb["id"],
        doc["documentId"],
        [{"content": "Local context", "metadata": {}}],
    )[0]
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.1),
        FakeReranker(),
        FakeWebFallback(),
    )
    request = ChatRequest(
        query="What is missing?",
        knowledgeBaseIds=[kb["id"]],
        mode="strict",
        options=RagOptions(enableRerank=False),
    )

    prepared = await service._prepare_chat(request, principal)

    assert prepared.effective_mode == "strict"
    assert prepared.used_web is False
    assert all(item.get("source") != "web" for item in prepared.sources)
