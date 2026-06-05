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
    def __init__(self, results: list[dict] | None = None):
        self.calls: list[dict] = []
        self.results = results if results is not None else [
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

    async def search(self, query, max_results=None):
        self.calls.append({"query": query, "max_results": max_results})
        return [dict(item) for item in self.results]


def make_repository(tmp_path):
    settings = Settings(data_dir=tmp_path, coverage_threshold=0.4, web_max_results=2)
    db = Database(settings)
    db.initialize()
    return settings, RagRepository(db)


def test_build_messages_include_current_date_for_time_sensitive_questions(tmp_path, monkeypatch):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )
    monkeypatch.setattr(
        service,
        "_current_date_context",
        lambda: "当前日期是 2026-06-05（北京时间，Asia/Shanghai）。",
    )

    messages = service._build_messages(
        "现在最新的 Go 版本是多少？",
        [
            {
                "documentId": "web:https://go.dev/dl",
                "fileName": "Go Downloads",
                "content": "Go 1.26.3 is available.",
                "source": "web",
            }
        ],
    )

    system_prompt = str(messages[0].content)
    user_prompt = str(messages[1].content)
    assert "当前日期是 2026-06-05" in system_prompt
    assert "时间敏感信息" in system_prompt
    assert "不要把课程资料或网页快照中的旧年份当作当前年份" in system_prompt
    assert "优先采用更新、更权威的 Web 或官方来源" in system_prompt
    assert "Go 1.26.3 is available." in user_prompt


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

    web_fallback = FakeWebFallback()
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.1),
        FakeReranker(),
        web_fallback,
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
    assert web_fallback.calls == [{"query": "What is missing?", "max_results": 2}]
    assert [item["source"] for item in prepared.sources if item.get("source") == "web"] == ["web"]


async def test_open_mode_uses_web_fallback_even_when_coverage_is_high(tmp_path):
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
        [{"content": "Outdated local context", "metadata": {}}],
    )[0]
    web_fallback = FakeWebFallback()
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.9),
        FakeReranker(),
        web_fallback,
    )
    request = ChatRequest(
        query="What is the latest version?",
        knowledgeBaseIds=[kb["id"]],
        options=RagOptions(enableRerank=False),
    )

    prepared = await service._prepare_chat(request, principal)

    assert prepared.effective_mode == "open"
    assert prepared.used_web is True
    assert prepared.coverage_score == 0.9
    assert web_fallback.calls == [{"query": "What is the latest version?", "max_results": 2}]
    assert any(item.get("source") == "web" for item in prepared.sources)


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
    web_fallback = FakeWebFallback()
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.1),
        FakeReranker(),
        web_fallback,
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
    assert web_fallback.calls == []
    assert all(item.get("source") != "web" for item in prepared.sources)


async def test_open_mode_does_not_use_web_fallback_when_web_search_is_not_allowed(tmp_path):
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
            "allowWebSearch": False,
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
    web_fallback = FakeWebFallback()
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore(chunk["id"], 0.1),
        FakeReranker(),
        web_fallback,
    )
    request = ChatRequest(
        query="What is missing?",
        knowledgeBaseIds=[kb["id"]],
        options=RagOptions(enableRerank=False),
    )

    prepared = await service._prepare_chat(request, principal)

    assert prepared.effective_mode == "open"
    assert prepared.used_web is False
    assert web_fallback.calls == []
    assert all(item.get("source") != "web" for item in prepared.sources)
