from app.core.auth import Principal
from app.core.config import Settings
from app.schemas.rag import ChatRequest, RagOptions, RetrieveRequest
from app.services.database import Database
from app.services.rag_chain import RagChainService
from app.services.repository import RagRepository


class FakeVectorStore:
    def __init__(self, chunk_id: str, score: float):
        self.chunk_id = chunk_id
        self.score = score
        self.calls: list[dict] = []

    def similarity_search(self, query, knowledge_base_ids, top_k):
        return [{"chunkId": self.chunk_id, "score": self.score}]

    def hybrid_search(self, query, knowledge_base_ids, top_k):
        self.calls.append({"query": query, "knowledge_base_ids": knowledge_base_ids, "top_k": top_k})
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
    settings = Settings(data_dir=tmp_path, coverage_threshold=0.4, web_max_results=2, _env_file=None)
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
                "score": 0.9,
                "rerankScore": 0.9,
            }
        ],
    )

    system_prompt = str(messages[0].content)
    user_prompt = str(messages[1].content)
    assert "当前日期是 2026-06-05" in system_prompt
    assert "严禁编造" in system_prompt
    assert "强制引用" in system_prompt
    assert "冲突处理" in system_prompt
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
    assert prepared.used_web is False
    assert prepared.coverage_score == 0.9
    assert web_fallback.calls == []
    assert all(item.get("source") != "web" for item in prepared.sources)


async def test_open_mode_does_not_use_web_fallback_when_disabled(tmp_path):
    settings, repository = make_repository(tmp_path)
    settings.web_fallback_enabled = False
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

    assert prepared.used_web is False
    assert web_fallback.calls == []


def test_retrieve_keeps_low_rrf_like_relevance_score(tmp_path):
    settings, repository = make_repository(tmp_path)
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
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
        FakeVectorStore(chunk["id"], 1 / 61),
        FakeReranker(),
        FakeWebFallback(),
    )

    results = service.retrieve(
        RetrieveRequest(
            query="local",
            knowledgeBaseIds=[kb["id"]],
            topK=5,
            enableRerank=False,
            scoreThreshold=0,
        ),
        principal,
    )

    assert len(results) == 1
    assert results[0]["chunkId"] == chunk["id"]
    assert results[0]["score"] == 1 / 61


def test_retrieve_uses_settings_defaults_when_request_fields_are_omitted(tmp_path):
    settings, repository = make_repository(tmp_path)
    settings.default_top_k = 7
    settings.default_rerank_top_n = 4
    settings.default_score_threshold = 0.25
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
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
    vector_store = FakeVectorStore(chunk["id"], 0.2)
    service = RagChainService(settings, repository, vector_store, FakeReranker(), FakeWebFallback())

    results = service.retrieve(RetrieveRequest(query="local", knowledgeBaseIds=[kb["id"]]), principal)

    assert results == []
    assert vector_store.calls == [{"query": "local", "knowledge_base_ids": [kb["id"]], "top_k": 7}]


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


# ─── Post-Processing Tests ────────────────────────────────────────────────


def test_hallucination_detection_flags_suspicious_patterns(tmp_path):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )
    sources = [{"content": "红黑树是一种平衡二叉搜索树", "score": 0.9, "rerankScore": 0.9}]

    # 包含幻觉模式的回答
    answer_with_hallucination = "据我所知，红黑树是一种平衡二叉搜索树。通常来说插入复杂度是 O(log n)。"
    result = service._post_process(answer_with_hallucination, sources)
    assert len(result.hallucination_warnings) >= 2
    assert any("据我所知" in w for w in result.hallucination_warnings)

    # 正常引用的回答
    answer_normal = "红黑树是一种平衡二叉搜索树[1]。其插入时间复杂度为 O(log n)[1]。"
    result_normal = service._post_process(answer_normal, sources)
    assert len(result_normal.hallucination_warnings) == 0


def test_citation_validation_removes_invalid_citations(tmp_path):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )
    sources = [{"content": "资料1", "score": 0.9}, {"content": "资料2", "score": 0.8}]

    # 有效引用 [1][2] + 无效引用 [5]
    answer = "红黑树[1]是平衡树[5]，支持 O(log n) 插入[2]。"
    result = service._post_process(answer, sources)
    assert "[5]" not in result.answer
    assert "[1]" in result.answer
    assert "[2]" in result.answer


def test_citation_coverage_calculation(tmp_path):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )
    sources = [{"content": "数据结构", "score": 0.9}]

    # 全部引用的回答
    answer_full = "红黑树是平衡二叉搜索树[1]。插入复杂度 O(log n)[1]。"
    result = service._post_process(answer_full, sources)
    assert result.citation_coverage == 1.0

    # 部分引用的回答
    answer_partial = (
        "红黑树是平衡二叉搜索树[1]。"
        "删除操作也具有很好的时间复杂度保证。"
        "查找操作同样具有对数级别的性能表现。"
    )
    result_partial = service._post_process(answer_partial, sources)
    assert 0.0 < result_partial.citation_coverage < 1.0


def test_rejection_prompt_when_no_sources(tmp_path):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )

    # 空来源
    messages = service._build_messages("红黑树的时间复杂度是多少？", [])
    system_prompt = str(messages[0].content)
    assert "未检索到足够相关的资料" in system_prompt
    assert "严禁使用预训练知识" in system_prompt


def test_cot_injection_for_complex_queries(tmp_path):
    settings, repository = make_repository(tmp_path)
    service = RagChainService(
        settings,
        repository,
        FakeVectorStore("chunk", 0.9),
        FakeReranker(),
        FakeWebFallback(),
    )
    sources = [{"content": "红黑树 vs AVL 树", "score": 0.9, "rerankScore": 0.9}]

    # 复杂问题（比较类）应触发 CoT
    messages = service._build_messages("比较红黑树和 AVL 树的区别", sources)
    system_prompt = str(messages[0].content)
    assert "推理引导" in system_prompt
    assert "分步骤思考" in system_prompt

    # 简单问题不触发 CoT
    messages_simple = service._build_messages("红黑树的定义是什么", sources)
    system_prompt_simple = str(messages_simple[0].content)
    assert "推理引导" not in system_prompt_simple
