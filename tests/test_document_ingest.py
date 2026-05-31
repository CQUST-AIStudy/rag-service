from app.core.auth import Principal
from app.core.config import Settings
from app.schemas.rag import RetrieveRequest
from app.services.database import Database
from app.services.document_ingest import DocumentIngestService
from app.services.rag_chain import RagChainService
from app.services.repository import RagRepository


class FakeVectorStore:
    def __init__(self, fail_upsert: bool = False):
        self.fail_upsert = fail_upsert
        self.chunks: list[dict] = []
        self.deleted_document_ids: list[str] = []

    def upsert_chunks(self, chunks: list[dict]) -> None:
        if self.fail_upsert:
            raise RuntimeError("embedding failed")
        self.chunks = list(chunks)

    def delete_by_document(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)
        self.chunks = [item for item in self.chunks if item["documentId"] != document_id]

    def similarity_search(self, query, knowledge_base_ids, top_k):
        return [
            {
                "chunkId": item["id"],
                "documentId": item["documentId"],
                "knowledgeBaseId": item["knowledgeBaseId"],
                "score": 0.9,
                "metadata": item.get("metadata") or {},
            }
            for item in self.chunks
            if item["knowledgeBaseId"] in knowledge_base_ids
        ][:top_k]


class FakeReranker:
    def rerank(self, query, chunks, top_n):
        return chunks[:top_n]


class FakeWebFallback:
    async def search(self, query, max_results=None):
        return []


def make_repository(tmp_path):
    settings = Settings(data_dir=tmp_path)
    db = Database(settings)
    db.initialize()
    return settings, RagRepository(db)


def create_document(repository, tmp_path, chunk_size=128):
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": chunk_size,
            "chunkOverlap": 0,
        },
        embedding_dimensions=1024,
    )
    path = tmp_path / "guide.md"
    path.write_text("course material", encoding="utf-8")
    doc = repository.create_document(
        principal,
        kb["id"],
        "guide.md",
        str(path),
        "text/markdown",
        {},
    )
    return principal, kb, doc


def test_process_document_clears_chunks_when_vector_upsert_fails(monkeypatch, tmp_path):
    settings, repository = make_repository(tmp_path)
    principal, kb, doc = create_document(repository, tmp_path)
    vector_store = FakeVectorStore(fail_upsert=True)
    service = DocumentIngestService(repository, vector_store)
    monkeypatch.setattr(
        service,
        "_split_texts",
        lambda texts, kb: [{"content": f"chunk {index}", "tokenCount": 7, "metadata": {}} for index in range(3)],
    )

    service.process_document(doc["documentId"])

    updated = repository.require_document(doc["documentId"], principal)
    assert updated["status"] == "failed"
    assert updated["chunkCount"] == 0
    assert updated["tokenCount"] == 0
    assert "embedding failed" in updated["errorMessage"]
    assert repository.list_chunks(kb["id"], principal) == []
    assert vector_store.deleted_document_ids == [doc["documentId"], doc["documentId"]]


def test_process_document_completes_large_chunk_set_and_retrieve_hits(monkeypatch, tmp_path):
    settings, repository = make_repository(tmp_path)
    principal, kb, doc = create_document(repository, tmp_path)
    vector_store = FakeVectorStore()
    service = DocumentIngestService(repository, vector_store)
    monkeypatch.setattr(
        service,
        "_split_texts",
        lambda texts, kb: [
            {"content": f"binary search tree chunk {index}", "tokenCount": 24, "metadata": {"fileName": "guide.md"}}
            for index in range(55)
        ],
    )

    service.process_document(doc["documentId"])

    updated = repository.require_document(doc["documentId"], principal)
    chunks = repository.list_chunks(kb["id"], principal)
    assert updated["status"] == "completed"
    assert updated["chunkCount"] == 55
    assert updated["tokenCount"] == 55 * 24
    assert len(chunks) == 55
    assert len(vector_store.chunks) == 55

    rag_chain = RagChainService(settings, repository, vector_store, FakeReranker(), FakeWebFallback())
    results = rag_chain.retrieve(
        RetrieveRequest(query="binary search tree", knowledgeBaseIds=[kb["id"]], topK=5, enableRerank=False),
        principal,
    )

    assert len(results) == 5
    assert results[0]["documentId"] == doc["documentId"]
    assert "binary search tree" in results[0]["content"]
