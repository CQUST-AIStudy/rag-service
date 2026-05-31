from app.core.auth import Principal
from app.core.config import Settings
from app.services.database import Database
from app.services.repository import RagRepository


def make_repository(tmp_path):
    settings = Settings(data_dir=tmp_path)
    db = Database(settings)
    db.initialize()
    return RagRepository(db)


def test_knowledge_base_crud(tmp_path):
    repository = make_repository(tmp_path)
    principal = Principal(user_id="1", username="teacher", role="TEACHER")

    created = repository.create_knowledge_base(
        principal,
        {
            "name": "Data Structures",
            "description": "Course materials",
            "courseId": "course_001",
            "courseName": "Data Structures",
            "term": "2026 Spring",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
            "defaultMode": "open",
            "allowWebSearch": True,
            "requireCitation": False,
        },
        embedding_dimensions=1024,
    )

    assert created["name"] == "Data Structures"
    assert created["courseName"] == "Data Structures"
    assert created["term"] == "2026 Spring"
    assert created["defaultMode"] == "open"
    assert created["allowWebSearch"] is True
    assert created["requireCitation"] is False
    assert created["documentCount"] == 0
    assert repository.list_knowledge_bases(principal)[0]["id"] == created["id"]

    updated = repository.update_knowledge_base(
        created["id"],
        principal,
        {
            "name": "Data Structures Lab",
            "description": "",
            "courseId": "course_001",
            "courseName": "Data Structures Lab",
            "term": "2026 Fall",
            "chunkSize": 256,
            "chunkOverlap": 32,
            "docVisibility": "public",
            "defaultMode": "strict",
            "allowWebSearch": False,
            "requireCitation": True,
        },
    )

    assert updated["name"] == "Data Structures Lab"
    assert updated["chunkSize"] == 256
    assert updated["courseName"] == "Data Structures Lab"
    assert updated["term"] == "2026 Fall"
    assert updated["defaultMode"] == "strict"
    assert updated["allowWebSearch"] is False
    assert updated["requireCitation"] is True

    repository.delete_knowledge_base(created["id"], principal)
    assert repository.list_knowledge_bases(principal) == []


def test_document_and_chunk_lifecycle(tmp_path):
    repository = make_repository(tmp_path)
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
        {"chapter": "1"},
    )
    chunks = repository.replace_chunks(
        kb["id"],
        doc["documentId"],
        [{"content": "Linked-list reversal is O(n)", "metadata": {"chapter": "1"}}],
    )
    repository.update_document_status(doc["documentId"], "completed", chunk_count=1, token_count=10)

    assert len(chunks) == 1
    assert repository.require_document(doc["documentId"], principal)["status"] == "completed"
    assert repository.list_chunks(kb["id"], principal)[0]["content"] == "Linked-list reversal is O(n)"


def test_resource_gaps_respect_filters(tmp_path):
    repository = make_repository(tmp_path)
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

    for _ in range(3):
        repository.save_qa_log(
            principal,
            "conv",
            [kb["id"]],
            "missing topic",
            "answer",
            [],
            coverage_score=0.2,
            mode="open",
        )
    repository.save_qa_log(
        principal,
        "conv",
        [kb["id"]],
        "rare topic",
        "answer",
        [],
        coverage_score=0.1,
    )
    repository.save_qa_log(
        principal,
        "conv",
        [kb["id"]],
        "covered topic",
        "answer",
        [],
        coverage_score=0.9,
    )

    gaps = repository.resource_gaps(kb["id"], principal, coverage_threshold=0.4, min_frequency=2)

    assert gaps == [
        {
            "query": "missing topic",
            "count": 3,
            "frequency": 3,
            "avgCoverage": 0.20000000000000004,
        }
    ]
    analytics = repository.analytics(kb["id"], principal, coverage_threshold=0.4, min_frequency=2)
    assert analytics["hotQuestions"][0]["query"] == "missing topic"
    assert analytics["resourceGaps"] == gaps
