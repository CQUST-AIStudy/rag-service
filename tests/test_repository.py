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
            "name": "数据结构",
            "description": "课程资料",
            "courseId": "course_001",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
        },
        embedding_dimensions=1024,
    )

    assert created["name"] == "数据结构"
    assert created["documentCount"] == 0
    assert repository.list_knowledge_bases(principal)[0]["id"] == created["id"]

    updated = repository.update_knowledge_base(
        created["id"],
        principal,
        {
            "name": "数据结构实验",
            "description": "",
            "courseId": "course_001",
            "chunkSize": 256,
            "chunkOverlap": 32,
            "docVisibility": "public",
        },
    )

    assert updated["name"] == "数据结构实验"
    assert updated["chunkSize"] == 256

    repository.delete_knowledge_base(created["id"], principal)
    assert repository.list_knowledge_bases(principal) == []


def test_document_and_chunk_lifecycle(tmp_path):
    repository = make_repository(tmp_path)
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "算法",
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
        [{"content": "链表反转是 O(n)", "metadata": {"chapter": "1"}}],
    )
    repository.update_document_status(doc["documentId"], "completed", chunk_count=1, token_count=10)

    assert len(chunks) == 1
    assert repository.require_document(doc["documentId"], principal)["status"] == "completed"
    assert repository.list_chunks(kb["id"], principal)[0]["content"] == "链表反转是 O(n)"
