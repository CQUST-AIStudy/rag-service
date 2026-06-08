from app.api.knowledge_base import delete_knowledge_base
from app.core.auth import Principal
from app.core.config import Settings
from app.services.database import Database
from app.services.repository import RagRepository


class FakeVectorStore:
    def __init__(self):
        self.deleted_knowledge_base_ids: list[str] = []

    def delete_by_knowledge_base(self, knowledge_base_id: str) -> None:
        self.deleted_knowledge_base_ids.append(knowledge_base_id)


def make_repository(tmp_path) -> RagRepository:
    settings = Settings(data_dir=tmp_path, _env_file=None)
    db = Database(settings)
    db.initialize()
    return RagRepository(db)


def test_delete_knowledge_base_cleans_vectors_and_database(tmp_path):
    repository = make_repository(tmp_path)
    vector_store = FakeVectorStore()
    principal = Principal(user_id="1", username="teacher", role="TEACHER")
    kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Data Structures",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
        },
        embedding_dimensions=1024,
    )

    response = delete_knowledge_base(kb["id"], principal, repository, vector_store)

    assert response == {"code": 200, "message": "success", "data": None}
    assert vector_store.deleted_knowledge_base_ids == [kb["id"]]
    assert repository.list_knowledge_bases(principal) == []
