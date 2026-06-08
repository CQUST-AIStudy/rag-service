from io import BytesIO

from fastapi import BackgroundTasks, UploadFile

from app.api.document import reprocess_document, upload_document, upload_document_compat
from app.core.auth import Principal
from app.core.config import Settings
from app.core.responses import ApiError
from app.services.database import Database
from app.services.repository import RagRepository


class FakeIngestService:
    def validate_extension(self, filename: str) -> None:
        if not filename.endswith(".md"):
            raise ApiError(400, "unsupported")

    def process_document(self, document_id: str) -> None:
        return None


class DenyRepository:
    def require_knowledge_base(self, knowledge_base_id, principal, write=False):
        raise ApiError(403, "没有访问该知识库的权限")


def make_upload_file(name: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=name)


def make_repository(settings: Settings) -> RagRepository:
    db = Database(settings)
    db.initialize()
    return RagRepository(db)


async def test_upload_checks_write_permission_before_writing_file(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    principal = Principal(user_id="student", username="student", role="STUDENT")

    try:
        await upload_document(
            BackgroundTasks(),
            principal,
            settings,
            DenyRepository(),
            FakeIngestService(),
            make_upload_file("guide.md", b"course material"),
            "kb-denied",
            None,
        )
    except ApiError as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("upload should fail before writing without write permission")

    assert not (settings.upload_dir / "kb-denied").exists()


async def test_compat_upload_enforces_max_upload_size_and_cleans_file(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=0, _env_file=None)
    repository = make_repository(settings)
    principal = Principal(user_id="teacher", username="teacher", role="TEACHER")
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

    try:
        await upload_document_compat(
            BackgroundTasks(),
            principal,
            settings,
            repository,
            FakeIngestService(),
            make_upload_file("guide.md", b"x"),
            "textbook",
            kb["id"],
        )
    except ApiError as exc:
        assert exc.status_code == 413
    else:
        raise AssertionError("oversized compat upload should be rejected")

    upload_dir = settings.upload_dir / kb["id"]
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []


def test_reprocess_document_rejects_mismatched_knowledge_base(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    repository = make_repository(settings)
    principal = Principal(user_id="teacher", username="teacher", role="TEACHER")
    first_kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Data Structures",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
        },
        embedding_dimensions=1024,
    )
    second_kb = repository.create_knowledge_base(
        principal,
        {
            "name": "Algorithms",
            "embeddingModel": "text-embedding-v4",
            "chunkSize": 512,
            "chunkOverlap": 64,
        },
        embedding_dimensions=1024,
    )
    document = repository.create_document(
        principal,
        second_kb["id"],
        "guide.md",
        str(tmp_path / "guide.md"),
        "text/markdown",
        {},
    )
    repository.update_document_status(document["documentId"], "completed", chunk_count=1, token_count=10)

    try:
        reprocess_document(
            first_kb["id"],
            document["documentId"],
            BackgroundTasks(),
            principal,
            repository,
            FakeIngestService(),
        )
    except ApiError as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("mismatched knowledge base path should be rejected")

    updated = repository.require_document(document["documentId"], principal)
    assert updated["status"] == "completed"
