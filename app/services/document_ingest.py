import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.auth import Principal
from app.core.responses import ApiError
from app.services.repository import RagRepository
from app.services.vector_store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentIngestService:
    def __init__(self, repository: RagRepository, vector_store: VectorStore):
        self.repository = repository
        self.vector_store = vector_store

    def validate_extension(self, file_name: str) -> str:
        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ApiError(422, "不支持的文件格式")
        return suffix

    def process_document(self, document_id: str) -> None:
        doc = self.repository.require_document(document_id)
        kb = self.repository.require_knowledge_base(
            doc["knowledgeBaseId"],
            # The document already exists, so an internal admin principal is enough for processing.
            principal=Principal(user_id="internal", username="internal", role="ADMIN"),
        )
        try:
            path = Path(self._raw_document_path(doc))
            texts = self._load_text(path)
            chunks = self._split_texts(texts, kb)
            created_chunks = self.repository.replace_chunks(
                doc["knowledgeBaseId"],
                doc["documentId"],
                chunks,
            )
            self.vector_store.delete_by_document(doc["documentId"])
            self.vector_store.upsert_chunks(created_chunks)
            token_count = sum(item.get("tokenCount") or len(item["content"]) for item in chunks)
            self.repository.update_document_status(
                doc["documentId"],
                "completed",
                chunk_count=len(chunks),
                token_count=token_count,
            )
        except Exception as exc:
            self.repository.update_document_status(
                doc["documentId"],
                "failed",
                error_message=str(exc)[:1000],
            )

    def _raw_document_path(self, doc: dict[str, Any]) -> str:
        # stored_path is intentionally not exposed in the public document response.
        with self.repository.db.connect() as conn:
            row = conn.execute("SELECT stored_path FROM document WHERE id = ?", (doc["documentId"],)).fetchone()
        if not row:
            raise ApiError(404, "文档不存在")
        return row["stored_path"]

    def _load_text(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [{"content": self._read_text(path), "metadata": {"source": path.name}}]
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        raise ApiError(422, "不支持的文件格式")

    def _read_text(self, path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    def _load_pdf(self, path: Path) -> list[dict[str, Any]]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages):
            pages.append(
                {
                    "content": page.extract_text() or "",
                    "metadata": {"source": path.name, "page": index + 1},
                }
            )
        return pages

    def _load_docx(self, path: Path) -> list[dict[str, Any]]:
        import docx2txt

        return [{"content": docx2txt.process(str(path)) or "", "metadata": {"source": path.name}}]

    def _split_texts(self, texts: list[dict[str, Any]], kb: dict[str, Any]) -> list[dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(kb.get("chunkSize") or 512),
            chunk_overlap=int(kb.get("chunkOverlap") or 64),
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )
        chunks: list[dict[str, Any]] = []
        for item in texts:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            for chunk in splitter.split_text(content):
                metadata = dict(item.get("metadata") or {})
                metadata["fileName"] = metadata.get("source") or ""
                chunks.append(
                    {
                        "content": chunk,
                        "tokenCount": len(chunk),
                        "metadata": metadata,
                    }
                )
        if not chunks:
            raise ApiError(422, "文档没有可入库的文本内容")
        return chunks


def parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(400, "metadata 必须是合法 JSON 字符串") from exc
    if not isinstance(value, dict):
        raise ApiError(400, "metadata 必须是 JSON 对象")
    return value
