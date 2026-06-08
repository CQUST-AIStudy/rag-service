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
            error_message = str(exc)
            try:
                self._cleanup_document_artifacts(doc["documentId"])
            except Exception as cleanup_exc:
                error_message = f"{error_message}; 清理失败: {cleanup_exc}"
            self.repository.update_document_status(
                doc["documentId"],
                "failed",
                chunk_count=0,
                token_count=0,
                error_message=error_message[:1000],
            )

    def _cleanup_document_artifacts(self, document_id: str) -> None:
        try:
            self.vector_store.delete_by_document(document_id)
        except Exception:
            pass
        self.repository.delete_chunks_by_document(document_id)

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
        """PDF 全文提取后统一切分，避免跨页断裂"""
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            pages_text: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                # 提取表格并转为 Markdown 格式保留结构
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        md_table = self._table_to_markdown(table)
                        text += "\n\n" + md_table
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        return [{"content": full_text, "metadata": {"source": path.name}}]

    def _table_to_markdown(self, table: list[list[str | None]]) -> str:
        """将 pdfplumber 提取的表格转为 Markdown 表格"""
        if not table or not table[0]:
            return ""
        rows: list[str] = []
        for i, row in enumerate(table):
            cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)

    def _load_docx(self, path: Path) -> list[dict[str, Any]]:
        import docx2txt

        return [{"content": docx2txt.process(str(path)) or "", "metadata": {"source": path.name}}]

    def _split_texts(self, texts: list[dict[str, Any]], kb: dict[str, Any]) -> list[dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(kb.get("chunkSize") or 1500),
            chunk_overlap=int(kb.get("chunkOverlap") or 300),
            separators=[
                "\n## ", "\n### ", "\n#### ",   # Markdown 标题优先
                "\n\n",                          # 段落分隔
                "。\n", "！\n", "？\n",           # 中文句末+换行
                "\n",                            # 普通换行
                "。", "！", "？",                # 中文句末
                "；", "：", "，",                # 中文标点补充
                ".", "!", "?",                   # 英文句末
                " ", "",                         # 兜底
            ],
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
