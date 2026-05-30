import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.auth import Principal
from app.core.responses import ApiError
from app.services.database import Database, row_to_dict


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class RagRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_knowledge_base(
        self,
        principal: Principal,
        data: dict[str, Any],
        embedding_dimensions: int,
    ) -> dict[str, Any]:
        kb_id = make_id("kb")
        ts = now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_base (
                  id, owner_id, owner_role, name, description, course_id,
                  embedding_model, embedding_dimensions, chunk_size, chunk_overlap,
                  doc_visibility, class_ids_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb_id,
                    principal.user_id,
                    principal.role,
                    data["name"],
                    data.get("description") or "",
                    data.get("courseId") or data.get("course_id") or "",
                    data.get("embeddingModel") or data.get("embedding_model"),
                    embedding_dimensions,
                    int(data.get("chunkSize") or data.get("chunk_size") or 512),
                    int(data.get("chunkOverlap") or data.get("chunk_overlap") or 64),
                    data.get("docVisibility") or data.get("doc_visibility") or "public",
                    to_json(data.get("classIds") or data.get("class_ids") or []),
                    ts,
                    ts,
                ),
            )
        return self.require_knowledge_base(kb_id, principal)

    def list_knowledge_bases(
        self,
        principal: Principal,
        course_id: str | None = None,
        include_all_readable: bool = True,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = []
        if course_id:
            where.append("course_id = ?")
            params.append(course_id)
        if not principal.is_admin and include_all_readable:
            where.append("(owner_id = ? OR doc_visibility IN ('public','class'))")
            params.append(principal.user_id)
        elif not principal.is_admin:
            where.append("owner_id = ?")
            params.append(principal.user_id)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT kb.*,
                       (SELECT COUNT(*) FROM document d WHERE d.knowledge_base_id = kb.id) AS document_count
                FROM knowledge_base kb
                {where_sql}
                ORDER BY kb.created_at DESC
                """,
                params,
            ).fetchall()
        return [self._normalize_kb_row(dict(row)) for row in rows]

    def require_knowledge_base(self, kb_id: str, principal: Principal, write: bool = False) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT kb.*,
                       (SELECT COUNT(*) FROM document d WHERE d.knowledge_base_id = kb.id) AS document_count
                FROM knowledge_base kb WHERE kb.id = ?
                """,
                (kb_id,),
            ).fetchone()
        data = row_to_dict(row)
        if not data:
            raise ApiError(404, "知识库不存在")
        if not self._can_access_kb(data, principal, write):
            raise ApiError(403, "没有访问该知识库的权限")
        return self._normalize_kb_row(data)

    def delete_knowledge_base(self, kb_id: str, principal: Principal) -> None:
        self.require_knowledge_base(kb_id, principal, write=True)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM knowledge_base WHERE id = ?", (kb_id,))

    def update_knowledge_base(
        self,
        kb_id: str,
        principal: Principal,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        current = self.require_knowledge_base(kb_id, principal, write=True)
        ts = now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_base
                SET name = ?, description = ?, course_id = ?, chunk_size = ?, chunk_overlap = ?,
                    doc_visibility = ?, class_ids_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    data.get("name") or current["name"],
                    data.get("description") or "",
                    data.get("courseId") or data.get("course_id") or current.get("courseId") or "",
                    int(data.get("chunkSize") or data.get("chunk_size") or current["chunkSize"]),
                    int(data.get("chunkOverlap") or data.get("chunk_overlap") or current["chunkOverlap"]),
                    data.get("docVisibility") or data.get("doc_visibility") or current["docVisibility"],
                    to_json(data.get("classIds") or data.get("class_ids") or current["boundClassIds"]),
                    ts,
                    kb_id,
                ),
            )
        return self.require_knowledge_base(kb_id, principal)

    def create_document(
        self,
        principal: Principal,
        knowledge_base_id: str,
        file_name: str,
        stored_path: str,
        content_type: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.require_knowledge_base(knowledge_base_id, principal, write=True)
        doc_id = make_id("doc")
        ts = now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO document (
                  id, knowledge_base_id, owner_id, file_name, stored_path, content_type,
                  status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'processing', ?, ?)
                """,
                (
                    doc_id,
                    knowledge_base_id,
                    principal.user_id,
                    file_name,
                    stored_path,
                    content_type,
                    to_json(metadata),
                    ts,
                ),
            )
        return self.require_document(doc_id, principal)

    def require_document(self, document_id: str, principal: Principal | None = None) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM document WHERE id = ?", (document_id,)).fetchone()
        data = row_to_dict(row)
        if not data:
            raise ApiError(404, "文档不存在")
        if principal is not None:
            self.require_knowledge_base(data["knowledge_base_id"], principal)
        return self._normalize_document_row(data)

    def list_documents(self, knowledge_base_id: str, principal: Principal) -> list[dict[str, Any]]:
        self.require_knowledge_base(knowledge_base_id, principal)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document WHERE knowledge_base_id = ? ORDER BY created_at DESC",
                (knowledge_base_id,),
            ).fetchall()
        return [self._normalize_document_row(dict(row)) for row in rows]

    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        token_count: int = 0,
        error_message: str = "",
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE document
                SET status = ?, chunk_count = ?, token_count = ?, error_message = ?, processed_at = ?
                WHERE id = ?
                """,
                (status, chunk_count, token_count, error_message, now_iso(), document_id),
            )

    def queue_document_reprocess(self, document_id: str, principal: Principal) -> dict[str, Any]:
        doc = self.require_document(document_id, principal)
        self.require_knowledge_base(doc["knowledgeBaseId"], principal, write=True)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE document
                SET status = 'processing', error_message = '', processed_at = NULL
                WHERE id = ?
                """,
                (document_id,),
            )
        return self.require_document(document_id, principal)

    def queue_all_documents_reprocess(
        self,
        knowledge_base_id: str,
        principal: Principal,
    ) -> list[dict[str, Any]]:
        docs = self.list_documents(knowledge_base_id, principal)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE document
                SET status = 'processing', error_message = '', processed_at = NULL
                WHERE knowledge_base_id = ?
                """,
                (knowledge_base_id,),
            )
        return [self.require_document(item["documentId"], principal) for item in docs]

    def delete_document(self, document_id: str, principal: Principal) -> dict[str, Any]:
        doc = self.require_document(document_id, principal)
        self.require_knowledge_base(doc["knowledgeBaseId"], principal, write=True)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM document WHERE id = ?", (document_id,))
        return doc

    def replace_chunks(
        self,
        knowledge_base_id: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ts = now_iso()
        created: list[dict[str, Any]] = []
        with self.db.connect() as conn:
            conn.execute("DELETE FROM chunk WHERE document_id = ?", (document_id,))
            for index, item in enumerate(chunks):
                chunk_id = make_id("chunk")
                metadata = item.get("metadata") or {}
                conn.execute(
                    """
                    INSERT INTO chunk (
                      id, knowledge_base_id, document_id, chunk_index, content,
                      token_count, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        knowledge_base_id,
                        document_id,
                        index,
                        item["content"],
                        int(item.get("tokenCount") or len(item["content"])),
                        to_json(metadata),
                        ts,
                    ),
                )
                created.append(
                    {
                        "id": chunk_id,
                        "knowledgeBaseId": knowledge_base_id,
                        "documentId": document_id,
                        "content": item["content"],
                        "metadata": metadata,
                    }
                )
        return created

    def list_chunks(self, knowledge_base_id: str, principal: Principal) -> list[dict[str, Any]]:
        self.require_knowledge_base(knowledge_base_id, principal)
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, d.file_name
                FROM chunk c
                JOIN document d ON d.id = c.document_id
                WHERE c.knowledge_base_id = ?
                ORDER BY c.created_at DESC, c.chunk_index ASC
                """,
                (knowledge_base_id,),
            ).fetchall()
        return [self._normalize_chunk_row(dict(row)) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.file_name
                FROM chunk c
                JOIN document d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        chunks = [self._normalize_chunk_row(dict(row)) for row in rows]
        return sorted(chunks, key=lambda item: order.get(item["id"], 0))

    def create_or_touch_conversation(
        self,
        principal: Principal,
        conversation_id: str | None,
        knowledge_base_ids: list[str],
        query: str,
    ) -> str:
        ts = now_iso()
        conv_id = conversation_id or make_id("conv")
        with self.db.connect() as conn:
            existing = conn.execute("SELECT id FROM conversation WHERE id = ?", (conv_id,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE conversation SET updated_at = ?, knowledge_base_ids_json = ? WHERE id = ?",
                    (ts, to_json(knowledge_base_ids), conv_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO conversation (
                      id, owner_id, title, knowledge_base_ids_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (conv_id, principal.user_id, query[:80], to_json(knowledge_base_ids), ts, ts),
                )
        return conv_id

    def add_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_message (
                  id, conversation_id, role, content, sources_json, usage_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    make_id("msg"),
                    conversation_id,
                    role,
                    content,
                    to_json(sources or []),
                    to_json(usage or {}),
                    now_iso(),
                ),
            )

    def get_conversation_history(self, conversation_id: str, principal: Principal) -> dict[str, Any]:
        with self.db.connect() as conn:
            conv = conn.execute("SELECT * FROM conversation WHERE id = ?", (conversation_id,)).fetchone()
            if not conv:
                raise ApiError(404, "会话不存在")
            conv_data = dict(conv)
            if conv_data["owner_id"] != principal.user_id and not principal.is_admin:
                raise ApiError(403, "没有访问该会话的权限")
            rows = conn.execute(
                """
                SELECT * FROM conversation_message
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        return {
            "conversationId": conversation_id,
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "sources": from_json(row["sources_json"], []),
                    "usage": from_json(row["usage_json"], {}),
                    "createdAt": row["created_at"],
                }
                for row in rows
            ],
        }

    def delete_conversation(self, conversation_id: str, principal: Principal) -> None:
        self.get_conversation_history(conversation_id, principal)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM conversation WHERE id = ?", (conversation_id,))

    def save_qa_log(
        self,
        principal: Principal,
        conversation_id: str,
        knowledge_base_ids: list[str],
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        used_web: bool = False,
        intent_type: str = "rag",
    ) -> str:
        top1_score = float(sources[0].get("rerankScore") or sources[0].get("score") or 0) if sources else 0
        log_id = make_id("qa")
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_log (
                  id, owner_id, conversation_id, knowledge_base_ids_json, query, answer_text,
                  sources_json, top1_score, coverage_score, used_web, intent_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    principal.user_id,
                    conversation_id,
                    to_json(knowledge_base_ids),
                    query,
                    answer,
                    to_json(sources),
                    top1_score,
                    top1_score,
                    1 if used_web else 0,
                    intent_type,
                    now_iso(),
                ),
            )
        return log_id

    def set_feedback(self, qa_log_id: str, feedback: int, principal: Principal) -> None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT owner_id FROM qa_log WHERE id = ?", (qa_log_id,)).fetchone()
            if not row:
                raise ApiError(404, "问答日志不存在")
            if row["owner_id"] != principal.user_id and not principal.is_admin:
                raise ApiError(403, "没有反馈该问答的权限")
            conn.execute("UPDATE qa_log SET feedback = ? WHERE id = ?", (feedback, qa_log_id))

    def create_annotation(
        self,
        knowledge_base_id: str,
        chunk_id: str,
        annotation_type: str,
        note: str,
        principal: Principal,
    ) -> dict[str, Any]:
        self.require_knowledge_base(knowledge_base_id, principal, write=True)
        annotation_id = make_id("anno")
        ts = now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO annotation (
                  id, knowledge_base_id, chunk_id, annotation_type, note, owner_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (annotation_id, knowledge_base_id, chunk_id, annotation_type, note, principal.user_id, ts),
            )
        return {
            "id": annotation_id,
            "chunkId": chunk_id,
            "annotationType": annotation_type,
            "note": note,
            "teacherId": principal.user_id,
            "createdAt": ts,
        }

    def list_annotations(self, knowledge_base_id: str, principal: Principal) -> list[dict[str, Any]]:
        self.require_knowledge_base(knowledge_base_id, principal)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotation WHERE knowledge_base_id = ? ORDER BY created_at DESC",
                (knowledge_base_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "chunkId": row["chunk_id"],
                "annotationType": row["annotation_type"],
                "note": row["note"],
                "teacherId": row["owner_id"],
                "createdAt": row["created_at"],
            }
            for row in rows
        ]

    def delete_annotation(self, annotation_id: str, principal: Principal) -> None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT owner_id FROM annotation WHERE id = ?", (annotation_id,)).fetchone()
            if not row:
                raise ApiError(404, "标注不存在")
            if row["owner_id"] != principal.user_id and not principal.is_admin:
                raise ApiError(403, "没有删除该标注的权限")
            conn.execute("DELETE FROM annotation WHERE id = ?", (annotation_id,))

    def analytics(self, knowledge_base_id: str, principal: Principal) -> dict[str, Any]:
        self.require_knowledge_base(knowledge_base_id, principal)
        with self.db.connect() as conn:
            logs = conn.execute(
                """
                SELECT * FROM qa_log
                WHERE knowledge_base_ids_json LIKE ?
                ORDER BY created_at DESC
                """,
                (f"%{knowledge_base_id}%",),
            ).fetchall()
        items = [dict(row) for row in logs]
        return {
            "logs": items,
            "hotQuestions": self._hot_questions(items),
            "hitRate": self._hit_rate(items, 0.4),
            "citationCoverage": self._citation_coverage(items),
            "webTriggerRate": self._web_trigger_rate(items),
            "feedbackStats": self._feedback_stats(items),
            "resourceGaps": self._resource_gaps(items),
        }

    def _normalize_kb_row(self, data: dict[str, Any]) -> dict[str, Any]:
        document_count = int(data.get("document_count") or 0)
        return {
            "id": data["id"],
            "name": data["name"],
            "description": data.get("description") or "",
            "courseId": data.get("course_id") or "",
            "courseName": data.get("course_id") or "",
            "term": "",
            "embeddingModel": data.get("embedding_model") or "",
            "embeddingDimensions": data.get("embedding_dimensions") or 1024,
            "chunkSize": data.get("chunk_size") or 512,
            "chunkOverlap": data.get("chunk_overlap") or 64,
            "docVisibility": data.get("doc_visibility") or "public",
            "boundClassIds": from_json(data.get("class_ids_json"), []),
            "documentCount": document_count,
            "docCount": document_count,
            "defaultMode": "strict",
            "allowWebSearch": False,
            "requireCitation": True,
            "createdAt": data.get("created_at"),
            "updatedAt": data.get("updated_at"),
        }

    def _normalize_document_row(self, data: dict[str, Any]) -> dict[str, Any]:
        status = data["status"]
        return {
            "documentId": data["id"],
            "id": data["id"],
            "knowledgeBaseId": data["knowledge_base_id"],
            "courseSpaceId": data["knowledge_base_id"],
            "fileName": data["file_name"],
            "status": status,
            "legacyStatus": self._legacy_status(status),
            "chunkCount": int(data.get("chunk_count") or 0),
            "tokenCount": int(data.get("token_count") or 0),
            "metadata": from_json(data.get("metadata_json"), {}),
            "errorMessage": data.get("error_message") or "",
            "createdAt": data.get("created_at"),
            "processedAt": data.get("processed_at"),
        }

    def _normalize_chunk_row(self, data: dict[str, Any]) -> dict[str, Any]:
        content = data.get("content") or ""
        metadata = from_json(data.get("metadata_json"), {})
        return {
            "id": data["id"],
            "chunkId": data["id"],
            "documentId": data["document_id"],
            "knowledgeBaseId": data["knowledge_base_id"],
            "fileName": data.get("file_name") or metadata.get("fileName") or "",
            "content": content,
            "contentPreview": content[:200] + ("..." if len(content) > 200 else ""),
            "metadata": metadata,
            "chapterPath": metadata.get("chapter") or metadata.get("chapterPath") or "",
            "pageRange": str(metadata.get("page") or metadata.get("pageRange") or ""),
        }

    def _legacy_status(self, status: str) -> str:
        mapping = {"completed": "READY", "processing": "PROCESSING", "failed": "FAILED"}
        return mapping.get(status, status.upper())

    def _can_access_kb(self, data: dict[str, Any], principal: Principal, write: bool) -> bool:
        if principal.is_admin:
            return True
        if data["owner_id"] == principal.user_id:
            return True
        if write:
            return False
        return (data.get("doc_visibility") or "public") in {"public", "class"}

    def _hot_questions(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for item in logs:
            counts[item["query"]] = counts.get(item["query"], 0) + 1
        return [
            {"question": question, "count": count}
            for question, count in sorted(counts.items(), key=lambda entry: entry[1], reverse=True)[:20]
        ]

    def _hit_rate(self, logs: list[dict[str, Any]], threshold: float) -> float:
        if not logs:
            return 0
        hits = sum(1 for item in logs if float(item.get("top1_score") or 0) >= threshold)
        return hits / len(logs)

    def _citation_coverage(self, logs: list[dict[str, Any]]) -> dict[str, int]:
        coverage: dict[str, int] = {}
        for item in logs:
            for source in from_json(item.get("sources_json"), []):
                name = source.get("fileName") or source.get("documentId") or "引用资料"
                coverage[name] = coverage.get(name, 0) + 1
        return coverage

    def _web_trigger_rate(self, logs: list[dict[str, Any]]) -> float:
        if not logs:
            return 0
        used = sum(1 for item in logs if int(item.get("used_web") or 0) == 1)
        return used / len(logs)

    def _feedback_stats(self, logs: list[dict[str, Any]]) -> dict[str, int]:
        thumbs_up = sum(1 for item in logs if item.get("feedback") == 1)
        thumbs_down = sum(1 for item in logs if item.get("feedback") == -1)
        return {"thumbsUp": thumbs_up, "thumbsDown": thumbs_down, "total": thumbs_up + thumbs_down}

    def _resource_gaps(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gaps = [
            {
                "query": item["query"],
                "frequency": 1,
                "avgCoverage": float(item.get("coverage_score") or 0),
            }
            for item in logs
            if float(item.get("coverage_score") or 0) < 0.4
        ]
        return gaps[:20]
