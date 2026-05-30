from typing import Any

from langchain_chroma import Chroma

from app.core.config import Settings
from app.services.dashscope import DashScopeEmbeddings


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = DashScopeEmbeddings(settings)
        self.store = Chroma(
            collection_name="rag_chunks",
            persist_directory=str(settings.chroma_dir),
            embedding_function=self.embeddings,
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        self.store.add_texts(
            texts=[item["content"] for item in chunks],
            metadatas=[
                {
                    "chunkId": item["id"],
                    "knowledgeBaseId": item["knowledgeBaseId"],
                    "documentId": item["documentId"],
                    **(item.get("metadata") or {}),
                }
                for item in chunks
            ],
            ids=[item["id"] for item in chunks],
        )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.store.delete(ids=chunk_ids)

    def delete_by_document(self, document_id: str) -> None:
        collection = self.store._collection
        existing = collection.get(where={"documentId": document_id})
        ids = existing.get("ids") or []
        if ids:
            self.store.delete(ids=ids)

    def similarity_search(
        self,
        query: str,
        knowledge_base_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not knowledge_base_ids:
            return []
        where = (
            {"knowledgeBaseId": knowledge_base_ids[0]}
            if len(knowledge_base_ids) == 1
            else {"knowledgeBaseId": {"$in": knowledge_base_ids}}
        )
        results = self.store.similarity_search_with_score(query, k=top_k, filter=where)
        hits: list[dict[str, Any]] = []
        for doc, distance in results:
            metadata = dict(doc.metadata or {})
            score = self._distance_to_score(distance)
            hits.append(
                {
                    "chunkId": metadata.get("chunkId"),
                    "documentId": metadata.get("documentId"),
                    "knowledgeBaseId": metadata.get("knowledgeBaseId"),
                    "content": doc.page_content,
                    "score": score,
                    "metadata": metadata,
                }
            )
        return [item for item in hits if item.get("chunkId")]

    def _distance_to_score(self, distance: float) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0
        return max(0.0, min(1.0, 1.0 / (1.0 + value)))
