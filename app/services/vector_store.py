from typing import Any

import jieba
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

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
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index: BM25Okapi | None = None
        self._bm25_cache_key: tuple[str, ...] | None = None
        self._bm25_corpus_ids: list[str] = []
        self._bm25_corpus_map: dict[str, dict[str, Any]] = {}

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
        self._clear_bm25_cache()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.store.delete(ids=chunk_ids)
            self._clear_bm25_cache()

    def delete_by_document(self, document_id: str) -> None:
        collection = self.store._collection
        existing = collection.get(where={"documentId": document_id})
        ids = existing.get("ids") or []
        if ids:
            self.store.delete(ids=ids)
            self._clear_bm25_cache()

    def delete_by_knowledge_base(self, knowledge_base_id: str) -> None:
        collection = self.store._collection
        existing = collection.get(where={"knowledgeBaseId": knowledge_base_id})
        ids = existing.get("ids") or []
        if ids:
            self.store.delete(ids=ids)
            self._clear_bm25_cache()

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

    def hybrid_search(
        self,
        query: str,
        knowledge_base_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25，使用 RRF 融合"""
        # 向量检索
        vector_hits = self.similarity_search(query, knowledge_base_ids, top_k * 2)
        # BM25 检索
        bm25_hits = self._bm25_search(query, knowledge_base_ids, top_k * 2)
        # RRF 融合
        return self._rrf_merge(vector_hits, bm25_hits, top_k)

    def _bm25_search(
        self,
        query: str,
        knowledge_base_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """基于 BM25 的关键词检索"""
        self._ensure_bm25_index(knowledge_base_ids)
        if not self._bm25_index or not self._bm25_corpus_ids:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self._bm25_index.get_scores(query_tokens)

        # 取 top_k 个最高分
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            chunk_id = self._bm25_corpus_ids[idx]
            chunk_data = self._bm25_corpus_map.get(chunk_id, {})
            hits.append(
                {
                    "chunkId": chunk_id,
                    "documentId": chunk_data.get("documentId"),
                    "knowledgeBaseId": chunk_data.get("knowledgeBaseId"),
                    "content": chunk_data.get("content", ""),
                    "score": float(score),
                    "metadata": chunk_data.get("metadata", {}),
                }
            )
        return hits

    def _ensure_bm25_index(self, knowledge_base_ids: list[str]) -> None:
        """懒加载 BM25 索引"""
        cache_key = tuple(sorted(str(item) for item in knowledge_base_ids))
        if self._bm25_index is not None and self._bm25_cache_key == cache_key:
            return

        self._clear_bm25_cache()
        # 从 Chroma 获取所有相关 chunk
        collection = self.store._collection
        where = (
            {"knowledgeBaseId": knowledge_base_ids[0]}
            if len(knowledge_base_ids) == 1
            else {"knowledgeBaseId": {"$in": knowledge_base_ids}}
        )
        try:
            result = collection.get(where=where, include=["documents", "metadatas"])
        except Exception:
            self._clear_bm25_cache()
            return

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            self._clear_bm25_cache()
            return

        corpus: list[list[str]] = []
        self._bm25_corpus_ids = []
        self._bm25_corpus_map = {}

        for chunk_id, doc, meta in zip(ids, documents, metadatas, strict=False):
            if not doc:
                continue
            tokens = list(jieba.cut(doc))
            corpus.append(tokens)
            self._bm25_corpus_ids.append(chunk_id)
            self._bm25_corpus_map[chunk_id] = {
                "content": doc,
                "documentId": (meta or {}).get("documentId"),
                "knowledgeBaseId": (meta or {}).get("knowledgeBaseId"),
                "metadata": meta or {},
            }

        if corpus:
            self._bm25_index = BM25Okapi(corpus)
            self._bm25_cache_key = cache_key

    def _rrf_merge(
        self,
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion 融合排序"""
        scores: dict[str, float] = {}
        hit_data: dict[str, dict[str, Any]] = {}
        relevance_scores: dict[str, float] = {}
        max_bm25_score = max((float(hit.get("score") or 0) for hit in bm25_hits), default=0.0)

        for rank, hit in enumerate(vector_hits):
            cid = hit["chunkId"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            hit_data[cid] = hit
            relevance_scores[cid] = max(0.0, min(1.0, float(hit.get("score") or 0)))

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunkId"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            if cid not in hit_data:
                hit_data[cid] = hit
                relevance_scores[cid] = (
                    max(0.0, min(1.0, float(hit.get("score") or 0) / max_bm25_score))
                    if max_bm25_score > 0
                    else 0.0
                )

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for cid in sorted_ids:
            hit = hit_data[cid].copy()
            hit["fusionScore"] = scores[cid]
            hit["score"] = relevance_scores.get(cid, 0.0)
            results.append(hit)
        return results

    def _clear_bm25_cache(self) -> None:
        self._bm25_index = None
        self._bm25_cache_key = None
        self._bm25_corpus_ids = []
        self._bm25_corpus_map = {}

    def _distance_to_score(self, distance: float) -> float:
        """余弦距离转相似度：cosine distance ∈ [0, 2]，score ∈ [0, 1]"""
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0
        return max(0.0, min(1.0, 1.0 - value / 2.0))
