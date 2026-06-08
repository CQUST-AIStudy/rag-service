from app.services.vector_store import VectorStore


class FakeCollection:
    def __init__(self):
        self.calls: list[dict] = []

    def get(self, where, include=None):
        self.calls.append({"where": where, "include": include})
        kb_filter = where["knowledgeBaseId"]
        kb_id = kb_filter if isinstance(kb_filter, str) else kb_filter["$in"][0]
        return {
            "ids": [f"{kb_id}-chunk"],
            "documents": [f"{kb_id} binary tree"],
            "metadatas": [{"documentId": f"{kb_id}-doc", "knowledgeBaseId": kb_id}],
        }


class FakeStore:
    def __init__(self):
        self._collection = FakeCollection()
        self.deleted_ids: list[list[str]] = []

    def delete(self, ids):
        self.deleted_ids.append(list(ids))


def make_vector_store_without_chroma():
    vector_store = VectorStore.__new__(VectorStore)
    vector_store.store = FakeStore()
    vector_store._bm25_index = None
    vector_store._bm25_cache_key = None
    vector_store._bm25_corpus_ids = []
    vector_store._bm25_corpus_map = {}
    return vector_store


def test_rrf_merge_keeps_fusion_score_separate_from_relevance_score():
    vector_store = make_vector_store_without_chroma()

    results = vector_store._rrf_merge(
        [{"chunkId": "vector", "score": 0.82}],
        [{"chunkId": "bm25", "score": 12.0}, {"chunkId": "bm25-low", "score": 6.0}],
        top_k=3,
    )

    by_id = {item["chunkId"]: item for item in results}
    assert by_id["vector"]["fusionScore"] == 1 / 61
    assert by_id["vector"]["score"] == 0.82
    assert by_id["bm25"]["fusionScore"] == 1 / 61
    assert by_id["bm25"]["score"] == 1.0
    assert by_id["bm25-low"]["score"] == 0.5


def test_bm25_cache_is_scoped_by_knowledge_base_ids():
    vector_store = make_vector_store_without_chroma()

    vector_store._ensure_bm25_index(["kb-a"])
    first_ids = list(vector_store._bm25_corpus_ids)
    vector_store._ensure_bm25_index(["kb-a"])
    vector_store._ensure_bm25_index(["kb-b"])
    second_ids = list(vector_store._bm25_corpus_ids)

    assert first_ids == ["kb-a-chunk"]
    assert second_ids == ["kb-b-chunk"]
    assert len(vector_store.store._collection.calls) == 2


def test_delete_by_knowledge_base_deletes_vectors_and_clears_bm25_cache():
    vector_store = make_vector_store_without_chroma()
    vector_store._bm25_index = object()
    vector_store._bm25_cache_key = ("kb-a",)
    vector_store._bm25_corpus_ids = ["kb-a-chunk"]
    vector_store._bm25_corpus_map = {"kb-a-chunk": {"content": "binary tree"}}

    vector_store.delete_by_knowledge_base("kb-a")

    assert vector_store.store.deleted_ids == [["kb-a-chunk"]]
    assert vector_store._bm25_index is None
    assert vector_store._bm25_cache_key is None
    assert vector_store._bm25_corpus_ids == []
    assert vector_store._bm25_corpus_map == {}
