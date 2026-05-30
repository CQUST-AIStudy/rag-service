from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.responses import ApiError


class DashScopeEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.dashscope_api_key:
            raise ApiError(503, "DASHSCOPE_API_KEY 未配置，无法调用嵌入模型")
        if not texts:
            return []

        url = f"{self.settings.dashscope_compat_base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
            "dimensions": self.settings.embedding_dimensions,
        }
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ApiError(503, "嵌入模型服务不可用") from exc

        if response.status_code >= 400:
            raise ApiError(503, f"嵌入模型调用失败: HTTP {response.status_code}")

        body = response.json()
        data = body.get("data") or []
        data.sort(key=lambda item: item.get("index", 0))
        return [item.get("embedding") or [] for item in data]


class DashScopeReranker:
    def __init__(self, settings: Settings):
        self.settings = settings

    def rerank(self, query: str, documents: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if not documents:
            return []
        if not self.settings.dashscope_api_key:
            raise ApiError(503, "DASHSCOPE_API_KEY 未配置，无法调用重排模型")

        payload = {
            "model": self.settings.rerank_model,
            "input": {
                "query": query,
                "documents": [{"text": item["content"]} for item in documents],
            },
            "parameters": {
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
        }
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    self.settings.dashscope_rerank_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ApiError(503, "重排模型服务不可用") from exc

        if response.status_code >= 400:
            raise ApiError(503, f"重排模型调用失败: HTTP {response.status_code}")

        ranked = self._parse_results(response.json(), documents)
        if not ranked:
            return documents[:top_n]
        ranked.sort(key=lambda item: item.get("rerankScore", 0), reverse=True)
        return ranked[:top_n]

    def _parse_results(
        self,
        payload: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        output = payload.get("output") or payload.get("data") or payload
        results = output.get("results") or output.get("rerank_results") or []
        ranked: list[dict[str, Any]] = []
        for result in results:
            index = result.get("index")
            if index is None:
                index = result.get("document_index")
            if index is None or not (0 <= int(index) < len(documents)):
                continue
            score = (
                result.get("relevance_score")
                or result.get("score")
                or result.get("rerank_score")
                or 0
            )
            item = dict(documents[int(index)])
            item["rerankScore"] = float(score)
            ranked.append(item)
        return ranked
