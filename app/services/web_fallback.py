from typing import Any

import httpx

from app.core.config import Settings


class TavilyWebFallbackService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        if not self.settings.web_fallback_enabled or not self.settings.tavily_api_key:
            return []

        result_count = max(1, min(max_results or self.settings.web_max_results, 10))
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": result_count,
            "search_depth": "basic",
        }
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(self.settings.tavily_search_url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError:
            return []

        results = data.get("results") or []
        normalized: list[dict[str, Any]] = []
        for item in results[:result_count]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or item.get("snippet") or "").strip()
            normalized.append(
                {
                    "documentId": f"web:{url}" if url else "web",
                    "chunkId": f"web:{len(normalized) + 1}",
                    "fileName": title or url or "Web 资料",
                    "content": content,
                    "chunkContent": content,
                    "score": float(item.get("score") or 0),
                    "rerankScore": float(item.get("score") or 0),
                    "source": "web",
                    "metadata": {
                        "url": url,
                        "title": title,
                        "source": "web",
                    },
                }
            )
        return normalized
