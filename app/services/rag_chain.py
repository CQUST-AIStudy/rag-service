import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.auth import Principal
from app.core.config import Settings
from app.core.responses import ApiError
from app.schemas.rag import ChatRequest, RagOptions, RetrieveRequest
from app.services.dashscope import DashScopeReranker
from app.services.repository import RagRepository
from app.services.vector_store import VectorStore


class RagChainService:
    def __init__(
        self,
        settings: Settings,
        repository: RagRepository,
        vector_store: VectorStore,
        reranker: DashScopeReranker,
    ):
        self.settings = settings
        self.repository = repository
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(self, request: RetrieveRequest, principal: Principal) -> list[dict[str, Any]]:
        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        for kb_id in knowledge_base_ids:
            self.repository.require_knowledge_base(kb_id, principal)

        hits = self.vector_store.similarity_search(request.query, knowledge_base_ids, request.topK)
        chunks = self._hydrate_hits(hits)
        if request.enableRerank and chunks:
            chunks = self.reranker.rerank(request.query, chunks, request.rerankTopN)
        else:
            chunks = chunks[: request.rerankTopN]

        threshold = request.scoreThreshold
        return [
            item
            for item in chunks
            if float(item.get("rerankScore") or item.get("score") or 0) >= threshold
        ]

    async def chat(self, request: ChatRequest, principal: Principal) -> dict[str, Any]:
        conversation_id, sources = self._prepare_chat(request, principal)
        answer = await self._generate_answer(request.query, sources, request.options)
        usage = self._usage_placeholder(request.query, answer, sources)
        self._save_chat(principal, conversation_id, request, answer, sources, usage)
        return {
            "answer": answer,
            "conversationId": conversation_id,
            "sources": sources,
            "usage": usage,
        }

    async def stream_chat(self, request: ChatRequest, principal: Principal) -> AsyncIterator[str]:
        conversation_id = request.conversationId
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        try:
            conversation_id, sources = self._prepare_chat(request, principal)
            yield self._sse("retrieval", {"sources": sources})
            async for delta in self._stream_answer(request.query, sources, request.options):
                answer_parts.append(delta)
                yield self._sse("delta", {"content": delta})
            answer = "".join(answer_parts)
            usage = self._usage_placeholder(request.query, answer, sources)
            self._save_chat(principal, conversation_id, request, answer, sources, usage)
            yield self._sse("done", {"conversationId": conversation_id, "usage": usage})
        except ApiError as exc:
            yield self._sse("error", {"message": exc.message, "code": exc.code})
        except Exception as exc:
            yield self._sse("error", {"message": f"RAG 生成失败: {exc}", "code": 500})

    def _prepare_chat(
        self,
        request: ChatRequest,
        principal: Principal,
    ) -> tuple[str, list[dict[str, Any]]]:
        retrieve_request = RetrieveRequest(
            query=request.query,
            knowledgeBaseIds=request.knowledgeBaseIds,
            topK=request.options.topK,
            enableRerank=request.options.enableRerank,
            rerankTopN=request.options.rerankTopN,
            scoreThreshold=request.options.scoreThreshold,
        )
        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        conversation_id = self.repository.create_or_touch_conversation(
            principal,
            request.conversationId,
            knowledge_base_ids,
            request.query,
        )
        sources = self.retrieve(retrieve_request, principal)
        return conversation_id, sources

    def _normalize_kb_ids(self, knowledge_base_ids: list[str], principal: Principal) -> list[str]:
        normalized = [str(item) for item in knowledge_base_ids if str(item).strip()]
        if normalized:
            return normalized
        bases = self.repository.list_knowledge_bases(principal)
        if not bases:
            raise ApiError(404, "暂无可用知识库")
        return [bases[0]["id"]]

    def _hydrate_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunk_ids = [item["chunkId"] for item in hits if item.get("chunkId")]
        chunks = self.repository.get_chunks_by_ids(chunk_ids)
        hit_by_id = {item["chunkId"]: item for item in hits}
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            hit = hit_by_id.get(chunk["chunkId"]) or {}
            metadata = {**(chunk.get("metadata") or {}), **(hit.get("metadata") or {})}
            results.append(
                {
                    "documentId": chunk["documentId"],
                    "chunkId": chunk["chunkId"],
                    "fileName": chunk.get("fileName") or metadata.get("fileName") or "",
                    "content": chunk["content"],
                    "chunkContent": chunk["content"],
                    "score": float(hit.get("score") or 0),
                    "rerankScore": float(hit.get("score") or 0),
                    "metadata": metadata,
                }
            )
        return results

    async def _generate_answer(
        self,
        query: str,
        sources: list[dict[str, Any]],
        options: RagOptions,
    ) -> str:
        if not self.settings.dashscope_api_key:
            raise ApiError(503, "DASHSCOPE_API_KEY 未配置，无法调用生成模型")
        messages = self._build_messages(query, sources)
        llm = self._chat_model(options)
        response = await llm.ainvoke(messages)
        return str(response.content or "").strip()

    async def _stream_answer(
        self,
        query: str,
        sources: list[dict[str, Any]],
        options: RagOptions,
    ) -> AsyncIterator[str]:
        if not self.settings.dashscope_api_key:
            raise ApiError(503, "DASHSCOPE_API_KEY 未配置，无法调用生成模型")
        messages = self._build_messages(query, sources)
        llm = self._chat_model(options)
        async for chunk in llm.astream(messages):
            content = getattr(chunk, "content", "") or ""
            if content:
                yield str(content)

    def _chat_model(self, options: RagOptions) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.chat_model,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_compat_base_url,
            temperature=options.temperature,
            max_tokens=options.maxTokens,
            timeout=120,
        )

    def _build_messages(self, query: str, sources: list[dict[str, Any]]) -> list[Any]:
        context = "\n\n".join(
            f"[{index + 1}] {item.get('fileName') or item['documentId']}\n{item['content']}"
            for index, item in enumerate(sources)
        )
        system = (
            "你是重庆科技大学数据结构课程的 RAG 学习助手。"
            "请优先依据给定资料回答，保持准确、清晰，并在需要时引用 [1]、[2] 这样的来源编号。"
            "如果资料不足，请明确说明依据不足，不要编造。"
        )
        user = f"资料：\n{context or '（未检索到相关资料）'}\n\n问题：{query}"
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def _save_chat(
        self,
        principal: Principal,
        conversation_id: str,
        request: ChatRequest,
        answer: str,
        sources: list[dict[str, Any]],
        usage: dict[str, Any],
    ) -> None:
        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        self.repository.add_conversation_message(conversation_id, "user", request.query)
        self.repository.add_conversation_message(conversation_id, "assistant", answer, sources, usage)
        self.repository.save_qa_log(
            principal,
            conversation_id,
            knowledge_base_ids,
            request.query,
            answer,
            sources,
        )

    def _usage_placeholder(
        self,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, int]:
        return {
            "embeddingTokens": len(query),
            "rerankTokens": sum(len(item.get("content") or "") for item in sources),
            "llmPromptTokens": len(query) + sum(len(item.get("content") or "") for item in sources),
            "llmCompletionTokens": len(answer),
        }

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
