import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.auth import Principal
from app.core.config import Settings
from app.core.responses import ApiError
from app.schemas.rag import AssistantHistoryMessage, AssistantRequest, ChatRequest, RagOptions, RetrieveRequest
from app.services.dashscope import DashScopeReranker
from app.services.repository import RagRepository, normalize_mode
from app.services.vector_store import VectorStore
from app.services.web_fallback import TavilyWebFallbackService

CHINA_STANDARD_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")

# --- 幻觉检测模式 ---
HALLUCINATION_PATTERNS = [
    r"据我(所|了)知",
    r"根据(我的|常)识",
    r"通常(来说|情况下)",
    r"一般(来说|认为|而言)",
    r"我认为",
    r"应该是",
    r"大概(是|在)",
    r"可能(是|在|有)",
]

# --- 复杂查询关键词（触发 CoT）---
COMPLEX_KEYWORDS = ["比较", "分析", "为什么", "如何", "区别", "优缺点", "解释", "原理", "证明", "推导"]

MAX_ASSISTANT_HISTORY = 10

ASSISTANT_SYSTEM_PROMPT = """
## 角色定位
你是学习助手，面向学生提供学习问答、概念解释、复杂度分析和代码思路辅导。

## 回答要求
- 默认使用中文回答，除非用户明确要求其他语言。
- 回答要准确、清晰、适合学生理解。
- 可以给出步骤、例子、伪代码或 C/C++/Python 片段，但不要编造课程资料或外部链接。
- 如果用户问题不完整，先给出最可能的解释，并指出需要补充的信息。
""".strip()

WEB_ASSISTANT_SYSTEM_PROMPT = """
## 角色定位
你是学习助手，可以结合联网检索结果回答学生问题。

## 回答要求
- 默认使用中文回答，除非用户明确要求其他语言。
- 优先依据提供的联网资料回答，并在关键事实后标注来源编号，如 [1]。
- 如果联网资料不足，请明确说明检索结果有限，再给出谨慎的通用解释。
- 不要编造不存在的链接、标题、发布日期或来源。
""".strip()


@dataclass
class ChatPreparation:
    conversation_id: str
    knowledge_base_ids: list[str]
    sources: list[dict[str, Any]]
    effective_mode: str
    used_web: bool
    coverage_score: float


@dataclass
class PostProcessResult:
    answer: str
    hallucination_warnings: list[str] = field(default_factory=list)
    citation_coverage: float = 1.0


class RagChainService:
    def __init__(
        self,
        settings: Settings,
        repository: RagRepository,
        vector_store: VectorStore,
        reranker: DashScopeReranker,
        web_fallback: TavilyWebFallbackService,
    ):
        self.settings = settings
        self.repository = repository
        self.vector_store = vector_store
        self.reranker = reranker
        self.web_fallback = web_fallback

    # ─── Public API ───────────────────────────────────────────────────────

    def retrieve(self, request: RetrieveRequest, principal: Principal) -> list[dict[str, Any]]:
        request = self._retrieve_request_with_defaults(request)
        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        for kb_id in knowledge_base_ids:
            self.repository.require_knowledge_base(kb_id, principal)

        # 混合检索：向量 + BM25 RRF 融合
        hits = self.vector_store.hybrid_search(request.query, knowledge_base_ids, request.topK)
        chunks = self._hydrate_hits(hits, knowledge_base_ids)

        if request.enableRerank and chunks:
            chunks = self.reranker.rerank(request.query, chunks, request.rerankTopN)
            # rerank 后使用更高阈值
            threshold = max(request.scoreThreshold, 0.2)
        else:
            chunks = chunks[: request.rerankTopN]
            threshold = request.scoreThreshold

        return [
            item
            for item in chunks
            if float(item.get("rerankScore") or item.get("score") or 0) >= threshold
        ]

    async def chat(self, request: ChatRequest, principal: Principal) -> dict[str, Any]:
        request = self._chat_request_with_defaults(request)
        prepared = await self._prepare_chat(request, principal)
        answer = await self._generate_answer(request.query, prepared.sources, request.options)
        post = self._post_process(answer, prepared.sources)
        usage = self._usage_placeholder(request.query, post.answer, prepared.sources)
        self._save_chat(principal, prepared, request, post.answer, usage)
        return {
            "answer": post.answer,
            "conversationId": prepared.conversation_id,
            "sources": prepared.sources,
            "effectiveMode": prepared.effective_mode,
            "usedWeb": prepared.used_web,
            "coverageScore": prepared.coverage_score,
            "citationCoverage": post.citation_coverage,
            "hallucinationWarnings": post.hallucination_warnings,
            "usage": usage,
        }

    async def stream_chat(self, request: ChatRequest, principal: Principal) -> AsyncIterator[str]:
        request = self._chat_request_with_defaults(request)
        conversation_id = request.conversationId
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        try:
            prepared = await self._prepare_chat(request, principal)
            conversation_id = prepared.conversation_id
            sources = prepared.sources
            yield self._sse(
                "retrieval",
                {
                    "sources": sources,
                    "effectiveMode": prepared.effective_mode,
                    "usedWeb": prepared.used_web,
                    "coverageScore": prepared.coverage_score,
                },
            )
            async for delta in self._stream_answer(request.query, sources, request.options):
                answer_parts.append(delta)
                yield self._sse("delta", {"content": delta})
            answer = "".join(answer_parts)
            post = self._post_process(answer, sources)
            usage = self._usage_placeholder(request.query, post.answer, sources)
            self._save_chat(principal, prepared, request, post.answer, usage)
            yield self._sse(
                "done",
                {
                    "conversationId": conversation_id,
                    "usage": usage,
                    "effectiveMode": prepared.effective_mode,
                    "usedWeb": prepared.used_web,
                    "coverageScore": prepared.coverage_score,
                    "citationCoverage": post.citation_coverage,
                    "hallucinationWarnings": post.hallucination_warnings,
                },
            )
        except ApiError as exc:
            yield self._sse("error", {"message": exc.message, "code": exc.code})
        except Exception:
            yield self._sse("error", {"message": "RAG 生成失败，请稍后重试", "code": 500})

    async def stream_assistant(self, request: AssistantRequest, principal: Principal) -> AsyncIterator[str]:
        request = self._assistant_request_with_defaults(request)
        conversation_id = request.conversationId
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        effective_mode = request.mode
        used_web = False
        coverage_score = 0.0

        try:
            if request.mode == "rag":
                prepared = await self._prepare_assistant_rag(request, principal)
                conversation_id = prepared.conversation_id
                sources = prepared.sources
                effective_mode = prepared.effective_mode
                used_web = prepared.used_web
                coverage_score = prepared.coverage_score
                messages = self._build_messages(request.query, sources)
                knowledge_base_ids = prepared.knowledge_base_ids
            else:
                conversation_id = self.repository.create_or_touch_conversation(
                    principal,
                    request.conversationId,
                    [],
                    request.query,
                )
                sources = await self._assistant_web_sources(request)
                used_web = bool(sources)
                effective_mode = "web" if request.mode == "web" else ("ai-web" if request.enableWebSearch else "ai")
                messages = self._build_assistant_messages(request, sources)
                knowledge_base_ids = []

            yield self._sse(
                "retrieval",
                {
                    "sources": sources,
                    "effectiveMode": effective_mode,
                    "usedWeb": used_web,
                    "coverageScore": coverage_score,
                },
            )

            async for delta in self._stream_messages(messages, request.options):
                answer_parts.append(delta)
                yield self._sse("delta", {"content": delta})

            answer = "".join(answer_parts).strip()
            post = self._post_process(answer, sources)
            usage = self._usage_placeholder(request.query, post.answer, sources)
            self._save_assistant_chat(
                principal,
                conversation_id,
                knowledge_base_ids,
                request,
                post.answer,
                sources,
                usage,
                used_web,
                coverage_score,
                effective_mode,
            )
            yield self._sse(
                "done",
                {
                    "conversationId": conversation_id,
                    "usage": usage,
                    "effectiveMode": effective_mode,
                    "usedWeb": used_web,
                    "coverageScore": coverage_score,
                    "citationCoverage": post.citation_coverage,
                    "hallucinationWarnings": post.hallucination_warnings,
                },
            )
        except ApiError as exc:
            yield self._sse("error", {"message": exc.message, "code": exc.code})
        except Exception:
            yield self._sse("error", {"message": "AI 助手生成失败，请稍后重试", "code": 500})

    # ─── Chat Preparation ─────────────────────────────────────────────────

    async def _prepare_chat(
        self,
        request: ChatRequest,
        principal: Principal,
    ) -> ChatPreparation:
        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        knowledge_bases = [
            self.repository.require_knowledge_base(kb_id, principal)
            for kb_id in knowledge_base_ids
        ]
        options = self._rag_options_with_defaults(request.options)
        retrieve_request = RetrieveRequest(
            query=request.query,
            knowledgeBaseIds=knowledge_base_ids,
            topK=options.topK,
            enableRerank=options.enableRerank,
            rerankTopN=options.rerankTopN,
            scoreThreshold=options.scoreThreshold,
        )
        conversation_id = self.repository.create_or_touch_conversation(
            principal,
            request.conversationId,
            knowledge_base_ids,
            request.query,
        )
        sources = self.retrieve(retrieve_request, principal)
        coverage_score = self._coverage_score(sources)
        effective_mode = self._effective_mode(request.mode, knowledge_bases)
        allow_web_search = any(bool(item.get("allowWebSearch")) for item in knowledge_bases)
        used_web = False

        if (
            self.settings.web_fallback_enabled
            and effective_mode == "open"
            and allow_web_search
            and coverage_score < self.settings.coverage_threshold
        ):
            web_sources = await self.web_fallback.search(request.query, self.settings.web_max_results)
            if web_sources:
                sources = [*sources, *web_sources]
                used_web = True

        return ChatPreparation(
            conversation_id=conversation_id,
            knowledge_base_ids=knowledge_base_ids,
            sources=sources,
            effective_mode=effective_mode,
            used_web=used_web,
            coverage_score=coverage_score,
        )

    async def _prepare_assistant_rag(
        self,
        request: AssistantRequest,
        principal: Principal,
    ) -> ChatPreparation:
        if not request.knowledgeBaseIds:
            raise ApiError(400, "RAG 模式需要选择课程空间")

        knowledge_base_ids = self._normalize_kb_ids(request.knowledgeBaseIds, principal)
        knowledge_bases = [
            self.repository.require_knowledge_base(kb_id, principal)
            for kb_id in knowledge_base_ids
        ]
        options = self._rag_options_with_defaults(request.options)
        retrieve_request = RetrieveRequest(
            query=request.query,
            knowledgeBaseIds=knowledge_base_ids,
            topK=options.topK,
            enableRerank=options.enableRerank,
            rerankTopN=options.rerankTopN,
            scoreThreshold=options.scoreThreshold,
        )
        conversation_id = self.repository.create_or_touch_conversation(
            principal,
            request.conversationId,
            knowledge_base_ids,
            request.query,
        )
        sources = self.retrieve(retrieve_request, principal)
        coverage_score = self._coverage_score(sources)
        effective_mode = "open" if request.enableWebSearch else "strict"
        allow_web_search = any(bool(item.get("allowWebSearch")) for item in knowledge_bases)
        used_web = False

        if self.settings.web_fallback_enabled and request.enableWebSearch and allow_web_search:
            web_sources = await self.web_fallback.search(request.query, self.settings.web_max_results)
            if web_sources:
                sources = [*sources, *web_sources]
                used_web = True

        return ChatPreparation(
            conversation_id=conversation_id,
            knowledge_base_ids=knowledge_base_ids,
            sources=sources,
            effective_mode=effective_mode,
            used_web=used_web,
            coverage_score=coverage_score,
        )

    # ─── Query Rewriting (Phase 2: 多查询检索) ───────────────────────────

    async def _rewrite_query(self, query: str) -> list[str]:
        """使用 LLM 将用户查询改写为多个语义等价的正式查询"""
        llm = self._chat_model(RagOptions(temperature=0.3, maxTokens=200))
        prompt = (
            f"将以下问题改写为 3 个不同表述的搜索查询（保持语义一致），每行一个：\n"
            f"问题：{query}\n改写："
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            variants = [line.strip() for line in str(response.content).split("\n") if line.strip()]
            return [query] + variants[:3]
        except Exception:
            return [query]

    # ─── Generation ───────────────────────────────────────────────────────

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
        messages = self._build_messages(query, sources)
        async for delta in self._stream_messages(messages, options):
            yield delta

    async def _stream_messages(
        self,
        messages: list[Any],
        options: RagOptions,
    ) -> AsyncIterator[str]:
        if not self.settings.dashscope_api_key:
            raise ApiError(503, "DASHSCOPE_API_KEY 未配置，无法调用生成模型")
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

    # ─── Prompt Construction ──────────────────────────────────────────────

    def _build_messages(self, query: str, sources: list[dict[str, Any]]) -> list[Any]:
        context = "\n\n".join(
            f"[{index + 1}] {item.get('fileName') or item.get('documentId', '未知来源')}\n{item['content']}"
            for index, item in enumerate(sources)
        )
        current_date_context = self._current_date_context()
        coverage = self._coverage_score(sources)

        # 空资料或极低覆盖时切换为拒绝模式
        if not sources or coverage < 0.15:
            system = (
                "## 角色定位\n"
                "你是RAG 学习助手。\n\n"
                f"## 时间感知\n{current_date_context}\n\n"
                "## 指令\n"
                "当前未检索到足够相关的资料。请明确告知用户：\n"
                "「当前知识库中暂无与该问题高度相关的内容，无法给出可靠回答。」\n"
                "严禁使用预训练知识回答事实性问题。可建议用户换个问法或联系教师。\n"
                "如果问题是通用性的编程概念或基础知识，可以简要说明但必须注明「此为通用知识，非课程资料」。\n"
            )
        else:
            system = (
                "## 角色定位\n"
                "你是RAG 学习助手。\n\n"
                f"## 时间感知\n{current_date_context}\n\n"
                "## 核心原则\n"
                "1. 严禁编造：所有回答必须基于给定资料，资料不足时明确说明「依据不足，无法回答」\n"
                "2. 强制引用：每个事实性陈述必须标注来源编号 [1][2]...\n"
                "3. 时效判断：仅当资料明确提供日期时才回答时间相关问题，禁止推测\n\n"
                "## 引用规范\n"
                "- 每个事实陈述必须标注来源编号，如：快速排序的平均时间复杂度是 O(n log n)[1]\n"
                "- 如果多个来源支持同一观点，可标注多个编号 [1][3]\n"
                "- 如果资料只给出版本号但没有可靠日期，不要自行编造发布日期\n\n"
                "## 冲突处理\n"
                "- 当课程资料与 Web 来源冲突时，优先采用更新、更权威的来源并说明原因\n"
                "- 优先级：官方文档(.edu/.gov) > 知名技术网站 > 课程资料 > 个人博客\n"
                "- 当无法确定时，列出双方观点而非强制选择\n"
            )

            # CoT 引导：复杂查询时注入推理步骤
            if any(kw in query for kw in COMPLEX_KEYWORDS):
                system += (
                    "\n## 推理引导\n"
                    "这是一个需要分析的问题，请分步骤思考：\n"
                    "1. 从资料中提取相关信息\n"
                    "2. 分析各信息之间的关系\n"
                    "3. 得出结论并标注来源编号\n"
                )

        user = f"资料：\n{context or '（未检索到相关资料）'}\n\n问题：{query}"
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def _build_assistant_messages(
        self,
        request: AssistantRequest,
        sources: list[dict[str, Any]],
    ) -> list[Any]:
        messages: list[Any] = []
        if sources:
            context = "\n\n".join(
                f"[{index + 1}] {item.get('fileName') or item.get('documentId', '联网资料')}\n"
                f"URL: {self._source_url(item) or '无'}\n"
                f"{item.get('content') or item.get('chunkContent') or ''}"
                for index, item in enumerate(sources)
            )
            system = (
                WEB_ASSISTANT_SYSTEM_PROMPT
                + f"\n\n## 时间感知\n{self._current_date_context()}"
                + "\n\n## 联网资料\n"
                + context
            )
        else:
            system = ASSISTANT_SYSTEM_PROMPT + f"\n\n## 时间感知\n{self._current_date_context()}"
            if request.mode == "web" or request.enableWebSearch:
                system += "\n\n## 联网状态\n本次未检索到可用联网资料，请明确说明联网结果为空，再谨慎回答。"

        messages.append(SystemMessage(content=system))
        messages.extend(self._history_to_messages(request.history))
        messages.append(HumanMessage(content=request.query))
        return messages

    def _current_date_context(self) -> str:
        today = datetime.now(CHINA_STANDARD_TIME).date()
        return f"当前日期是 {today.isoformat()}（北京时间，Asia/Shanghai）。"

    # ─── Post-Processing ──────────────────────────────────────────────────

    def _post_process(self, answer: str, sources: list[dict[str, Any]]) -> PostProcessResult:
        """后处理：验证引用、检测幻觉"""
        warnings = self._detect_hallucination(answer, sources)
        answer = self._validate_citations(answer, sources)
        coverage = self._check_citation_coverage(answer, sources)
        return PostProcessResult(
            answer=answer,
            hallucination_warnings=warnings,
            citation_coverage=coverage,
        )

    def _detect_hallucination(self, answer: str, sources: list[dict[str, Any]]) -> list[str]:
        """检测答案中的幻觉风险指标"""
        warnings: list[str] = []
        for pattern in HALLUCINATION_PATTERNS:
            match = re.search(pattern, answer)
            if match:
                warnings.append(f"检测到可能的非资料来源表述: '{match.group()}'")

        # 有资料但无引用
        citations = re.findall(r"\[(\d+)\]", answer)
        if not citations and sources:
            warnings.append("答案未包含任何来源引用")

        return warnings

    def _validate_citations(self, answer: str, sources: list[dict[str, Any]]) -> str:
        """验证引用编号是否有效，移除无效引用"""
        max_valid = len(sources)
        if max_valid == 0:
            return answer
        citations = set(re.findall(r"\[(\d+)\]", answer))
        for cite in citations:
            if int(cite) > max_valid or int(cite) < 1:
                answer = answer.replace(f"[{cite}]", "")
        return answer

    def _check_citation_coverage(self, answer: str, sources: list[dict[str, Any]]) -> float:
        """计算引用覆盖率：有引用的句子 / 总事实性句子"""
        if not sources:
            return 1.0
        sentences = re.split(r"[。！？\n]", answer)
        factual_sentences = [s for s in sentences if len(s.strip()) > 10]
        if not factual_sentences:
            return 1.0
        cited_sentences = [s for s in factual_sentences if re.search(r"\[\d+\]", s)]
        return len(cited_sentences) / len(factual_sentences)

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _normalize_kb_ids(self, knowledge_base_ids: list[str], principal: Principal) -> list[str]:
        normalized = [str(item) for item in knowledge_base_ids if str(item).strip()]
        if normalized:
            return normalized
        bases = self.repository.list_knowledge_bases(principal)
        if not bases:
            raise ApiError(404, "暂无可用知识库")
        return [bases[0]["id"]]

    def _hydrate_hits(self, hits: list[dict[str, Any]], knowledge_base_ids: list[str]) -> list[dict[str, Any]]:
        chunk_ids = [item["chunkId"] for item in hits if item.get("chunkId")]
        chunks = self.repository.get_chunks_by_ids_for_knowledge_bases(chunk_ids, knowledge_base_ids)
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
                    "fusionScore": float(hit.get("fusionScore") or 0),
                    "metadata": metadata,
                }
            )
        return results

    def _retrieve_request_with_defaults(self, request: RetrieveRequest) -> RetrieveRequest:
        updates: dict[str, Any] = {}
        fields_set = request.model_fields_set
        if "topK" not in fields_set:
            updates["topK"] = self.settings.default_top_k
        if "rerankTopN" not in fields_set:
            updates["rerankTopN"] = self.settings.default_rerank_top_n
        if "scoreThreshold" not in fields_set:
            updates["scoreThreshold"] = self.settings.default_score_threshold
        return request.model_copy(update=updates) if updates else request

    def _rag_options_with_defaults(self, options: RagOptions) -> RagOptions:
        updates: dict[str, Any] = {}
        fields_set = options.model_fields_set
        if "topK" not in fields_set:
            updates["topK"] = self.settings.default_top_k
        if "rerankTopN" not in fields_set:
            updates["rerankTopN"] = self.settings.default_rerank_top_n
        if "scoreThreshold" not in fields_set:
            updates["scoreThreshold"] = self.settings.default_score_threshold
        return options.model_copy(update=updates) if updates else options

    def _chat_request_with_defaults(self, request: ChatRequest) -> ChatRequest:
        options = self._rag_options_with_defaults(request.options)
        return request.model_copy(update={"options": options}) if options is not request.options else request

    def _assistant_request_with_defaults(self, request: AssistantRequest) -> AssistantRequest:
        options = self._rag_options_with_defaults(request.options)
        return request.model_copy(update={"options": options}) if options is not request.options else request

    async def _assistant_web_sources(self, request: AssistantRequest) -> list[dict[str, Any]]:
        should_search = request.mode == "web" or request.enableWebSearch
        if not should_search or not self.settings.web_fallback_enabled:
            return []
        return await self.web_fallback.search(request.query, self.settings.web_max_results)

    def _history_to_messages(self, history: list[AssistantHistoryMessage]) -> list[Any]:
        if not history:
            return []
        normalized: list[Any] = []
        for item in history[-MAX_ASSISTANT_HISTORY:]:
            content = str(item.content or "").strip()
            if not content:
                continue
            role = str(item.role or "").strip().lower()
            if role in {"assistant", "ai"}:
                normalized.append(AIMessage(content=content[:8000]))
            elif role == "user":
                normalized.append(HumanMessage(content=content[:8000]))
        return normalized

    def _source_url(self, source: dict[str, Any]) -> str:
        metadata = source.get("metadata") or {}
        return str(source.get("url") or metadata.get("url") or "")

    def _coverage_score(self, sources: list[dict[str, Any]]) -> float:
        if not sources:
            return 0
        return max(float(item.get("rerankScore") or item.get("score") or 0) for item in sources)

    def _effective_mode(self, request_mode: str | None, knowledge_bases: list[dict[str, Any]]) -> str:
        requested = normalize_mode(request_mode, "")
        if requested:
            return requested
        return "open" if any(item.get("defaultMode") == "open" for item in knowledge_bases) else "strict"

    def _save_chat(
        self,
        principal: Principal,
        prepared: ChatPreparation,
        request: ChatRequest,
        answer: str,
        usage: dict[str, Any],
    ) -> None:
        self.repository.add_conversation_message(prepared.conversation_id, "user", request.query)
        self.repository.add_conversation_message(prepared.conversation_id, "assistant", answer, prepared.sources, usage)
        self.repository.save_qa_log(
            principal,
            prepared.conversation_id,
            prepared.knowledge_base_ids,
            request.query,
            answer,
            prepared.sources,
            used_web=prepared.used_web,
            coverage_score=prepared.coverage_score,
            mode=prepared.effective_mode,
        )

    def _save_assistant_chat(
        self,
        principal: Principal,
        conversation_id: str,
        knowledge_base_ids: list[str],
        request: AssistantRequest,
        answer: str,
        sources: list[dict[str, Any]],
        usage: dict[str, Any],
        used_web: bool,
        coverage_score: float,
        effective_mode: str,
    ) -> None:
        self.repository.add_conversation_message(conversation_id, "user", request.query)
        self.repository.add_conversation_message(conversation_id, "assistant", answer, sources, usage)
        self.repository.save_qa_log(
            principal,
            conversation_id,
            knowledge_base_ids,
            request.query,
            answer,
            sources,
            used_web=used_web,
            coverage_score=coverage_score,
            mode="open" if used_web else "strict",
            intent_type=f"assistant:{effective_mode}",
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
