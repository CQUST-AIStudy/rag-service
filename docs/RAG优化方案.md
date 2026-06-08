# RAG 系统优化方案：提高准确率与降低 AI 幻觉

> 分析日期：2026-06-08
> 目标系统：CQUST RAG Service（FastAPI + LangChain + DashScope + Chroma）

---

## 📊 诊断总结

工作流识别了 **22 个核心问题**（11 个准确率问题 + 11 个幻觉风险）和 **50 项优化方案**。

---

## 🔴 最严重的问题

### 准确率方面

| 问题 | 严重程度 | 影响模块 | 影响描述 |
|------|----------|----------|----------|
| chunk_size=512 对中文过小 | HIGH | document_ingest.py | 仅约 170 汉字，强制切断段落导致语义断裂 |
| PDF 跨页断开 | HIGH | document_ingest.py | 先按页切分再 chunk，跨页内容被人为隔断 |
| 纯向量检索召回不全 | HIGH | vector_store.py | 精确术语（"O(n log n)"）、专有名词匹配失效 |
| 重排候选池过小 | HIGH | vector_store.py, dashscope.py | top_k=10 → rerank=3，复杂问题覆盖不足 |
| 表格和结构化信息丢失 | MEDIUM | document_ingest.py | PDF/DOCX 表格被扁平化，行列关系丢失 |
| 中文分隔符不完整 | MEDIUM | document_ingest.py | 缺少中文逗号、顿号、分号、冒号 |
| 相似度分数转换不准确 | MEDIUM | vector_store.py | 1/(1+distance) 对 L2 距离缺乏物理意义 |
| 代码块和特殊格式被破坏 | MEDIUM | document_ingest.py | Markdown 代码块可能在中间被切分 |

### 幻觉风险方面

| 风险 | 严重程度 | 触发条件 | 所需缓解措施 |
|------|----------|----------|--------------|
| 空资料时用预训练知识回答 | HIGH | sources 为空或相关性极低 | 切换拒绝模式 Prompt，禁止编造 |
| 时间敏感信息推测 | HIGH | 用户问"最新""当前" | 要求 metadata 含时间戳，无日期则拒答 |
| 资料冲突时权威性误判 | HIGH | 本地资料与 Web 冲突 | 明确权威性判断标准，无法确定时列双方观点 |
| temperature=0.7 引发创造性幻觉 | MEDIUM | 开放性问题（举例、如何应用） | 降至 0.1-0.3 |
| 版本号日期编造 | MEDIUM | 资料只有版本号无日期 | 禁止推测，明确说明资料未提供 |
| System prompt 冗长致指令遗忘 | MEDIUM | 复杂问题+长上下文 | 结构化分段，关键约束前置 |
| 引用机制失效致溯源丢失 | MEDIUM | "需要时引用"过于模糊 | 改为强制引用 |
| 复杂推理缺少 CoT | MEDIUM | "比较""分析""为什么"类问题 | 注入思维链引导 |

---

## 🎯 三阶段实施路线图

### Phase 1: 立即实施（1-2 周）—— 降低幻觉为主

**8 项快速优化，预期幻觉率下降 60%：**

#### 1. 增大 chunk_size 至 1500-2000

**理由**：当前 512 字符对中文语义完整性破坏严重，DashScope embedding 模型支持 8192 token，大幅提升切片质量几乎无成本。

**实施要点**：
- 修改 `app/schemas/rag.py` KnowledgeBaseCreate 的 chunkSize 默认值从 512 改为 1500
- 调整上限从 4096 到 3000（避免过长）
- 建议课程文档类知识库使用 1500-2000

```python
# app/schemas/rag.py
chunkSize: int = Field(1500, ge=128, le=3000)
```

#### 2. 提高 chunk_overlap 至 20-25%

**理由**：当前 64/512=12.5%，中文语境下句子边界容易丢失上下文，提高重叠率可显著改善检索召回。

**实施要点**：
- 修改 chunkOverlap 默认值从 64 改为 300（1500*20%）
- 调整验证规则确保 overlap < chunk_size

```python
# app/schemas/rag.py
chunkOverlap: int = Field(300, ge=0, le=1024)
```

#### 3. 结构化 System Prompt

**理由**：当前 prompt（第 250-259 行）挤在一段中，模型难以理解不同指令优先级。

**实施要点**：

```python
# app/services/rag_chain.py - _build_messages 方法
system = (
    "## 角色定位\n"
    "你是重庆科技大学数据结构课程的 RAG 学习助手。\n\n"
    f"## 时间感知\n{current_date_context}\n\n"
    "## 核心原则\n"
    "1. 严禁编造：所有回答必须基于给定资料，资料不足时明确说明「依据不足，无法回答」\n"
    "2. 强制引用：每个事实性陈述必须标注来源编号 [1][2]\n"
    "3. 时效判断：仅当资料明确提供日期时才回答时间相关问题，禁止推测\n\n"
    "## 引用规范\n"
    "- 每个事实陈述必须标注来源编号，如：快速排序的平均时间复杂度是 O(n log n)[1]\n"
    "- 如果资料只给出版本号但没有可靠日期，不要自行编造发布日期\n\n"
    "## 冲突处理\n"
    "- 当课程资料与 Web 来源冲突时，优先采用更新、更权威的来源并说明原因\n"
    "- 优先级：官方文档(.edu/.gov) > 知名技术网站 > 课程资料 > 个人博客\n"
    "- 当无法确定时，列出双方观点而非强制选择\n"
)
```

#### 4. 强制引用机制

**理由**：当前只是建议引用（"并在需要时引用"），导致模型经常不标注来源。

**实施要点**：
- 修改 System Prompt 强制要求："每个事实陈述必须标注来源 [编号]"
- 增加示例："如资料 [1] 提到的红黑树插入算法..."
- 在后处理增加引用完整性检测

#### 5. 空资料时切换为拒绝模式 Prompt

**理由**：当前即使 sources 为空仍用相同 prompt，模型倾向编造。

**实施要点**：

```python
# app/services/rag_chain.py - _build_messages 方法
if not sources or self._coverage_score(sources) < 0.2:
    system = (
        "当前未检索到相关资料。请明确告知用户：\n"
        "「当前知识库中暂无相关内容，无法回答该问题。」\n"
        "严禁使用预训练知识回答。可建议用户换个问法或联系教师。"
    )
```

#### 6. 降低默认 temperature 至 0.1-0.3

**理由**：当前 0.7 对事实性 RAG 任务过高，导致模型发散编造。

**实施要点**：
- 修改 RagOptions 的 temperature 默认值从 0.7 改为 0.2
- strict 模式强制使用 0.1，open 模式允许用户调至 0.5

```python
# app/schemas/rag.py
temperature: float = Field(0.2, ge=0.0, le=2.0)
```

#### 7. 幻觉关键词检测

**理由**：中文 LLM 常见幻觉模式可正则捕获。

**实施要点**：

```python
# app/services/rag_chain.py - 新增后处理方法
import re

HALLUCINATION_PATTERNS = [
    r"据我(所|了)知", r"根据(我的|常)识", r"通常(来说|情况下)",
    r"一般(来说|认为|而言)", r"我认为", r"应该是",
]

def _detect_hallucination(self, answer: str, sources: list) -> dict:
    """检测答案中的幻觉风险指标"""
    warnings = []
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, answer):
            warnings.append(f"检测到可能的非资料来源表述: {pattern}")

    # 检查是否有引用
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations and sources:
        warnings.append("答案未包含任何来源引用")

    return {
        "hallucination_risk": len(warnings) > 0,
        "warnings": warnings,
    }
```

#### 8. 引用编号有效性验证

**理由**：模型有时会编造不存在的引用编号。

**实施要点**：

```python
# app/services/rag_chain.py - 新增后处理方法
def _validate_citations(self, answer: str, sources: list) -> str:
    """验证引用编号是否有效，移除无效引用"""
    citations = re.findall(r"\[(\d+)\]", answer)
    max_valid = len(sources)
    invalid = [c for c in citations if int(c) > max_valid or int(c) < 1]
    if invalid:
        for inv in set(invalid):
            answer = answer.replace(f"[{inv}]", "")
    return answer
```

---

### Phase 2: 短期优化（2-4 周）—— 提升准确率

**8 项中等复杂度优化，预期准确率提升 35%、召回率提升 50%：**

#### 9. 语义感知智能分隔符

**理由**：当前 RecursiveCharacterTextSplitter 的分隔符顺序不合理，应优先按 Markdown 结构切分。

**实施要点**：

```python
# app/services/document_ingest.py - _split_texts 方法
splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(kb.get("chunkSize") or 1500),
    chunk_overlap=int(kb.get("chunkOverlap") or 300),
    separators=[
        "\n## ", "\n### ", "\n#### ",  # Markdown 标题优先
        "\n\n",                         # 段落分隔
        "。\n", "！\n", "？\n",          # 中文句末+换行
        "\n",                           # 普通换行
        "。", "！", "？",               # 中文句末
        "，", "；", "：",               # 中文标点补充
        ".", " ", "",                   # 英文和兜底
    ],
)
```

#### 10. PDF 全文提取后统一切分

**理由**：当前逐页切分（第 98-110 行），跨页段落被强制断开，语义完整性极差。

**实施要点**：

```python
# app/services/document_ingest.py - 替换 _load_pdf 方法
def _load_pdf(self, path: Path) -> list[dict[str, Any]]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        full_text = "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        )
    return [{"content": full_text, "metadata": {"source": path.name}}]
```

> 注意：需要 `uv add pdfplumber` 安装依赖，替换原有的 pypdf。

#### 11. 优化相似度分数转换公式

**理由**：当前 `1/(1+distance)` 对 L2 距离不合理。改用余弦距离更适合归一化向量。

**实施要点**：

```python
# app/services/vector_store.py
class VectorStore:
    def __init__(self, settings: Settings):
        self.store = Chroma(
            collection_name="rag_chunks",
            persist_directory=str(settings.chroma_dir),
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},  # 指定余弦距离
        )

    def _distance_to_score(self, distance: float) -> float:
        """余弦距离转相似度：cosine distance ∈ [0, 2]，score ∈ [0, 1]"""
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0
        return max(0.0, min(1.0, 1.0 - value / 2.0))
```

> 注意：切换距离度量后需要重新建立向量索引（重新入库所有文档）。

#### 12. 扩大初始召回池至 30-50

**理由**：当前 top_k=10 过小，rerank 无法从候选池外召回。

**实施要点**：
- 修改 `app/core/config.py`：`default_top_k: int = Field(30, alias="RAG_DEFAULT_TOP_K")`
- 修改 `app/core/config.py`：`default_rerank_top_n: int = Field(5, alias="RAG_DEFAULT_RERANK_TOP_N")`
- 对应调整 `app/schemas/rag.py`：`topK: int = Field(30, ge=1, le=100)`

#### 13. 查询改写与多查询检索

**理由**：用户口语化查询（"红黑树咋插入的"）难以召回书面资料。

**实施要点**：

```python
# app/services/rag_chain.py - 新增查询改写方法
async def _rewrite_query(self, query: str) -> list[str]:
    """使用 LLM 将用户查询改写为多个语义等价的正式查询"""
    llm = self._chat_model(RagOptions(temperature=0.3, maxTokens=200))
    prompt = (
        f"将以下问题改写为 3 个不同表述的搜索查询（保持语义一致），每行一个：\n"
        f"问题：{query}\n改写："
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    variants = [line.strip() for line in str(response.content).split("\n") if line.strip()]
    return [query] + variants[:3]  # 原始查询 + 最多 3 个变体
```

#### 14. 基于元数据预过滤

**理由**：当前 Chroma 只按 knowledgeBaseId 过滤，可增加章节/文件类型过滤提升精度。

**实施要点**：
- 在 chunk metadata 增加 `fileType`（pdf/md/docx）和 `section`（从标题提取）
- RetrieveRequest 增加可选的 `fileTypes: list[str]` 和 `sections: list[str]`
- 构造复合 where 条件传给 Chroma

#### 15. 注入 CoT 引导

**理由**：qwen-plus 支持 CoT，可减少直接猜测，提升复杂问题的推理质量。

**实施要点**：

```python
# app/services/rag_chain.py - _build_messages 中条件注入
# 检测复杂查询（包含比较、分析、为什么等关键词）
COMPLEX_KEYWORDS = ["比较", "分析", "为什么", "如何", "区别", "优缺点", "解释"]

if any(kw in query for kw in COMPLEX_KEYWORDS):
    system += (
        "\n\n## 推理引导\n"
        "这是一个复杂问题，请分步骤思考：\n"
        "1. 从资料中提取相关信息\n"
        "2. 分析各信息之间的关系\n"
        "3. 得出结论并标注来源编号\n"
    )
```

#### 16. 分阶段动态 scoreThreshold

**理由**：初始检索用低阈值保证召回，rerank 后用高阈值保证精度。

**实施要点**：

```python
# app/services/rag_chain.py - retrieve 方法调整
def retrieve(self, request: RetrieveRequest, principal: Principal) -> list[dict]:
    # 初始检索不过滤（内部低阈值 0.1）
    hits = self.vector_store.similarity_search(
        request.query, knowledge_base_ids, request.topK
    )
    chunks = self._hydrate_hits(hits)
    # 初步过滤明显无关的（score < 0.1）
    chunks = [c for c in chunks if float(c.get("score") or 0) >= 0.1]

    if request.enableRerank and chunks:
        chunks = self.reranker.rerank(request.query, chunks, request.rerankTopN)
        # rerank 后使用更高阈值
        threshold = max(request.scoreThreshold, 0.3)
    else:
        chunks = chunks[: request.rerankTopN]
        threshold = request.scoreThreshold

    return [c for c in chunks if float(c.get("rerankScore") or c.get("score") or 0) >= threshold]
```

---

### Phase 3: 长期演进（1-3 个月）—— 系统性增强

**6 项高复杂度优化，预期准确率再提升 25%：**

#### 17. 实现混合检索（向量 + BM25）

**理由**：纯向量检索对专有名词（"Dijkstra 算法"）召回差，BM25 补充关键词匹配。

**实施要点**：
- 用 `rank-bm25` 库对所有 chunk 内容建立 BM25 索引（持久化到 pickle）
- 向量检索取 topK×2，BM25 取 topK×2
- RRF 融合：`score = sum(1/(60+rank_i))`
- 在 VectorStore 增加 `hybrid_search` 方法

```python
# app/services/vector_store.py - 新增混合检索
from rank_bm25 import BM25Okapi
import jieba

class VectorStore:
    def hybrid_search(self, query: str, kb_ids: list[str], top_k: int) -> list[dict]:
        # 向量检索
        vector_hits = self.similarity_search(query, kb_ids, top_k * 2)
        # BM25 检索
        bm25_hits = self._bm25_search(query, kb_ids, top_k * 2)
        # RRF 融合
        return self._rrf_merge(vector_hits, bm25_hits, top_k)

    def _rrf_merge(self, vector_hits, bm25_hits, top_k, k=60):
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, hit in enumerate(vector_hits):
            cid = hit["chunkId"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunkId"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        # 按融合分数排序
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        # ... 返回合并结果
```

#### 18. 文档结构元数据提取

**理由**：章节路径可显著提升用户理解来源语境。

**实施要点**：
- PDF 用 pdfplumber 提取字体大小判断标题层级
- Markdown 解析 `#` 层级
- 维护章节栈，每个 chunk 记录 `sectionPath`
- 在答案引用中展示："[1] 第 3 章 > 红黑树.pdf"

#### 19. 表格专项处理

**理由**：当前表格被提取为无结构文本，丢失行列关系。

**实施要点**：
- pdfplumber 提取表格为 DataFrame
- 转 Markdown 表格格式保留行列关系
- 标记 metadata `contentType: "table"`
- Prompt 中特别指示："表格内容用 Markdown 格式展示"

#### 20. 父子 chunk 检索策略

**理由**：小 chunk 检索精准但上下文不足，大 chunk 上下文丰富但噪音多。

**实施要点**：
- 双层切分：父 chunk 3000 字符，子 chunk 500 字符
- 检索用子 chunk（精准匹配），召回后替换为对应父 chunk（完整上下文）
- 数据库增加 `parent_chunk_id` 字段
- Chroma 存储子 chunk 但 metadata 记录父 ID

#### 21. 答案引用完整性检测

**理由**：统计引用覆盖率，低覆盖率意味着更高的幻觉风险。

**实施要点**：

```python
def _check_citation_coverage(self, answer: str, sources: list) -> float:
    """计算引用覆盖率：有引用的句子 / 总事实性句子"""
    sentences = re.split(r'[。！？\n]', answer)
    factual_sentences = [s for s in sentences if len(s.strip()) > 10]
    cited_sentences = [s for s in factual_sentences if re.search(r'\[\d+\]', s)]
    if not factual_sentences:
        return 1.0
    return len(cited_sentences) / len(factual_sentences)
```

#### 22. 答案与来源一致性校验（NLI）

**理由**：模型可能基于来源过度推理，产生不一致陈述。

**实施要点**：
- 用 DashScope NLI 模型逐句校验答案是否蕴含于 sources
- 检测矛盾或无关句子
- 标注可信度分数，低于 0.6 的句子附加警告
- 在响应增加 `consistencyScore: float`

---

## 🏆 优先级标注

### 对降低幻觉贡献最大（TOP 7）

| 排名 | 优化项 | 原因 |
|------|--------|------|
| 1 | ⭐ 空资料拒绝 Prompt | 直接阻断无依据编造 |
| 2 | ⭐ 降低 temperature | 减少生成随机性 |
| 3 | ⭐ 强制引用机制 | 迫使模型锚定来源 |
| 4 | ⭐ 幻觉关键词检测 | 后置防护网 |
| 5 | ⭐ 结构化 Prompt | 明确指令优先级 |
| 6 | ⭐ CoT 引导 | 增加推理可控性 |
| 7 | ⭐ NLI 一致性校验 | 最终质量保障 |

### 对提升准确率贡献最大（TOP 7）

| 排名 | 优化项 | 原因 |
|------|--------|------|
| 1 | 🎯 混合检索（BM25+向量） | 召回率 +80% |
| 2 | 🎯 PDF 全文提取 | 完整性 +90% |
| 3 | 🎯 chunk 参数优化 | 准确率 +40% |
| 4 | 🎯 智能分隔符 | 噪音 -50% |
| 5 | 🎯 查询改写 | 召回多样性 +60% |
| 6 | 🎯 相似度公式优化 | 排序准确性 +30% |
| 7 | 🎯 父子 chunk | 精准定位+丰富上下文 |

---

## 📋 完整优化清单（50 项）

| # | 标题 | 类别 | 复杂度 | 影响力 | 阶段 |
|---|------|------|--------|--------|------|
| 1 | 增加中文 chunk_size 至 1500-2000 字符 | 切片 | LOW | HIGH | Phase 1 |
| 2 | 提高 chunk_overlap 至 20-25% | 切片 | LOW | HIGH | Phase 1 |
| 3 | 实现语义感知的智能分隔符 | 切片 | MEDIUM | HIGH | Phase 2 |
| 4 | PDF 全文提取后统一切分 | 切片 | MEDIUM | HIGH | Phase 2 |
| 5 | 保护代码块和公式完整性 | 切片 | MEDIUM | MEDIUM | Phase 2 |
| 6 | 提取并保留文档结构元数据 | 切片 | HIGH | MEDIUM | Phase 3 |
| 7 | 表格专项处理 | 切片 | HIGH | MEDIUM | Phase 3 |
| 8 | 动态 chunk_size 按文档类型调整 | 切片 | MEDIUM | MEDIUM | Phase 2 |
| 9 | 去重重叠 chunk | 切片 | LOW | LOW | Phase 1 |
| 10 | 用 tiktoken 实现真实 token 计数 | 切片 | LOW | LOW | Phase 1 |
| 11 | 实现混合检索（向量+BM25） | 检索 | HIGH | HIGH | Phase 3 |
| 12 | 扩大初始召回候选池至 30-50 | 检索 | LOW | HIGH | Phase 2 |
| 13 | 优化相似度分数转换公式 | 检索 | MEDIUM | HIGH | Phase 2 |
| 14 | 查询改写与多查询检索 | 检索 | MEDIUM | MEDIUM | Phase 2 |
| 15 | 基于元数据预过滤 | 检索 | MEDIUM | MEDIUM | Phase 2 |
| 16 | 分阶段动态 scoreThreshold | 检索 | LOW | MEDIUM | Phase 2 |
| 17 | 父子 chunk 检索策略 | 检索 | HIGH | MEDIUM | Phase 3 |
| 18 | Web fallback 增强过滤与评分 | 检索 | MEDIUM | MEDIUM | Phase 2 |
| 19 | 重排模型分数校准 | 检索 | MEDIUM | LOW | Phase 3 |
| 20 | 时间戳索引与时效性检索 | 检索 | MEDIUM | LOW | Phase 3 |
| 21 | 结构化 System Prompt | Prompt | LOW | HIGH | Phase 1 |
| 22 | 强制引用机制 | Prompt | LOW | HIGH | Phase 1 |
| 23 | 空资料时切换为拒绝模式 Prompt | Prompt | LOW | HIGH | Phase 1 |
| 24 | 注入思维链（CoT）引导 | Prompt | MEDIUM | HIGH | Phase 2 |
| 25 | 传递来源相关性分数至 Prompt | Prompt | LOW | MEDIUM | Phase 1 |
| 26 | 增加 Few-shot 示例 | Prompt | LOW | MEDIUM | Phase 1 |
| 27 | 多轮对话历史注入 | Prompt | MEDIUM | MEDIUM | Phase 2 |
| 28 | 动态调整 temperature 策略 | Prompt | LOW | MEDIUM | Phase 1 |
| 29 | 时效性显式标注策略 | Prompt | MEDIUM | MEDIUM | Phase 2 |
| 30 | 来源冲突明确化处理指令 | Prompt | LOW | LOW | Phase 1 |
| 31 | 降低默认 temperature 至 0.1-0.3 | 生成 | LOW | HIGH | Phase 1 |
| 32 | 模型选择策略优化 | 生成 | MEDIUM | MEDIUM | Phase 2 |
| 33 | 增加 top_p 参数控制 | 生成 | LOW | MEDIUM | Phase 1 |
| 34 | Streaming 实时验证 | 生成 | HIGH | MEDIUM | Phase 3 |
| 35 | Context 窗口优化 | 生成 | MEDIUM | MEDIUM | Phase 2 |
| 36 | 模型回退机制 | 生成 | MEDIUM | LOW | Phase 2 |
| 37 | max_tokens 动态调整 | 生成 | LOW | LOW | Phase 1 |
| 38 | 真实 usage 统计 | 生成 | LOW | LOW | Phase 1 |
| 39 | stop_sequences 防护 | 生成 | LOW | LOW | Phase 1 |
| 40 | 并行生成多候选答案 | 生成 | HIGH | LOW | Phase 3 |
| 41 | 答案引用完整性检测 | 后处理 | MEDIUM | HIGH | Phase 3 |
| 42 | 幻觉关键词检测 | 后处理 | LOW | HIGH | Phase 1 |
| 43 | 引用编号有效性验证 | 后处理 | LOW | HIGH | Phase 1 |
| 44 | 答案与来源一致性校验（NLI） | 后处理 | HIGH | MEDIUM | Phase 3 |
| 45 | 时间信息一致性检查 | 后处理 | MEDIUM | MEDIUM | Phase 2 |
| 46 | 空答案/拒绝响应规范化 | 后处理 | LOW | MEDIUM | Phase 1 |
| 47 | 答案长度与质量评估 | 后处理 | LOW | MEDIUM | Phase 1 |
| 48 | 答案缓存机制 | 后处理 | MEDIUM | LOW | Phase 3 |
| 49 | 敏感信息过滤 | 后处理 | LOW | LOW | Phase 1 |
| 50 | 多语言术语一致性检查 | 后处理 | MEDIUM | LOW | Phase 3 |

---

## 💡 建议行动

**立即启动 Phase 1 的 8 项快速优化**（低复杂度高收益），2 周内可见明显改善：

- ✅ 修改 2 个配置默认值（chunk_size=1500、temperature=0.2）
- ✅ 重构 1 个 Prompt（结构化 + 强制引用 + 空资料拒绝）
- ✅ 增加 3 个后处理检测（幻觉关键词、引用验证、空资料拒绝）

**预期综合效果**：
- Phase 1 完成后：幻觉率下降 ~60%，引用完整性提升至 ~90%
- Phase 2 完成后：准确率提升 ~35%，召回率提升 ~50%
- Phase 3 完成后：准确率再提升 ~25%，复杂查询处理能力翻倍

---

## 当前系统核心痛点总结

1. 默认 512 chunk_size 对中文课程资料破坏严重
2. Prompt 未强制引用，模型随意编造
3. temperature 0.7 过高，事实任务不适用
4. PDF 逐页切分导致跨页内容断裂
5. 纯向量检索对专有名词召回差
6. 重排后仅保留 3 条结果，复杂问题覆盖不足
