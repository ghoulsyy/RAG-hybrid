# RAG 检索增强生成问答系统 —— 混合检索

基于 **LangChain + ChromaDB + DeepSeek** 的 RAG 智能百科问答系统，支持文档上传、多轮对话、混合检索与量化评估。

## 项目亮点

- **混合检索**：稠密向量检索（语义匹配）+ BM25 稀疏检索（关键词匹配），并行召回、合并去重，兼顾语义理解和关键词命中
- **多轮对话**：基于 LangChain `RunnableWithMessageHistory` 的会话历史管理，支持上下文连续对话
- **量化评估**：RAGAS 框架 4 项指标评估检索和生成质量，30 条测试用例覆盖多领域问答
- **双入口**：Streamlit Web 界面（给人用）+ FastAPI REST API（给程序用），API 自带 Swagger 文档
- **文档管理**：MD5 去重、自动分块、向量化入库，一键上传 txt 即更新知识库

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户入口                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │  Streamlit   │  │  Streamlit   │  │   FastAPI    │  │
│   │  (问答界面)   │  │ (文件上传)    │  │  (REST API)  │  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│          │                 │                 │          │
│          └─────────────────┼─────────────────┘          │
│                            ▼                             │
│              ┌──────────────────────┐                   │
│              │     RagService       │                   │
│              │  (LangChain 链式编排) │                   │
│              └──────────┬───────────┘                   │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼               ▼               ▼               │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐      │
│  │ 混合检索    │  │ Prompt模板  │  │ 对话历史管理  │      │
│  │ Dense+BM25 │  │ (System+   │  │ FileChatMsgs │      │
│  │ 合并去重    │  │  Context+  │  │ (JSON持久化) │      │
│  │            │  │  History)  │  │              │      │
│  └─────┬──────┘  └────────────┘  └──────────────┘      │
│        │                                                 │
│   ┌────┴─────┐                                          │
│   ▼          ▼                                          │
│ ┌──────┐ ┌───────┐  ┌──────────┐  ┌───────────┐        │
│ │Dense │ │BM25   │  │ DeepSeek │  │ BGE-small │        │
│ │向量检索│ │稀疏检索│  │  LLM    │  │ Embedding │        │
│ │ Chroma│ │ jieba │  │          │  │           │        │
│ └──────┘ └───────┘  └──────────┘  └───────────┘        │
└─────────────────────────────────────────────────────────┘
```

### 检索流程

```
用户问题
   │
   ▼
┌──────────────┐    ┌──────────────┐
│ 稠密检索 (Dense)│  │ 稀疏检索 (BM25)│   ← 并行召回，各取 top_k×5
│ ChromaDB 语义  │    │ jieba 分词匹配 │
└──────┬───────┘    └──────┬───────┘
   │                      │
   └──────────┬───────────┘
              ▼
       ┌──────────────┐
       │  合并去重      │   ← 按 page_content 前200字符去重
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │  Top-K 结果    │   ← 取最终 top_k=4 条最相关文档
       └──────────────┘
```

## 技术栈

| 组件 | 技术 | 选型理由 |
|------|------|---------|
| LLM | DeepSeek-V4-flash | 中文能力强，性价比高 |
| Embedding | BGE-small-zh-v1.5 | 中文语义 SOTA，轻量 (<100MB) |
| 向量库 | ChromaDB | 轻量零配置，适合原型与中小规模 |
| 中文分词 | jieba | 中文分词首选，BM25 稀疏检索基础 |
| 编排框架 | LangChain | 生态成熟，链式编排 + 消息历史管理 |
| Web 界面 | Streamlit | 纯 Python，快速搭建交互式 UI |
| API 服务 | FastAPI | 高性能异步，自动生成 Swagger 文档 |
| 评估 | RAGAS | 业界标准 RAG 评估框架，4 维度量化 |

## 项目结构

```
RAG项目实例/
├── api.py                  # FastAPI 服务入口（/api/chat, /api/upload）
├── app_qa.py               # Streamlit 问答界面
├── app_file_uploader.py    # Streamlit 文件上传界面
├── rag.py                  # RAG 核心链：Prompt → LLM → 输出解析
├── hybrid_search.py        # 混合检索服务：Dense + BM25 + 合并去重
├── knolege_base.py         # 知识库管理：分块 + 向量化 + MD5 去重
├── vector_stores.py        # ChromaDB 向量库封装
├── sparse_retriever.py     # BM25 稀疏检索（jieba 分词 + 索引持久化）
├── reranker.py             # Reranker 精排（当前因版本冲突暂未启用）
├── file_history_store.py   # 对话历史 JSON 持久化存储
├── config_data.py          # 全局配置（模型/路径/分块/检索参数）
├── evaluation/
│   ├── eval_rag.py         # RAGAS 评估脚本
│   ├── test_dataset.json   # 测试数据集（21 条问答对）
│   └── results.csv         # 评估结果
├── data/
│   ├── chroma_db/          # ChromaDB 持久化向量数据
│   ├── bm25_index/         # BM25 稀疏索引持久化
│   └── md5.txt             # 文档 MD5 去重记录
├── chat_history/           # 会话历史 JSON 文件
├── .env.example            # 环境变量模板
└── requirements.txt        # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

`.env` 配置说明：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | — |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-small-zh-v1.5` |
| `CHAT_MODEL` | 对话模型 | `deepseek-v4-flash` |

### 3. 启动服务

**方式 A：Streamlit Web 界面（给人用）**

```bash
# 问答界面
streamlit run app_qa.py

# 文件上传界面
streamlit run app_file_uploader.py
```

**方式 B：FastAPI 服务（给程序用）**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- Swagger 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/health`

### 4. 上传知识库文档

**Streamlit**：打开文件上传界面 → 拖入 `.txt` 文件，系统自动分块、向量化、去重。

**API**：
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@你的文档.txt"
```

### 5. 开始问答

**Streamlit**：在聊天框输入问题即可。

**API**：
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "量子纠缠是什么？", "session_id": "user_001"}'
```

## API 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务状态检查 |
| GET | `/api/health` | 健康检查 + 向量库文档数量 |
| POST | `/api/chat` | 问答接口，传入 `question` 和可选 `session_id` |
| POST | `/api/upload` | 上传 `.txt` 文件至知识库 |

详细请求/响应格式见 Swagger 文档：`http://localhost:8000/docs`

## 核心设计决策

### 1. 为什么用混合检索而不是纯向量检索？

纯向量检索擅长捕捉语义相似性（"好吃" ≈ "美味"），但对专有名词、数字、公式等精确匹配场景表现差。BM25 基于词频统计，能精确匹配关键词。两者互补：

| 场景 | 向量检索 | BM25 |
|------|---------|------|
| "好吃的食物" vs "有什么美味佳肴" | ✅ 语义匹配 | ❌ 词不重叠 |
| "CaCO₃ + H₂O + CO₂" 精确搜索 | ❌ 被稀释 | ✅ 精确命中 |
| "138亿年前" 数字查询 | ❌ 向量难表征 | ✅ 关键词命中 |

### 2. 合并去重策略

Dense 和 BM25 各自召回 `top_k × 5` 条候选，两者可能有重叠。按 `page_content` 前 200 字符去重，保留先出现的那个。在不引入精排模型（Reranker）的前提下，直接在合并结果上截取 top_k 返回，简单高效。

### 3. 为什么分块大小选 500，重叠 45？

- **chunk_size=500**：太小丢失上下文（如"它"指代什么），太大稀释语义、引入噪声
- **chunk_overlap=45**：约 10% 重叠，保证关键信息不会恰好落在切割边界被截断

### 4. 为什么用 MD5 去重？

上传文档时计算全文 MD5，与 `data/md5.txt` 比对。相同内容不重复入库，避免知识库膨胀。

## 评估结果

基于 30 条测试用例（覆盖物理、生物、地理、化学等多领域），使用 RAGAS 框架评估：

| 指标 | 得分 | 含义 |
|------|------|------|
| **Faithfulness**（忠实度） | **0.965** | 答案是否完全基于检索上下文，有无幻觉 |
| **Context Recall**（上下文召回率） | **0.981** | 检索是否找全了回答所需的信息 |
| **Context Precision**（上下文精度） | **0.904** | 检索结果中相关文档的比例 |
| **Answer Relevancy**（答案相关性） | **0.760** | 答案是否紧扣问题 |

> 运行评估：`python evaluation/eval_rag.py`

### 评估解读

- **忠实度 0.965**：Prompt 明确要求"参考资料没有相关信息就如实告知，不要编造"，30 条中 26 条满分。少数低分案例（如"三峡大坝哪年通水"）是模型诚实回答"资料未提及"反而被判低，实际行为正确
- **召回率 0.981**：Dense + BM25 并行召回（各取 top_k×5）基本覆盖所有相关文档，仅 1 条因知识库本身缺乏资料而偏低
- **精度 0.904**：合并去重有效过滤冗余，无 Reranker 精排仍能保持较高精度
- **相关性 0.760**：部分回答偏冗长，可通过优化 Prompt 或限制回答长度进一步提升

## 后续优化方向

- [ ] 修复 FlagEmbedding 版本冲突，启用 Reranker 精排以提升 Context Precision
- [ ] 支持 PDF、Word、Markdown 等多格式文档解析
- [ ] 引入查询改写（Query Rewriting），提升模糊问题的检索效果
- [ ] 增加缓存层，相同/相似问题直接返回结果
- [ ] Docker 容器化部署
- [ ] 接入更多 LLM（通义千问、文心一言等）作为可选后端
