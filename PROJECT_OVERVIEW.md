# Project Overview — Document-Grounded RAG Chat System

---

## What the Project Is

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets users have a multi-turn conversation with a collection of PDF documents. The system retrieves the most relevant passages from the indexed corpus and uses a large language model to generate a grounded, cited answer — it will not hallucinate information that is not in the documents.

**Key properties:**
- **Strictly grounded** — answers come only from uploaded PDFs
- **Source-cited** — every answer includes `(filename, page)` references
- **Multi-turn** — remembers the last 4 exchanges; resolves follow-up questions
- **Streaming** — tokens stream word-by-word as they are generated
- **Multi-document** — handles any number of PDFs; routes queries to the right document(s) automatically

---

## Directory Structure

```
final/
├── app.py                      Main Streamlit application (UI logic)
├── main.py                     CLI entry-point for testing
├── create_db.py                Run once to ingest PDFs → Qdrant
│
├── config/
│   └── settings.py             Central config dataclass (all knobs in one place)
│
├── ingestion/
│   ├── loader.py               PDF → LangChain Documents  (PyMuPDF)
│   ├── splitter.py             Recursive character chunking
│   ├── contextualizer.py       Optional Anthropic-style contextual chunk headers
│   ├── summarizer.py           Per-PDF summaries for document routing
│   └── vector_store.py         Qdrant create / load helpers
│
├── retrieval/
│   ├── router.py               Query → relevant PDF(s) via summary vectors
│   ├── hyde.py                 Hypothetical Document Embedding generation
│   ├── retrievers.py           BM25 + vector MMR, hybrid merge + dedupe
│   ├── reranker.py             Cross-encoder scoring + threshold filter
│   ├── rewrite.py              Follow-up → standalone query
│   └── formatter.py            Chunks → numbered context block for the prompt
│
├── llm/
│   ├── models.py               Gemini + embedding model factories
│   └── prompts.py              All LangChain prompt templates
│
├── conversation/
│   └── memory.py               Bounded ring-buffer of recent turns
│
├── pipelines/
│   ├── ingestion_pipeline.py   Orchestrates the full ingest flow
│   └── rag_pipeline.py         RAGPipeline class — query-time flow
│
├── styles/                     CSS split into 9 partials (Midnight v2 dark theme)
│   ├── _tokens.css
│   ├── _streamlit-chrome.css
│   ├── _base.css
│   ├── _sidebar.css
│   ├── _welcome.css
│   ├── _messages.css
│   ├── _chat-input.css
│   ├── _loading.css
│   └── _responsive.css
│
└── utils/
    └── logger.py               Centralized logger factory
```

---

## Tech Stack

### Language
**Python 3.11+**

### UI Framework
**Streamlit**
- Single-page chat interface
- Custom dark theme (Midnight v2) injected via `st.markdown` CSS partials
- Streaming via Python generator pattern
- Sidebar displays the indexed document library
- Starter prompt cards on the empty/welcome state

### LLM — Answer Generation
**Google Gemini 2.5 Flash** via Vertex AI (`langchain-google-genai`)
- Model: `gemini-2.5-flash`
- Region: `us-central1`
- Temperature: `0.2` (factual, low creativity)
- Max tokens: `2 048`
- Streaming enabled — tokens yielded live to the UI

### LLM — Utility Calls
The same Gemini 2.5 Flash model is reused for all internal LLM calls (non-streaming, short outputs):

| Purpose | Max tokens |
|---|---|
| Query rewriting | 128 |
| HyDE passage generation | 256 |
| Contextual chunk headers | 160 |
| Document summarization | 512 |

### Embeddings
**Google `text-embedding-004`** via Vertex AI
- Output dimensions: **768**
- Region: `us-central1`
- Batch size: 200 chunks per API call (Vertex cap is 250)
- Used for: chunk embeddings, summary embeddings, and query embedding at retrieval time

### Vector Database
**Qdrant Cloud** (managed cloud instance)
- Two collections:
  - `rag_documents` — all chunk embeddings
  - `rag_document_summaries` — one vector per PDF (powers routing)
- Distance metric: cosine similarity
- Payload index: keyword index on `metadata.source` for fast per-document filtering
- Client: `qdrant-client` + `langchain-qdrant`

### Keyword Retrieval
**BM25** (`langchain-community` `BM25Retriever`, backed by `rank-bm25`)
- One index built per source PDF plus one global index
- Indices built at startup by scrolling all chunks out of Qdrant (no separate store needed)
- Purely in-memory; rebuilt on every cold start

### Reranking
**Sentence-Transformers Cross-Encoder** (HuggingFace, runs locally)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- No API key — downloaded and cached by HuggingFace Hub
- Scores `(query, chunk)` pairs with bi-directional attention (much more accurate than bi-encoder cosine similarity)

### Orchestration
**LangChain** (`langchain-core`, `langchain-text-splitters`, `langchain-community`)
- Document abstraction and metadata propagation through the pipeline
- `ChatPromptTemplate` for all prompts
- Retriever interface for BM25 and Qdrant
- No LangChain agents or chains — plain Python orchestration for full control and debuggability

### Observability
**LangSmith**
- `@traceable` decorator on every pipeline step
- Captures inputs, outputs, latency, and token usage per step
- Metadata tags on the main trace: retriever strategy, model name, reranker

### PDF Parsing
**PyMuPDF** (`fitz`)
- Native text extraction per page
- Table-of-contents parsing — maps each page to its section heading

---

## Techniques & Design Decisions

### 1. Retrieval-Augmented Generation (RAG)
The LLM never answers from parametric memory. Every response is generated exclusively from passages retrieved from the document corpus. The system prompt explicitly forbids hallucination and instructs the model to say "I could not find the answer" when the retrieved context is insufficient.

### 2. Hybrid Retrieval — BM25 + Vector MMR
Two retrievers run in parallel and their results are merged:
- **BM25** excels at exact keyword matches and rare technical terms.
- **Vector retrieval** (dense embeddings) excels at semantic similarity and paraphrased queries.

Combining both gives better recall than either alone. Results are deduplicated by exact text before reranking.

### 3. Maximal Marginal Relevance (MMR)
The vector retriever uses MMR instead of pure similarity search. MMR balances relevance vs. diversity: it fetches 50 candidates but returns the 12 that cover the most ground, avoiding near-duplicate passages from adjacent paragraphs.

### 4. Hypothetical Document Embedding (HyDE)
Before vector retrieval, Gemini Flash generates a *hypothetical passage* that would appear in a textbook answering the query. This passage is embedded and used as the vector search query instead of the bare user question.

**Why:** the embedding of a well-formed answer-like passage sits much closer in vector space to actual document chunks than a short, informal question does. HyDE is used only for vector retrieval — BM25 still uses the rewritten user question because keyword matching works best with the actual query terms, not a hallucinated passage.

### 5. Cross-Encoder Reranking
After hybrid retrieval produces up to ~24 candidate chunks, a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores every `(query, chunk)` pair using bi-directional attention — it reads both inputs together rather than independently. This is far more accurate than bi-encoder cosine similarity but too slow to run over the entire corpus, hence the two-stage retrieve-then-rerank architecture.

### 6. Confidence Threshold Filtering
After reranking, chunks scoring below `0.20` (on a sigmoid-mapped `[0, 1]` scale) are discarded regardless of how many that removes. If all chunks are filtered out, the system returns a graceful fallback message rather than generating a low-confidence or hallucinated answer.

### 7. Document Routing
At ingestion time, each PDF gets a short 4–6 sentence summary embedded and stored in a separate Qdrant collection. At query time the rewritten query is compared against these summaries. Only the most relevant PDF(s) are searched, restricting both the BM25 index and the Qdrant vector filter to those sources.

Benefits:
- Reduces irrelevant chunks competing for rerank slots
- Speeds up retrieval on large corpora
- Handles cross-document queries by including multiple PDFs when their summaries fall within `0.15` cosine similarity of the top match

### 8. Query Rewriting
Follow-up questions often contain pronouns or implicit references ("what about its disadvantages?", "can you explain that further?"). Before any retrieval happens, Gemini Flash rewrites such questions into fully standalone queries using the last 4 conversation turns. If the question is already self-contained the rewriter returns it unchanged.

### 9. Contextual Chunking (Anthropic-style, optional)
When enabled, each chunk gets a 1–3 sentence retrieval-oriented header generated by Gemini Flash before embedding. The header states the chunk's topic and its location in the document.

**Why:** chunks extracted in isolation lose context — *"...the gradient of the loss with respect to W..."* has no retrieval signal without knowing it's from a chapter on backpropagation. The header gives the embedding model that surrounding context. This is a one-time ingestion cost (one LLM call per chunk) and runs in parallel via a thread pool (8 workers).

### 10. Bounded Conversation Memory
A simple in-memory ring buffer stores the last N `(user, assistant)` turn pairs. No external memory store or vector memory — the history is short enough to pass directly to the rewriter prompt. The buffer is bounded (default: 4 turns) to avoid prompt bloat and keep the rewrite call cheap. Memory is updated only after the full answer has finished streaming.

### 11. Streaming Output
The LLM answer streams token-by-token directly to the Streamlit UI via Python generators. The pipeline yields each token from Gemini's streaming API and the UI renders it as it arrives, giving a live typewriter effect.

### 12. Section-Aware Metadata
PyMuPDF extracts the PDF table of contents (TOC). Each page is mapped to the section it falls under by a backwards linear scan of the TOC. This section name travels with every chunk through the pipeline and appears in the sources panel alongside filename and page number.

---

## Prompt Design

| Prompt | Purpose | Key rules |
|---|---|---|
| `ANSWER_PROMPT` | Final user-facing answer | Answer only from context; cite as `(filename, p.N)`; match depth to question; no hallucination |
| `SUMMARIZE_PROMPT` | Per-PDF routing summary at ingestion | 4–6 sentences: topic, key concepts, document type |
| `HYDE_PROMPT` | Hypothetical retrieval passage | Dense, factual paragraph; no hedging or refusals |
| `REWRITE_PROMPT` | Follow-up → standalone query | Return only the rewritten question; if already standalone, return unchanged |
| `CONTEXTUALIZE_PROMPT` | Chunk context header at ingestion | 1–3 sentences; specific terms from the chunk; no invented facts |

---

## Configuration Reference

All settings live in `config/settings.py`. No other module hardcodes values.

| Parameter | Default | Description |
|---|---|---|
| `pdf_path` | `"pdfs"` | Source folder or single PDF path |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between adjacent chunks |
| `contextual_chunking_enabled` | `False` | Toggle Anthropic-style context headers |
| `embedding_model` | `text-embedding-004` | Vertex AI embedding model |
| `embedding_batch_size` | `200` | Chunks per Vertex API call |
| `llm_model` | `gemini-2.5-flash` | Vertex AI LLM model |
| `llm_temperature` | `0.2` | Low = factual, deterministic |
| `llm_max_tokens` | `2048` | Max answer length |
| `qdrant_collection` | `rag_documents` | Main chunk collection |
| `qdrant_summary_collection` | `rag_document_summaries` | Summary collection for routing |
| `qdrant_vector_size` | `768` | Must match embedding model output |
| `routing_similarity_gap` | `0.15` | Cosine gap for multi-document routing |
| `vector_k` | `12` | Chunks returned by vector retriever |
| `vector_fetch_k` | `50` | Candidates before MMR selection |
| `vector_mmr_lambda` | `0.5` | MMR balance — `0` = diverse, `1` = relevant |
| `bm25_k` | `12` | Chunks returned by BM25 |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local reranker |
| `rerank_top_n` | `12` | Max chunks after rerank sort |
| `rerank_score_threshold` | `0.20` | Minimum confidence to keep a chunk |
| `hyde_enabled` | `True` | Toggle HyDE generation |
| `hyde_max_tokens` | `256` | Max length of HyDE passage |
| `history_window` | `4` | Conversation turns kept in memory |
| `log_level` | `"INFO"` | Logging verbosity |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID for Vertex AI billing |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `CLUSTER_ENDPOINT` | Yes | Qdrant Cloud cluster URL |
| `LANGCHAIN_API_KEY` | Optional | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Set to `"true"` to enable LangSmith traces |

---

## How to Run

**1. Place PDF files in the `pdfs/` folder.**

**2. Ingest (run once, or whenever PDFs change):**
```bash
python create_db.py
```

**3. Start the chat interface:**
```bash
streamlit run app.py
```
