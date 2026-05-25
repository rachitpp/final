# RAG System — Complete Pipeline

---

## Part 1: Ingestion Pipeline
> Run once via `create_db.py` whenever PDFs change.

```
PDFs on disk  (pdfs/ folder or single file)
        │
        ▼
┌───────────────────────────────────┐
│  LOAD  (ingestion/loader.py)      │
│  • PyMuPDF opens each PDF         │
│  • One Document per page          │
│  • Reads TOC → maps each page     │
│    to its section heading         │
│  • Metadata: source, page,        │
│    section                        │
└──────────────┬────────────────────┘
               │  list[Document]  (one per page)
               ▼
┌──────────────────────────────────────────┐
│  SPLIT  (ingestion/splitter.py)          │
│  • RecursiveCharacterTextSplitter        │
│  • chunk_size    = 1 000 chars           │
│  • chunk_overlap = 200 chars             │
│  • Separators: \n\n → \n → ". " → " "   │
│  • Metadata inherited from source page   │
└──────────────┬───────────────────────────┘
               │  list[Document]  (chunks)
               ▼
┌─────────────────────────────────────────────────────────┐
│  CONTEXTUAL CHUNKING  (ingestion/contextualizer.py)     │
│  [OPTIONAL — toggle contextual_chunking_enabled = True] │
│  • One Gemini Flash call per chunk                      │
│  • Generates a 1–3 sentence retrieval header:           │
│    topic + document provenance                          │
│  • Header prepended to the chunk text before embedding  │
│  • Runs in parallel via ThreadPoolExecutor (8 workers)  │
│  • Improves embedding quality for isolated chunks       │
└──────────────┬──────────────────────────────────────────┘
               │  list[Document]  (contextualized chunks)
               ▼
┌────────────────────────────────────────────────────────────┐
│  EMBED + PERSIST  (ingestion/vector_store.py)              │
│  • Google text-embedding-004 via Vertex AI (768 dims)      │
│  • Batched at 200 chunks / call  (Vertex cap = 250)        │
│  • Stored in Qdrant Cloud — collection: rag_documents      │
│  • Cosine similarity + keyword payload index on "source"   │
└──────────────┬─────────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
[rag_documents]   SUMMARIZE  (ingestion/summarizer.py)
Qdrant collection │  • One Gemini Flash call per PDF
                  │  • First 8 000 chars used as input
                  │  • 4–6 sentence summary: topic, key
                  │    concepts, document type / level
                  ▼
            ┌──────────────────────────────────────┐
            │  EMBED + PERSIST SUMMARIES            │
            │  • Same embedding model (768 dims)    │
            │  • Qdrant Cloud collection:           │
            │    rag_document_summaries             │
            │    (one vector per PDF)               │
            └──────────────────────────────────────┘
                  [rag_document_summaries]
                  Qdrant collection
```

**Result:** two Qdrant collections — chunks + summaries — ready for query time.

---

## Part 2: RAG Query Pipeline
> Runs on every user question.

```
User types a question in the Streamlit UI
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  QUERY REWRITE  (retrieval/rewrite.py)                   │
│  • Looks at the last 4 conversation turns (memory)       │
│  • Gemini Flash resolves pronouns and implicit refs      │
│    into a fully self-contained standalone query          │
│  • No-op when there is no chat history                   │
│                                                          │
│  e.g.  "what are its advantages?"                        │
│     →  "What are the advantages of attention             │
│         mechanisms in transformers?"                     │
└──────────────────────────┬───────────────────────────────┘
                           │  rewritten query
                           ▼
┌──────────────────────────────────────────────────────────┐
│  DOCUMENT ROUTER  (retrieval/router.py)                  │
│  • Embeds the rewritten query                            │
│  • Similarity-searches the summary collection            │
│  • Picks the best-matching PDF(s)                        │
│  • Includes any PDF within 0.15 cosine gap of the top    │
│    score — handles genuine cross-document queries        │
│  • Returns a list of source filenames                    │
└──────────────────────────┬───────────────────────────────┘
                           │  selected source(s)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  HyDE GENERATION  (retrieval/hyde.py)                    │
│  [enabled by default — toggle hyde_enabled = False]      │
│  • Gemini Flash writes a dense hypothetical passage      │
│    that WOULD answer the query (never shown to user)     │
│  • Embeds closer to real document chunks than the        │
│    bare question does                                    │
│  • Used ONLY as the vector retrieval query               │
└──────────────────────────┬───────────────────────────────┘
                           │
           ┌───────────────┴──────────────┐
           │                              │
           ▼                              ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│  BM25 RETRIEVAL      │   │  VECTOR MMR RETRIEVAL            │
│  (retrieval/         │   │  (retrieval/retrievers.py)       │
│   retrievers.py)     │   │                                  │
│  • Keyword matching  │   │  • Semantic similarity via       │
│  • Pre-built index   │   │    text-embedding-004            │
│    per PDF + global  │   │  • Maximal Marginal Relevance    │
│  • Query: rewritten  │   │    (MMR) for result diversity    │
│    user question     │   │  • fetch_k = 50 candidates,      │
│  • Returns top 12    │   │    returns top 12 diverse        │
│    keyword matches   │   │  • Filtered to routed source(s)  │
│                      │   │    via Qdrant payload filter      │
│                      │   │  • Query: HyDE passage           │
└──────────┬───────────┘   └──────────────┬───────────────────┘
           │  12 chunks                   │  12 chunks
           └──────────────┬───────────────┘
                          │  merged + deduplicated
                          ▼  (up to ~24 unique candidates)
┌──────────────────────────────────────────────────────────┐
│  CROSS-ENCODER RERANK + FILTER  (retrieval/reranker.py)  │
│  • Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (local)  │
│  • Scores every (query, chunk) pair with bi-attention    │
│  • Logit → sigmoid → confidence score [0, 1]             │
│  • Sorts descending, keeps top 12                        │
│  • Drops anything below threshold = 0.20                 │
│  • Attaches rerank_score to each chunk's metadata        │
└──────────────────────────┬───────────────────────────────┘
                           │
                  ┌────────┴────────┐
           no chunks kept      chunks kept
                  │                │
                  ▼                ▼
         Graceful fallback    FORMAT CONTEXT
         message returned     (retrieval/formatter.py)
                              numbered context block
                                       │
                                       ▼
                          ┌─────────────────────────────────┐
                          │  ANSWER GENERATION  (llm/)      │
                          │  • Gemini 2.5 Flash, Vertex AI  │
                          │  • Grounded system prompt:      │
                          │    cite as (filename, p.N),     │
                          │    no hallucination             │
                          │  • Streaming: tokens yielded    │
                          │    live to the UI               │
                          │  • temperature = 0.2            │
                          └──────────────┬──────────────────┘
                                         │  streamed tokens
                                         ▼
                          ┌─────────────────────────────────┐
                          │  CONVERSATION MEMORY            │
                          │  (conversation/memory.py)       │
                          │  • Completed answer + query     │
                          │    saved to memory              │
                          │  • Bounded ring: last 4 turns   │
                          │  • Fed to next query's rewriter │
                          └─────────────────────────────────┘
                                         │
                                         ▼
                          Answer streams into Streamlit UI.
                          Sources panel shows (file, page,
                          section) for each kept chunk.
```

---

## Observability

Every major step is decorated with `@traceable` (LangSmith). Traced steps:

```
ingestion_pipeline → load_documents → split_documents →
contextualize_chunks → create_vector_store → summarize_documents

rag_pipeline.answer → query_rewrite → document_router →
hyde_generation → hybrid_retrieve → rerank_and_filter
```

Full traces — inputs, outputs, latency, token counts — are visible in the LangSmith dashboard per run.

---

## Key Numbers (default settings)

| Parameter | Value |
|---|---|
| Chunk size | 1 000 chars |
| Chunk overlap | 200 chars |
| Embedding dimensions | 768 |
| BM25 candidates | 12 |
| Vector MMR candidates | 12 (fetched from 50) |
| After dedupe | ≤ 24 unique candidates |
| After rerank cap | ≤ 12 |
| After threshold filter | variable (drop score < 0.20) |
| Conversation memory | last 4 turns |
| HyDE passage length | ≤ 256 tokens |
| LLM answer length | ≤ 2 048 tokens |
| Routing similarity gap | 0.15 cosine |
