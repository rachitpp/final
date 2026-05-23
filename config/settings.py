# =============================================================
# Central configuration.
# Edit values here; no other module hardcodes settings.
# =============================================================
import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Settings:
    # --- Paths ---
    pdf_path: str = "pdfs"  # file OR folder of PDFs

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    # Contextual chunking (Anthropic-style): prepend an LLM-generated
    # topic/provenance header to each chunk before embedding. One Gemini
    # Flash call per chunk during ingestion. Off by default — flip on
    # before re-running create_db.py.
    contextual_chunking_enabled: bool = False

    # --- Embeddings (Vertex AI) ---
    embedding_model: str = "text-embedding-004"
    embedding_location: str = "us-central1"
    embedding_batch_size: int = 200  # Vertex caps at 250 per call

    # --- LLM (Gemini via Vertex AI) ---
    llm_model: str = "gemini-2.5-flash"
    llm_location: str = "us-central1"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # --- Vector store (Qdrant Cloud) ---
    qdrant_url: str = field(default_factory=lambda: os.environ.get("CLUSTER_ENDPOINT", "https://your-cluster-url.qdrant.io"))
    qdrant_collection: str = "rag_documents"
    qdrant_summary_collection: str = "rag_document_summaries"
    qdrant_vector_size: int = 768          # text-embedding-004 -> 768 dims

    # --- Document routing ---
    routing_similarity_gap: float = 0.15  # include extra PDFs within this gap of top score

    # --- Retrieval ---
    vector_k: int = 12         # final chunks from vector retriever
    vector_fetch_k: int = 50   # candidates before MMR diversification
    vector_mmr_lambda: float = 0.5  # 0=diversity, 1=relevance
    bm25_k: int = 12

    # --- Reranking + confidence filter ---
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 12             # max chunks kept after rerank
    # Threshold is on sigmoid(logit) so it lives in [0, 1].
    # 0.5 = "model thinks chunk is more relevant than not". Tune per corpus.
    rerank_score_threshold: float = 0.2

    # --- HYDE ---
    hyde_enabled: bool = True
    hyde_max_tokens: int = 256

    # --- Conversation memory ---
    history_window: int = 4  # last N (user, assistant) turns kept

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()
