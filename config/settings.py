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
    pdf_path: str = "pdf"  # file OR folder of PDFs

    # --- Chunking ---
    # 400 chars keeps individual numbered clauses in their own chunk so short
    # rules (e.g. same-day journey tiers) don't drown in adjacent rate tables.
    chunk_size: int = 400
    chunk_overlap: int = 75
    chunk_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    # Contextual chunking (Anthropic-style): prepend an LLM-generated
    # topic/provenance header to each chunk before embedding. One Gemini
    # Flash call per chunk during ingestion.
    contextual_chunking_enabled: bool = True

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
    qdrant_url: str = field(default_factory=lambda: os.environ.get("CLUSTER_ENDPOINT", ""))
    qdrant_collection: str = "rag_documents"
    qdrant_summary_collection: str = "rag_document_summaries"
    qdrant_vector_size: int = 768          # text-embedding-004 -> 768 dims

    # --- Document routing ---
    routing_similarity_gap: float = 0.25  # include extra PDFs within this gap of top score

    # --- Retrieval ---
    vector_k: int = 15         # final chunks from vector retriever
    vector_fetch_k: int = 60   # candidates before MMR diversification
    vector_mmr_lambda: float = 0.5  # 0=diversity, 1=relevance
    bm25_k: int = 15

    # --- Reranking + confidence filter ---
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 12             # max chunks kept after rerank; lower = tighter precision
    # Threshold is on sigmoid(logit) so it lives in [0, 1].
    # ms-marco-MiniLM-L-6-v2 underscores HR/policy passages (trained on web search).
    # Keep low so multi-hop "intermediate" chunks (e.g. country→category lookups)
    # aren't filtered before the LLM can use them.
    rerank_score_threshold: float = 0.1

    # --- HYDE ---
    hyde_enabled: bool = True
    hyde_max_tokens: int = 256

    # --- Multi-query retrieval ---
    # Generates alternative phrasings of the query before retrieval.
    # Helps when document vocabulary differs from the user's phrasing.
    multi_query_enabled: bool = True

    # --- Query decomposition ---
    # Breaks multi-hop and multi-part questions into atomic sub-queries so each
    # required fact gets its own retrieval pass. Each sub-query adds ~1 LLM call
    # (HYDE) and one BM25 + vector retrieval pass.
    decomposer_enabled: bool = True

    # --- Routing strict filter ---
    # When True, the global (cross-corpus) retrieval pass only fires when the
    # router is uncertain (selects ALL sources). When the router picks a specific
    # document, retrieval is restricted to that document — preventing foreign-travel
    # chunks from appearing in domestic-travel answers and vice-versa.
    # Set False to restore the original safety-net behaviour (global pass always).
    routing_strict_filter: bool = True

    # --- Conversation memory ---
    history_window: int = 4  # last N (user, assistant) turns kept

    # --- Logging ---
    log_level: str = "INFO"


    def __post_init__(self) -> None:
        if not self.qdrant_url:
            raise EnvironmentError(
                "CLUSTER_ENDPOINT environment variable is not set. "
                "Set it to your Qdrant Cloud cluster URL before running the app."
            )


settings = Settings()
