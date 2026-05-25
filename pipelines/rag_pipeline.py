from typing import Iterator
from langsmith import traceable

from ingestion.vector_store import load_vector_store, load_summary_store
from retrieval.retrievers import (
    build_bm25_retrievers,
    build_vector_retriever,
    hybrid_retrieve,
)
from retrieval.router import DocumentRouter
from retrieval.reranker import build_cross_encoder, rerank_and_filter
from retrieval.formatter import format_docs
from retrieval.hyde import generate_hyde
from retrieval.rewrite import rewrite_query
from conversation.memory import ConversationMemory
from llm.models import get_llm
from llm.prompts import ANSWER_PROMPT
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

NO_CONTEXT_FALLBACK = (
    "I could not find anything in the documents that confidently answers "
    "this question. Try rephrasing or asking something more specific."
)


class RAGPipeline:
    """
    End-to-end retrieval pipeline. One class, built once, reused.
    Holds the heavy objects (vector store, BM25 indices, router,
    cross-encoder, LLM, memory). Everything else is a plain function.
    """

    def __init__(self) -> None:
        store = load_vector_store()
        self._store = store
        self.bm25_by_source = build_bm25_retrievers(store)
        self.cross_encoder = build_cross_encoder()
        self.llm = get_llm(streaming=True)
        self.memory = ConversationMemory(max_turns=settings.history_window)
        self.router = DocumentRouter(load_summary_store())
        self._last_kept: list = []
        logger.info("RAG pipeline initialized.")

    @traceable(
        name="rag_pipeline.answer",
        metadata={
            "retriever": "router+hyde+bm25+vector-mmr",
            "llm": "gemini-2.5-flash",
            "rerank": "ms-marco-MiniLM-L-6-v2",
        },
    )
    def stream_answer(self, query: str) -> Iterator[str]:
        """
        Run the full flow and yield the answer token-by-token.
        Updates memory after streaming completes.

        Flow:
            rewrite -> route -> HYDE -> hybrid retrieve -> rerank+filter -> LLM
        """
        # 1. Standalone query (uses chat history if any).
        rewritten = rewrite_query(query, self.memory.turns())

        # 2. Route: find which PDF(s) are relevant to this query.
        sources = self.router.route(rewritten)

        # 3. Pick the right BM25 index/indices for the routed sources.
        if len(sources) == 1 and sources[0] in self.bm25_by_source:
            bm25 = self.bm25_by_source[sources[0]]
        else:
            bm25 = self.bm25_by_source["all"]

        # 4. Build a source-filtered vector retriever on the fly (cheap — no I/O).
        vector = build_vector_retriever(self._store, sources=sources)

        # 5. HYDE passage for vector retrieval (NEVER shown to user).
        hyde_doc = (
            generate_hyde(rewritten) if settings.hyde_enabled else rewritten
        )

        # 6. BM25 on the rewritten keywords; vector MMR on the HYDE passage.
        candidates = hybrid_retrieve(
            bm25_query=rewritten,
            vector_query=hyde_doc,
            bm25_retriever=bm25,
            vector_retriever=vector,
        )

        # 7. Cross-encoder rerank + confidence filter.
        kept = rerank_and_filter(rewritten, candidates, self.cross_encoder)
        self._last_kept = kept

        # 8. Fallback when everything is filtered out.
        if not kept:
            logger.info("No chunks passed the confidence threshold.")
            yield NO_CONTEXT_FALLBACK
            self.memory.add(query, NO_CONTEXT_FALLBACK)
            return

        # 9. Stream the answer from Gemini.
        context = format_docs(kept)
        messages = ANSWER_PROMPT.format_messages(
            context=context, question=rewritten
        )

        collected: list[str] = []
        for chunk in self.llm.stream(messages):
            piece = getattr(chunk, "content", None) or ""
            if piece:
                collected.append(piece)
                yield piece

        # 10. Save the turn for future follow-ups.
        self.memory.add(query, "".join(collected))

    def reset(self) -> None:
        """Clear conversation memory for a fresh session."""
        self.memory.clear()

    def last_sources(self) -> list[tuple[str, int | str, str | None]]:
        """
        Unique (source, page, section) tuples from the most recent
        answer, preserving rerank order. Returns [] if the last call
        hit the fallback path.
        """
        seen: set = set()
        out: list[tuple[str, int | str, str | None]] = []
        for d in self._last_kept:
            key = (
                d.metadata.get("source", "unknown"),
                d.metadata.get("page", "?"),
                d.metadata.get("section"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out
