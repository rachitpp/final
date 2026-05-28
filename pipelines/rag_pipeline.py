from typing import Iterator, List
from langchain_core.documents import Document
from langsmith import traceable

from ingestion.vector_store import load_vector_store, load_summary_store
from retrieval.retrievers import (
    build_bm25_retrievers,
    build_vector_retriever,
    hybrid_retrieve,
    dedupe_docs,
)
from retrieval.router import DocumentRouter
from retrieval.reranker import build_cross_encoder, rerank_and_filter
from retrieval.formatter import format_docs
from retrieval.hyde import generate_hyde
from retrieval.multi_query import generate_query_variants
from retrieval.decomposer import decompose_query
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
        # Real source PDFs (the "all" key is the global BM25 index, not a doc).
        self._sources_all = sorted(self.bm25_by_source.keys() - {"all"})
        logger.info("RAG pipeline initialized.")

    def _retrieve_candidates(
        self,
        bm25_query: str,
        vector_query: str,
        sources: list[str] | None,
    ) -> List[Document]:
        """
        Hybrid retrieval (BM25 + vector MMR) for a given source scope.
        `sources=None` searches the whole corpus (global BM25 + unfiltered
        vector). No reranking here — the caller reranks the merged pool.
        """
        if sources and len(sources) == 1 and sources[0] in self.bm25_by_source:
            bm25 = self.bm25_by_source[sources[0]]
        else:
            bm25 = self.bm25_by_source["all"]

        vector = build_vector_retriever(self._store, sources=sources)
        return hybrid_retrieve(
            bm25_query=bm25_query,
            vector_query=vector_query,
            bm25_retriever=bm25,
            vector_retriever=vector,
        )

    def _should_do_global_pass(self, routed_sources: list[str]) -> bool:
        """
        Decide whether to run an additional global (cross-corpus) retrieval pass.

        With routing_strict_filter=True (default for multi-doc policy corpora):
          - Only do the global pass when the router is uncertain, i.e. it selected
            ALL available sources. A confident routing decision scopes all retrieval
            to the selected document(s), preventing cross-domain contamination.

        With routing_strict_filter=False (original safety-net behaviour):
          - Always do the global pass when routing picked a subset, so a wrong
            routing decision can still surface the correct chunk.
        """
        if len(self._sources_all) <= 1:
            return False  # single-doc corpus: global == routed, skip
        if settings.routing_strict_filter:
            # Global pass only when router couldn't pick a subset (uncertain).
            return set(routed_sources) == set(self._sources_all)
        else:
            # Original: global pass whenever routing is selective.
            return set(routed_sources) != set(self._sources_all)

    @traceable(
        name="rag_pipeline.answer",
        metadata={
            "retriever": "router+decompose+hyde+bm25+vector-mmr",
            "llm": "gemini-2.5-flash",
            "rerank": "ms-marco-MiniLM-L-6-v2",
        },
    )
    def stream_answer(self, query: str) -> Iterator[str]:
        """
        Run the full flow and yield the answer token-by-token.
        Updates memory after streaming completes.

        Flow:
            rewrite → decompose → route → per-sub-query HYDE + hybrid retrieve
            → (optional global pass) → multi-query BM25 → rerank+filter → LLM
        """
        # 1. Resolve follow-up pronouns into a standalone question.
        rewritten = rewrite_query(query, self.memory.turns())

        # 2. Route to the most relevant source document(s).
        sources = self.router.route(rewritten)

        # 3. Decompose multi-hop / multi-part questions into atomic sub-queries.
        #    Single-hop questions come back as [rewritten] unchanged.
        sub_queries: List[str] = (
            decompose_query(rewritten) if settings.decomposer_enabled else [rewritten]
        )

        # 4. For each sub-query: generate HYDE (primary only, cost control) and
        #    run hybrid BM25 + vector retrieval scoped to the routed source(s).
        candidates: List[Document] = []
        for i, sq in enumerate(sub_queries):
            vec_query = (
                generate_hyde(sq) if (settings.hyde_enabled and i == 0) else sq
            )
            candidates += self._retrieve_candidates(sq, vec_query, sources)

        # 5. Optional global pass — retrieves across the full corpus.
        #    Controlled by routing_strict_filter (see _should_do_global_pass).
        if self._should_do_global_pass(sources):
            for i, sq in enumerate(sub_queries):
                vec_query = (
                    generate_hyde(sq) if (settings.hyde_enabled and i == 0) else sq
                )
                candidates += self._retrieve_candidates(sq, vec_query, None)

        candidates = dedupe_docs(candidates)

        # 6. Multi-query: extra BM25 passes with abbreviation-expanded / formally-
        #    phrased variants. Only patches keyword gaps; semantics covered by HYDE.
        #    Respects the source filter: when strict routing is on and the router
        #    picked a single document, use that document's BM25 index — not "all" —
        #    so multi-query can't smuggle in chunks from other-domain documents.
        if settings.multi_query_enabled:
            if (
                settings.routing_strict_filter
                and len(sources) == 1
                and sources[0] in self.bm25_by_source
            ):
                mq_bm25 = self.bm25_by_source[sources[0]]
            else:
                mq_bm25 = self.bm25_by_source["all"]
            for variant in generate_query_variants(rewritten)[1:]:  # skip original
                for doc in mq_bm25.invoke(variant):
                    candidates.append(doc)
            candidates = dedupe_docs(candidates)
            logger.info("After multi-query BM25: %d candidates", len(candidates))

        # 7. Cross-encoder rerank + confidence filter across the merged pool.
        #    Always scored against the full (rewritten) question so the model
        #    weighs each chunk against the complete information need.
        kept = rerank_and_filter(rewritten, candidates, self.cross_encoder)
        self._last_kept = kept

        # 8. Fallback when everything is filtered out.
        if not kept:
            logger.info("No chunks passed the confidence threshold.")
            yield NO_CONTEXT_FALLBACK
            self.memory.add(rewritten, NO_CONTEXT_FALLBACK)
            return

        # 9. Stream the answer from Gemini.
        context = format_docs(kept)
        messages = ANSWER_PROMPT.format_messages(
            context=context, question=rewritten
        )

        collected: list[str] = []
        for chunk in self.llm.stream(messages):
            # Gemini 2.5 Flash (thinking enabled) delivers chunk.content as
            # either a plain string or a list of content-part dicts.
            # Normalize to str so we never yield a list object.
            raw = getattr(chunk, "content", None)
            if isinstance(raw, str):
                piece = raw
            elif isinstance(raw, list):
                piece = "".join(
                    p.get("text", "") for p in raw
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                piece = ""
            if piece:
                collected.append(piece)
                yield piece

        # 10. Save the standalone (rewritten) question so future follow-ups
        #     resolve pronouns against the unambiguous version.
        self.memory.add(rewritten, "".join(collected))

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
