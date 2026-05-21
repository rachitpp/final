from langchain_qdrant import QdrantVectorStore
from langsmith import traceable

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRouter:
    """
    Routes a query to the most relevant PDF(s) by comparing its embedding
    against stored document summaries.

    Returns a list of source filenames. If multiple PDFs score within
    `routing_similarity_gap` of the best match, all are included — this
    handles cases where the query is genuinely cross-document.
    """

    def __init__(self, summary_store: QdrantVectorStore) -> None:
        self._store = summary_store
        # Cache the full source list for the fallback path using scroll
        # (no embedding needed — just reading payload metadata).
        points, _ = summary_store.client.scroll(
            collection_name=summary_store.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        self._all_sources: list[str] = [
            p.payload.get(summary_store.metadata_payload_key, {}).get("source", "unknown")
            for p in points
        ]

    @traceable(name="document_router")
    def route(self, query: str) -> list[str]:
        """Return the source filename(s) most relevant to this query."""
        n = max(len(self._all_sources), 1)
        results = self._store.similarity_search_with_score(query, k=n)

        if not results:
            logger.info("Router: no summaries found, searching all sources")
            return self._all_sources

        best_score = results[0][1]
        cutoff = best_score - settings.routing_similarity_gap

        selected = [
            doc.metadata["source"]
            for doc, score in results
            if score >= cutoff
        ]

        logger.info(f"Router: query routed to {selected} (best={best_score:.3f})")
        return selected
