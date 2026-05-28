from typing import List
import math

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def build_cross_encoder() -> HuggingFaceCrossEncoder:
    """Create and return the local cross-encoder model."""
    return HuggingFaceCrossEncoder(model_name=settings.cross_encoder_model)


def sigmoid(score: float) -> float:
    """Convert a raw model logit into a confidence score between 0 and 1."""
    return 1.0 / (1.0 + math.exp(-score))


@traceable(name="rerank_and_filter")
def rerank_and_filter(
    query: str,
    docs: List[Document],
    cross_encoder: HuggingFaceCrossEncoder,
    top_n: int | None = None,
    threshold: float | None = None,
) -> List[Document]:
    """Rerank retrieved documents with a cross-encoder, then keep the strongest matches.

    Prose and tables are handled separately. Prose is filtered by confidence
    threshold within the top_n. Tables are NOT score-thresholded: MiniLM (web-search
    trained) scores markdown grids near-zero regardless of relevance — even the
    exactly-correct rate table lands around 0.003 — so any threshold silently drops
    it, and the score barely orders tables against each other either.

    Since a table's only trustworthy relevance signal is which document it belongs to,
    we order force-kept tables by the relevance of their *source* (the best prose
    score that source earned for this query), then by their own table score, and keep
    the top ``rerank_max_tables``. A table thus rides along with the document the
    prose deemed relevant — so a multi-hop question hitting the foreign-travel policy
    keeps both that policy's tables (rates + country classification) instead of
    spending a slot on an unrelated table from the other policy. This is what carries
    rate/category numbers into multi-hop answers.
    """
    if not docs:
        return []

    top_n = top_n if top_n is not None else settings.rerank_top_n
    threshold = threshold if threshold is not None else settings.rerank_score_threshold
    max_tables = settings.rerank_max_tables

    pairs = [(query, doc.page_content) for doc in docs]
    raw_scores = cross_encoder.score(pairs)

    scored_docs = [(doc, sigmoid(float(score))) for doc, score in zip(docs, raw_scores)]
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    prose = [(d, c) for d, c in scored_docs if not d.metadata.get("is_table")]
    tables = [(d, c) for d, c in scored_docs if d.metadata.get("is_table")]

    filtered_docs = []
    for doc, confidence in prose[:top_n]:
        if confidence >= threshold:
            doc.metadata["rerank_score"] = round(confidence, 4)
            filtered_docs.append(doc)

    # Best prose score per source = how relevant that document is to this query.
    # Tables inherit it so they sort with their document, not by their own (noisy)
    # cross-encoder score. Sources with no surviving prose fall back to 0.0.
    source_relevance: dict[str, float] = {}
    for doc, confidence in prose:
        src = doc.metadata.get("source")
        if confidence > source_relevance.get(src, 0.0):
            source_relevance[src] = confidence

    tables.sort(
        key=lambda item: (
            source_relevance.get(item[0].metadata.get("source"), 0.0),
            item[1],
        ),
        reverse=True,
    )

    n_tables_kept = 0
    for doc, confidence in tables[:max_tables]:
        doc.metadata["rerank_score"] = round(confidence, 4)
        filtered_docs.append(doc)
        n_tables_kept += 1

    logger.debug(
        "Rerank kept %d documents (prose=%d, tables=%d/%d, prose_threshold=%s, max_tables=%s)",
        len(filtered_docs),
        len(filtered_docs) - n_tables_kept,
        n_tables_kept,
        len(tables),
        threshold,
        max_tables,
    )
    return filtered_docs