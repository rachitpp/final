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
    """Rerank retrieved documents with a cross-encoder, then keep the strongest matches."""
    if not docs:
        return []

    top_n = top_n if top_n is not None else settings.rerank_top_n
    threshold = threshold if threshold is not None else settings.rerank_score_threshold

    pairs = [(query, doc.page_content) for doc in docs]
    raw_scores = cross_encoder.score(pairs)

    scored_docs = [(doc, sigmoid(float(score))) for doc, score in zip(docs, raw_scores)]
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    filtered_docs = []
    for doc, confidence in scored_docs[:top_n]:
        if confidence >= threshold:
            doc.metadata["rerank_score"] = round(confidence, 4)
            filtered_docs.append(doc)

    logger.debug(
        "Rerank kept %d/%d documents above threshold=%s",
        len(filtered_docs),
        min(top_n, len(scored_docs)),
        threshold,
    )
    return filtered_docs