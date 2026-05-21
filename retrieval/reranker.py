import math
from typing import List, Tuple
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langsmith import traceable
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def build_cross_encoder() -> HuggingFaceCrossEncoder:
    """Small local cross-encoder. No API key required."""
    return HuggingFaceCrossEncoder(model_name=settings.cross_encoder_model)


def _sigmoid(x: float) -> float:
    """Map raw cross-encoder logits into a [0, 1] confidence score."""
    return 1.0 / (1.0 + math.exp(-x))


@traceable(name="rerank_and_filter")
def rerank_and_filter(
    query: str,
    docs: List[Document],
    cross_encoder: HuggingFaceCrossEncoder,
    top_n: int | None = None,
    threshold: float | None = None,
) -> List[Document]:
    """
    Score every candidate against the (rewritten) query, sort descending,
    keep at most top_n, then DROP anything below the confidence threshold.

    Attaches `rerank_score` to each kept doc's metadata so the prompt can
    optionally cite it.

    Returns [] when nothing clears the bar — the caller is expected to
    emit a graceful fallback.
    """
    if not docs:
        return []

    top_n = top_n or settings.rerank_top_n
    threshold = settings.rerank_score_threshold if threshold is None else threshold

    pairs = [(query, d.page_content) for d in docs]
    raw_scores = cross_encoder.score(pairs)
    scored: List[Tuple[Document, float]] = sorted(
        [(d, _sigmoid(float(s))) for d, s in zip(docs, raw_scores)],
        key=lambda x: x[1],
        reverse=True,
    )

    top = scored[:top_n]
    kept = [(d, s) for d, s in top if s >= threshold]

    for d, s in top:
        verdict = "KEEP" if s >= threshold else "DROP"
        preview = d.page_content[:300].replace("\n", " ")
        logger.info(f"  [{verdict}] score={s:.3f}  {preview!r}")
    logger.info(
        f"Rerank kept {len(kept)}/{len(top)} above threshold={threshold}"
    )

    for d, s in kept:
        d.metadata["rerank_score"] = round(s, 4)

    return [d for d, _ in kept]
