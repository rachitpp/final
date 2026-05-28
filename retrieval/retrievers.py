from typing import List
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langsmith import traceable
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _source_filter(sources: list[str] | None) -> Filter | None:
    """Build a Qdrant payload filter restricting results to given sources."""
    if not sources:
        return None
    return Filter(
        must=[FieldCondition(key="metadata.source", match=MatchAny(any=sources))]
    )


def build_vector_retriever(store: QdrantVectorStore, sources: list[str] | None = None):
    """Vector retriever with MMR for diversity, optionally filtered by source."""
    search_kwargs = {
        "k": settings.vector_k,
        "fetch_k": settings.vector_fetch_k,
        "lambda_mult": settings.vector_mmr_lambda,
    }
    f = _source_filter(sources)
    if f:
        search_kwargs["filter"] = f

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def _scroll_all_docs(store: QdrantVectorStore) -> List[Document]:
    """
    Pull every stored chunk back out of Qdrant via scroll().
    QdrantVectorStore stores page_content under `content_payload_key`
    and metadata under `metadata_payload_key` (defaults: 'page_content'
    and 'metadata').
    """
    docs: List[Document] = []
    offset = None
    content_key = store.content_payload_key
    metadata_key = store.metadata_payload_key

    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            docs.append(Document(
                page_content=payload.get(content_key, ""),
                metadata=payload.get(metadata_key, {}) or {},
            ))
        if offset is None:
            break
    return docs


def build_bm25_retrievers(store: QdrantVectorStore) -> dict[str, BM25Retriever]:
    """
    Build one BM25 index per source PDF plus one global index.
    Returns {"all": ..., "MachineLearning.pdf": ..., ...}
    Built once at startup so per-query filtering is free.
    """
    all_docs = _scroll_all_docs(store)

    by_source: dict[str, list[Document]] = {}
    for d in all_docs:
        src = d.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(d)

    retrievers: dict[str, BM25Retriever] = {}
    for src, docs in by_source.items():
        r = BM25Retriever.from_documents(docs)
        r.k = settings.bm25_k
        retrievers[src] = r

    global_r = BM25Retriever.from_documents(all_docs)
    global_r.k = settings.bm25_k
    retrievers["all"] = global_r

    logger.info(f"Built BM25 indices for: {list(by_source.keys())} + global")
    return retrievers


def dedupe_docs(docs: List[Document]) -> List[Document]:
    """Drop duplicates by exact page_content, preserving order."""
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


@traceable(name="hybrid_retrieve")
def hybrid_retrieve(
    bm25_query: str,
    vector_query: str,
    bm25_retriever: BM25Retriever,
    vector_retriever,
) -> List[Document]:
    """
    BM25 uses the (rewritten) user keywords.
    Vector MMR uses the HYDE passage — a richer semantic query.

    We concatenate + dedupe; the cross-encoder does final ranking,
    so we don't need to score-fuse here.
    """
    bm25_docs = bm25_retriever.invoke(bm25_query)
    vec_docs = vector_retriever.invoke(vector_query)
    # logger.info(f"Retrieved BM25={len(bm25_docs)}, Vector={len(vec_docs)}")
    merged = dedupe_docs(bm25_docs + vec_docs)
    # logger.info(f"After dedupe: {len(merged)} candidate(s)")
    return merged
