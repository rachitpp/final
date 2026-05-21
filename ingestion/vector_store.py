import os
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from config.settings import settings
from llm.models import get_embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)


def _make_client() -> QdrantClient:
    """Connect to Qdrant Cloud using URL and API key from env."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


def _ensure_collection(client: QdrantClient) -> None:
    """Create the collection with cosine distance if it doesn't exist."""
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"Created Qdrant collection '{settings.qdrant_collection}' "
            f"(size={settings.qdrant_vector_size}, distance=cosine)"
        )
    # Always ensure the source index exists — idempotent, safe to call every run.
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )


@traceable(name="create_vector_store")
def create_vector_store(chunks: list[Document]) -> QdrantVectorStore:
    """
    Embed chunks in batches (Vertex caps at ~250/call) and
    persist to Qdrant Cloud using cosine similarity.
    """
    client = _make_client()
    _ensure_collection(client)

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embedding_model(),
    )

    batch_size = settings.embedding_batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1} ({len(batch)} chunk(s))"
        )
        store.add_documents(batch)

    logger.info(
        f"Persisted {len(chunks)} chunk(s) to collection "
        f"'{settings.qdrant_collection}'"
    )
    return store


def load_vector_store() -> QdrantVectorStore:
    """Connect to the existing Qdrant Cloud collection."""
    client = _make_client()
    # Ensure the source index exists on startup so filtering works even
    # on collections created before routing was introduced.
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embedding_model(),
    )


def create_summary_store(summaries: dict[str, str]) -> QdrantVectorStore:
    """
    Embed one summary per PDF and persist to a separate Qdrant collection.
    Always recreates the collection so re-ingestion stays in sync.
    """
    client = _make_client()

    if client.collection_exists(settings.qdrant_summary_collection):
        client.delete_collection(settings.qdrant_summary_collection)

    client.create_collection(
        collection_name=settings.qdrant_summary_collection,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_size,
            distance=Distance.COSINE,
        ),
    )

    docs = [
        Document(page_content=summary, metadata={"source": source})
        for source, summary in summaries.items()
    ]

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_summary_collection,
        embedding=get_embedding_model(),
    )
    store.add_documents(docs)
    logger.info(f"Stored {len(docs)} document summary/summaries in '{settings.qdrant_summary_collection}'")
    return store


def load_summary_store() -> QdrantVectorStore:
    """Connect to the existing document summaries collection."""
    client = _make_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_summary_collection,
        embedding=get_embedding_model(),
    )
