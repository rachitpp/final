from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@traceable(name="split_documents")
def split_documents(docs: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter with overlap so chunks don't cut
    sentences mid-thought.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunk(s)")
    return chunks
