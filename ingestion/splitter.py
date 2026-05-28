import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches clause IDs like "1.", "1.1", "1.1a", "1.2b" at the start of a line.
_CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?[a-z]?)\s", re.MULTILINE)


def _extract_clause_id(text: str) -> str | None:
    """Return the first clause number found in a chunk, e.g. '1.1a'."""
    m = _CLAUSE_RE.search(text)
    return m.group(1) if m else None


@traceable(name="split_documents")
def split_documents(docs: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter with overlap so chunks don't cut
    sentences mid-thought. Table documents bypass the splitter so their
    markdown grid is never broken mid-row.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
    )
    tables = [d for d in docs if d.metadata.get("is_table")]
    prose = [d for d in docs if not d.metadata.get("is_table")]
    chunks = splitter.split_documents(prose) + tables

    # Enrich every chunk with the first clause ID found in its text so callers
    # can filter by clause number (e.g. "1.1b") when needed.
    for chunk in chunks:
        if not chunk.metadata.get("clause_id"):
            clause = _extract_clause_id(chunk.page_content)
            if clause:
                chunk.metadata["clause_id"] = clause

    logger.info(f"Created {len(chunks)} chunk(s) ({len(tables)} table block(s) kept whole)")
    return chunks
