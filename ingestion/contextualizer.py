from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langsmith import traceable

from config.settings import settings
from llm.models import get_llm
from llm.prompts import CONTEXTUALIZE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

# Cost note: one Gemini Flash call per chunk. A 150-page book yields
# ~300 chunks → ~300 calls. Acceptable as a one-time ingestion cost;
# never run this on a query path.

_MAX_WORKERS = 8
_llm = None


def _ctx_llm():
    global _llm
    if _llm is None:
        # Short, deterministic output. ~3 sentences cap.
        # thinking_budget=0: without it, Gemini 2.5 Flash spends ~140 of the
        # 160-token budget on hidden thinking and emits 5-word truncated
        # headers ("This table on page 1 of" with no rest of the sentence).
        _llm = get_llm(streaming=False, max_tokens=160, thinking_budget=0)
    return _llm


def _contextualize_one(chunk: Document) -> Document:
    """Return a copy of `chunk` with a context header prepended."""
    try:
        messages = CONTEXTUALIZE_PROMPT.format_messages(
            source=chunk.metadata.get("source", "unknown"),
            section=chunk.metadata.get("section") or "n/a",
            chunk=chunk.page_content,
        )
        header = _ctx_llm().invoke(messages).content.strip()
        if not header:
            return chunk
        return Document(
            page_content=f"{header}\n\n{chunk.page_content}",
            metadata=chunk.metadata,
        )
    except Exception as e:
        logger.warning(f"Contextualization failed ({e!r}); keeping original chunk")
        return chunk


@traceable(name="contextualize_chunks")
def contextualize_chunks(chunks: list[Document]) -> list[Document]:
    """Fan out across a thread pool; preserve input order."""
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        out = list(pool.map(_contextualize_one, chunks))
    logger.info(f"Contextualized {len(out)} chunk(s)")
    return out
