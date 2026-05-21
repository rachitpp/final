from collections import defaultdict

from langchain_core.documents import Document
from langsmith import traceable

from llm.models import get_llm
from llm.prompts import SUMMARIZE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_CHARS = 8000  # first N chars of each PDF used as summary input

_llm = None


def _sum_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(streaming=False, max_tokens=512)
    return _llm


@traceable(name="summarize_documents")
def summarize_documents(docs: list[Document]) -> dict[str, str]:
    """
    Generate one plain-text summary per unique source PDF.
    Returns {filename: summary_text}.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        grouped[d.metadata.get("source", "unknown")].append(d.page_content)

    summaries: dict[str, str] = {}
    for source, pages in grouped.items():
        sample = "\n\n".join(pages)[:_SAMPLE_CHARS]
        messages = SUMMARIZE_PROMPT.format_messages(source=source, content=sample)
        summary = _sum_llm().invoke(messages).content.strip()
        summaries[source] = summary
        logger.info(f"Summarized '{source}' ({len(summary)} chars)")

    return summaries
