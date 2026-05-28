from langsmith import traceable
from llm.models import get_llm
from llm.prompts import HYDE_PROMPT
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache a single non-streaming LLM for HYDE generation.
_llm = None


def _hyde_llm():
    global _llm
    if _llm is None:
        # thinking_budget=0: see note in retrieval/multi_query.py
        _llm = get_llm(
            streaming=False,
            max_tokens=settings.hyde_max_tokens,
            thinking_budget=0,
        )
    return _llm


@traceable(name="hyde_generation")
def generate_hyde(query: str) -> str:
    """
    Generate a hypothetical passage that *would* answer the query.
    We embed THIS passage (richer than the bare question) for vector
    retrieval. The passage is NEVER shown to the user — it's a query.

    On failure, returns the original query so retrieval still works.
    """
    try:
        messages = HYDE_PROMPT.format_messages(question=query)
        passage = _hyde_llm().invoke(messages).content.strip()
        if not passage:
            return query
        # logger.info(f"HYDE produced {len(passage)} chars")
        return passage
    except Exception as e:
        logger.warning(f"HYDE failed ({e!r}); falling back to original query")
        return query
