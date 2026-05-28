from typing import List, Tuple
from langsmith import traceable
from llm.models import get_llm
from llm.prompts import REWRITE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

# Small non-streaming LLM for rewriting; cached.
_llm = None


def _rewrite_llm():
    global _llm
    if _llm is None:
        # thinking_budget=0: see note in retrieval/multi_query.py
        _llm = get_llm(streaming=False, max_tokens=128, thinking_budget=0)
    return _llm


def _format_history(history: List[Tuple[str, str]]) -> str:
    return "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in history
    )


@traceable(name="query_rewrite")
def rewrite_query(query: str, history: List[Tuple[str, str]]) -> str:
    """
    Resolve a follow-up question into a standalone one using the
    last few conversation turns. No-op when there is no history.

    Example:
        history: [("What is attention?", "...mechanism in transformers...")]
        query  : "what are its advantages?"
        ->       "What are the advantages of attention mechanisms?"
    """
    if not history:
        return query
    try:
        messages = REWRITE_PROMPT.format_messages(
            history=_format_history(history),
            question=query,
        )
        rewritten = _rewrite_llm().invoke(messages).content.strip()
        # Reject clearly broken rewrites: must be non-empty, different from
        # the original, and at least half as long (word count). A rewrite
        # shorter than 4 words is almost certainly a truncation or hallucination.
        original_words = len(query.split())
        rewrite_words = len(rewritten.split()) if rewritten else 0
        sane = rewritten and rewrite_words >= max(4, original_words // 2)
        if sane and rewritten.lower() != query.lower():
            logger.info(f"Rewrote: '{query}' -> '{rewritten}'")
            return rewritten
        if rewritten and not sane:
            logger.warning(
                f"Rewrite rejected (too short: {rewrite_words} words): '{rewritten}'"
            )
        return query
    except Exception as e:
        logger.warning(f"Rewrite failed ({e!r}); using original query")
        return query
