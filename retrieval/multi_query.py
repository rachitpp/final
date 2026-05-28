from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from llm.models import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You generate alternative BM25 search queries for retrieving passages from HR travel policy documents.

Rules:
- Output exactly 3 alternative queries, one per line.
- Use DIFFERENT vocabulary than the original — think about how a formal HR/policy document would phrase the answer.
- Expand abbreviations: BA → "boarding allowance", DA → "daily allowance", LA → "lodging allowance", TA → "travel allowance", HRA → "house rent allowance".
- Replace informal words with formal policy terms (e.g. "removed" → "not admissible", "not payable", "shall not be paid").
- If the query mentions a band or city category, include variants using the formal tier names (Category A / B / C, Band 7/8/9/10).
- Keep each query under 15 words.
- Output ONLY the 3 queries, no numbering, no extra text."""),
    ("human", "Original query: {query}\n\nAlternative queries:"),
])

_llm = None


def _llm_instance():
    global _llm
    if _llm is None:
        # thinking_budget=0 disables Gemini 2.5 Flash's hidden thinking tokens,
        # which would otherwise eat the max_output_tokens budget and truncate
        # the visible variants mid-word.
        _llm = get_llm(streaming=False, max_tokens=400, thinking_budget=0)
    return _llm


@traceable(name="multi_query_generation")
def generate_query_variants(query: str) -> List[str]:
    """
    Return the original query plus up to 3 LLM-generated reformulations.
    On failure, returns just the original so retrieval still works.
    """
    try:
        messages = _PROMPT.format_messages(query=query)
        output = _llm_instance().invoke(messages).content.strip()
        variants = [line.strip() for line in output.splitlines() if line.strip()][:3]
        all_queries = [query] + variants
        logger.info("Multi-query variants: %s", all_queries)
        return all_queries
    except Exception as e:
        logger.warning("Multi-query generation failed (%r); using original query only", e)
        return [query]
