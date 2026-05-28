from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from llm.models import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You decompose complex questions into atomic retrieval sub-queries for HR travel policy documents.

Rules:
- If the question is already atomic and single-hop, output it unchanged on one line.
- For multi-part questions ("what is X and what about Y?"), produce one sub-query per distinct part.
- For multi-hop questions that require an intermediate lookup before the final fact, produce:
    1. One sub-query for the intermediate lookup (e.g. "What country category is Japan classified under?")
    2. One sub-query for the final rate/rule using the expected result (e.g. "What is Band 9 DA for category A countries?")
- Examples of multi-hop questions in travel policy:
    "What DA does Band 9 get in Japan?" →
        "What category is Japan classified under for foreign travel?"
        "What is the daily allowance for Band 9 in category A countries?"
    "What lodging and boarding does a Band 8 employee get in Mumbai?" →
        "What city category is Mumbai?"
        "What is the Band 8 lodging entitlement with bills in a category A city?"
        "What is the Band 8 boarding allowance in a category A city?"
- Output one sub-query per line. Maximum 4 sub-queries.
- Output ONLY the sub-queries, no numbering, no preamble, no extra text."""),
    ("human", "Question: {question}\n\nSub-queries:"),
])

_llm = None


def _decomposer_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(streaming=False, max_tokens=200)
    return _llm


@traceable(name="query_decomposition")
def decompose_query(question: str) -> List[str]:
    """
    Break a complex question into atomic sub-queries for retrieval.
    Returns a list with the original question if the question is already
    atomic, or 2-4 sub-queries for multi-hop / multi-part questions.
    On LLM failure, returns [question] so retrieval still works.
    """
    try:
        messages = _PROMPT.format_messages(question=question)
        output = _decomposer_llm().invoke(messages).content.strip()
        sub_queries = [line.strip() for line in output.splitlines() if line.strip()][:4]
        if len(sub_queries) <= 1:
            return [question]
        logger.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)
        return sub_queries
    except Exception as e:
        logger.warning("Decomposition failed (%r); using original question", e)
        return [question]
