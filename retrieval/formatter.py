from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Render docs into a single context string with provenance tags.
    Includes the rerank score when available so the LLM (and reader)
    can see which chunks were most confident. The `§section` segment
    is included only when the chunk carries a section title.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section")
        score = doc.metadata.get("rerank_score")
        section_str = f", §{section}" if section else ""
        score_str = f" | score={score}" if score is not None else ""
        parts.append(
            f"[Chunk {i} | {source}, p.{page}{section_str}{score_str}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)
