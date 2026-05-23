from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Render docs into a single context string with provenance tags.
    Includes the rerank score when available so the LLM (and reader)
    can see which chunks were most confident. The `§section` segment
    is included only when the chunk carries a section title.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Source: {source}, p.{page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)
