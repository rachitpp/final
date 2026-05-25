from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Render docs into a single context string with provenance tags.
    The §section segment is included only when the chunk carries a
    section title extracted from the PDF table of contents.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section")
        section_str = f", §{section}" if section else ""
        parts.append(
            f"[Source: {source}, p.{page}{section_str}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)
