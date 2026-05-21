import os

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langsmith import traceable

from utils.logger import get_logger

logger = get_logger(__name__)


def _load_one(path: str) -> list[Document]:
    """Open a PDF; emit one Document per non-empty page with TOC section."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.warning(f"Skipping '{path}': {e!r}")
        return []
    source = os.path.basename(path)
    # toc rows: [level, title, page]. Linear scan is fine — TOCs are tiny.
    toc = [(p, t) for _l, t, p in doc.get_toc() if p > 0]
    out: list[Document] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        section = next((t for p, t in reversed(toc) if p <= i), None)
        out.append(Document(
            page_content=text,
            metadata={"source": source, "page": i, "section": section},
        ))
    doc.close()
    return out


@traceable(name="load_documents")
def load_documents(path: str) -> list[Document]:
    """Load a single PDF or every PDF in a directory."""
    if os.path.isdir(path):
        paths = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(".pdf"))
    else:
        paths = [path]
    docs: list[Document] = []
    for p in paths:
        docs.extend(_load_one(p))
    logger.info(f"Loaded {len(docs)} page(s) from '{path}'")
    return docs
