import os
import re

import fitz  # PyMuPDF — prose text + TOC/section mapping
import pdfplumber  # table extraction (recovers row/column structure)
from langchain_core.documents import Document
from langsmith import traceable

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chrome / portal-nav detection
# These PDFs are HTML-to-PDF exports from a web portal, so every page carries
# browser chrome (URL, timestamp, login state, nav menu) that must be stripped
# from prose before embedding.
# ---------------------------------------------------------------------------
_CHROME_MARKERS = (
    "policy-details.php",
    "Page ",
    "Logged in as",
    "Logout",
    "Change Password",
    "Welcome",
    "http://",
    "https://",
)
_TIMESTAMP_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d+:\d+\s*[AP]M\s*$")
_BULLET_NAV_RE = re.compile(r"^[•·▪▸►]\s*\S")  # e.g. "• Policy", "• Annexure"


def _is_chrome_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(m in s for m in _CHROME_MARKERS):
        return True
    if _TIMESTAMP_RE.match(s):
        return True
    if _BULLET_NAV_RE.match(s):
        return True
    return False


def _clean_prose(text: str) -> str:
    """Remove browser/portal chrome lines from HTML-to-PDF prose text."""
    lines = [ln for ln in text.splitlines() if not _is_chrome_line(ln)]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Table extraction (pdfplumber)
# ---------------------------------------------------------------------------

def _clean_cell(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _looks_like_chrome(cells: list[str]) -> bool:
    joined = " ".join(cells)
    return any(m in joined for m in _CHROME_MARKERS)


def _table_to_markdown(rows: list[list[str | None]]) -> str | None:
    """
    Render a pdfplumber table as a markdown grid. Returns None for tables
    too sparse or chrome-only to be useful.
    """
    cleaned: list[list[str]] = []
    for raw in rows:
        cells = [_clean_cell(c) for c in raw]
        if not any(cells) or _looks_like_chrome(cells):
            continue
        cleaned.append(cells)

    if len(cleaned) < 2:
        return None

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    non_empty_cols = sum(1 for c in range(width) if any(r[c] for r in cleaned))
    if non_empty_cols < 2:
        return None

    header = cleaned[0]
    if not any(header):
        header = [f"Col{i+1}" for i in range(width)]
    sep = ["---"] * width
    body = cleaned[1:]

    def fmt(r: list[str]) -> str:
        return "| " + " | ".join(c or "" for c in r) + " |"

    return "\n".join([fmt(header), fmt(sep), *[fmt(r) for r in body]])


def _extract_tables(path: str) -> tuple[dict[int, list[str]], dict[int, list[tuple]]]:
    """
    Extract tables from all pages.

    Returns
    -------
    tables_by_page  : {1-based page number: [markdown_table, ...]}
    bboxes_by_page  : {1-based page number: [(x0, top, x1, bottom), ...]}
                      Bounding boxes in pdfplumber page coordinates
                      (origin = top-left, y increases downward) — same
                      system as PyMuPDF, so boxes are directly comparable.
    """
    tables_by_page: dict[int, list[str]] = {}
    bboxes_by_page: dict[int, list[tuple]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                rendered: list[str] = []
                bboxes: list[tuple] = []
                for table in page.find_tables():
                    md = _table_to_markdown(table.extract())
                    if md:
                        rendered.append(md)
                        bboxes.append(table.bbox)
                if rendered:
                    tables_by_page[i] = rendered
                    bboxes_by_page[i] = bboxes
    except Exception as e:
        logger.warning(f"Table extraction failed for '{path}': {e!r}")
    return tables_by_page, bboxes_by_page


# ---------------------------------------------------------------------------
# Prose extraction (PyMuPDF) — table regions excluded
# ---------------------------------------------------------------------------

def _block_overlaps_table(
    bx0: float, by0: float, bx1: float, by1: float,
    table_bboxes: list[tuple],
    threshold: float = 0.4,
) -> bool:
    """
    True if more than `threshold` (40 %) of this text block's area
    falls inside any table bounding box.  Threshold < 1.0 tolerates
    slight bbox differences between pdfplumber and PyMuPDF.
    """
    b_area = max((bx1 - bx0) * (by1 - by0), 1.0)
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ox = max(0.0, min(bx1, tx1) - max(bx0, tx0))
        oy = max(0.0, min(by1, ty1) - max(by0, ty0))
        if ox * oy / b_area >= threshold:
            return True
    return False


def _prose_without_tables(fitz_page, table_bboxes: list[tuple]) -> str:
    """
    Extract text from a PyMuPDF page, skipping any text blocks that sit
    inside a pdfplumber-detected table region.  This prevents the same
    table data appearing twice: once as a clean TABLE chunk (pdfplumber)
    and once as linearized garbage inside a prose chunk (PyMuPDF).
    """
    if not table_bboxes:
        return fitz_page.get_text("text").strip()

    parts: list[str] = []
    for block in fitz_page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, block_type = block
        if block_type != 0:          # skip image blocks
            continue
        if _block_overlaps_table(bx0, by0, bx1, by1, table_bboxes):
            continue
        parts.append(text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Per-file loader
# ---------------------------------------------------------------------------

def _load_one(path: str) -> list[Document]:
    """
    Emit Documents for one PDF:
      • one prose Document per non-empty page  (table regions excluded)
      • one extra Document per detected table   (clean markdown grid, kept
        whole so chunking never splits a table apart)
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.warning(f"Skipping '{path}': {e!r}")
        return []

    source = os.path.basename(path)
    toc = [(p, t) for _l, t, p in doc.get_toc() if p > 0]
    tables_by_page, bboxes_by_page = _extract_tables(path)

    out: list[Document] = []
    for i, page in enumerate(doc, start=1):
        section = next((t for p, t in reversed(toc) if p <= i), None)
        table_bboxes = bboxes_by_page.get(i, [])

        # Prose: PyMuPDF text with table blocks removed, then chrome stripped.
        raw_text = _prose_without_tables(page, table_bboxes)
        text = _clean_prose(raw_text)
        if text:
            out.append(Document(
                page_content=text,
                metadata={"source": source, "page": i, "section": section},
            ))

        # Tables as their own units so they embed as a clean grid rather than
        # a linearized number soup, and are never split mid-row by the chunker.
        for md in tables_by_page.get(i, []):
            header = f"Table from {source}, page {i}"
            if section:
                header += f" (section: {section})"
            out.append(Document(
                page_content=f"{header}:\n{md}",
                metadata={
                    "source": source,
                    "page": i,
                    "section": section,
                    "is_table": True,
                },
            ))

    doc.close()
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="load_documents")
def load_documents(path: str) -> list[Document]:
    """Load a single PDF or every PDF in a directory."""
    if os.path.isdir(path):
        paths = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(".pdf")
        )
    else:
        paths = [path]

    docs: list[Document] = []
    for p in paths:
        docs.extend(_load_one(p))

    n_tables = sum(1 for d in docs if d.metadata.get("is_table"))
    logger.info(
        f"Loaded {len(docs)} document(s) from '{path}' "
        f"({n_tables} table block(s))"
    )
    return docs
