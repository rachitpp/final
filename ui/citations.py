import html
import re

_CITATION_RE = re.compile(
    r"\(\s*([^()\n]+?\.[A-Za-z0-9]{1,6})\s*,?\s*(?:p\.?|pg\.?|page)\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def chipify_citations(text: str) -> str:
    """Wrap source citations in styled <span class='cite'> chips."""
    def repl(match: "re.Match[str]") -> str:
        filename = match.group(1).strip()
        page = match.group(2)
        safe = filename.replace("_", "&#95;").replace("*", "&#42;")
        attr_doc = html.escape(filename, quote=True)
        return (
            f"<span class='cite' role='button' tabindex='0' "
            f"data-doc=\"{attr_doc}\" data-page=\"{page}\" "
            f"title='View this source in the citation list'>"
            f"{safe} · p.{page}</span>"
        )

    return _CITATION_RE.sub(repl, text)


def sources_html(text: str) -> str:
    """Build the 'N sources' disclosure panel from an answer's citations."""
    pages: "dict[str, set[int]]" = {}
    order: "list[str]" = []
    for match in _CITATION_RE.finditer(text):
        name = match.group(1).strip()
        page = int(match.group(2))
        if name not in pages:
            pages[name] = set()
            order.append(name)
        pages[name].add(page)

    if not order:
        return ""

    total = sum(len(p) for p in pages.values())
    noun = "source" if total == 1 else "sources"

    items = []
    for name in order:
        safe = html.escape(name)
        attr_doc = html.escape(name, quote=True)
        pgs = ", ".join(str(p) for p in sorted(pages[name]))
        items.append(
            f"<li class='source-item' data-doc=\"{attr_doc}\">"
            f"<span class='nm'>{safe}</span> "
            f"<span class='pg'>p. {pgs}</span></li>"
        )

    return (
        "\n\n<div class='sources'>"
        "<button class='sources-toggle' type='button' aria-expanded='false'>"
        "<span class='chev' aria-hidden='true'>&#9654;</span> "
        f"{total} {noun}</button>"
        f"<ul class='sources-list'>{''.join(items)}</ul>"
        "</div>"
    )
