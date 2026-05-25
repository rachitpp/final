"""
Streamlit UI for the RAG system.
Run with:  streamlit run app.py

Styling lives in ``styles/`` partials — this file is logic only.
"""
import html
import re
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
import streamlit.components.v1 as components
from pipelines.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Repaint at most this often while streaming, so long answers don't
# re-render their whole body on every single token (~20fps cap).
_STREAM_REPAINT_INTERVAL = 0.05

# Tracks whether the pipeline has been built *in this process* (not just
# this browser session). cache_resource is process-global, so a module
# global mirrors it correctly: we only show the warm-up loader when a
# build is genuinely about to happen, avoiding a 1-frame flash for
# sessions that join after the cache is already warm.
_PIPELINE_BUILT = False


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="expanded",
)


_COPY_ICON = (
    "<svg viewBox='0 0 24 24' width='14' height='14' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round' aria-hidden='true'>"
    "<rect x='9' y='9' width='13' height='13' rx='2'></rect>"
    "<path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'></path>"
    "</svg>"
)

# Appended to the final assistant render (not mid-stream). Rendered as part
# of Streamlit's own markdown so React keeps it across reruns; the click is
# handled by a single delegated listener in install_client_behaviors().
ASSISTANT_ACTIONS_HTML = (
    "\n\n<div class='msg-actions'>"
    "<button class='copy-btn' type='button' aria-label='Copy answer'>"
    f"{_COPY_ICON}<span class='copy-label'>Copy</span>"
    "</button>"
    "</div>"
)

THINKING_HTML = (
    "<div class='thinking'>"
    "<span class='thinking-glyph' aria-hidden='true'>\u25d0</span>"
    "<span class='thinking-stages'>"
    "<span class='stage stage-1'>Embedding query</span>"
    "<span class='stage stage-2'>Retrieving passages</span>"
    "<span class='stage stage-3'>Reranking results</span>"
    "<span class='stage stage-4'>Writing answer</span>"
    "</span>"
    "</div>"
)

ERROR_HTML = (
    "<div class='error-note'>"
    "<span class='error-mark'>!</span>"
    "<span>Something went wrong fetching that answer. "
    "Please try again in a moment.</span>"
    "</div>"
)

# Matches inline source citations like "(MachineLearning.pdf, p.7)",
# "(notes.docx p. 12)", "(data.csv, page 3)" — a filename with a short
# extension followed by a page reference, inside parentheses.
_CITATION_RE = re.compile(
    r"\(\s*([^()\n]+?\.[A-Za-z0-9]{1,6})\s*,?\s*(?:p\.?|pg\.?|page)\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def chipify_citations(text: str) -> str:
    """Wrap source citations in styled <span class='cite'> chips.

    Applied only to the final answer (not mid-stream) so partial text
    can't produce broken markup. Escapes markdown-sensitive characters
    in the filename so e.g. under_scores don't render as emphasis.

    Each chip carries ``data-doc`` and ``data-page`` so the client-side
    handler can match it to the corresponding entry in the sources list
    and pulse-highlight that entry on click.
    """
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
    """Build the subtle 'N sources' disclosure from an answer's citations.

    Reuses the same citation regex as the chips, so the list always matches
    what's cited in the text. Documents are deduped, their pages collected
    and sorted, and the whole thing rendered as a quiet toggle + indented
    list. Returns "" when the answer cites nothing, so unsourced answers get
    no disclosure at all. The toggle is wired by the delegated listener in
    install_client_behaviors().
    """
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


def load_css() -> None:
    """Inject stylesheets from styles/ partials, concatenated in order."""
    styles_dir = Path(__file__).parent / "styles"
    partials = [
        "_tokens.css",
        "_streamlit-chrome.css",
        "_base.css",
        "_sidebar.css",
        "_welcome.css",
        "_messages.css",
        "_chat-input.css",
        "_loading.css",
        "_responsive.css",
    ]
    chunks: list[str] = []
    for name in partials:
        path = styles_dir / name
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("CSS partial not found: %s — skipping", path)
    if not chunks:
        logger.warning("No CSS partials found in %s — running unstyled", styles_dir)
        return
    st.markdown(f"<style>{''.join(chunks)}</style>", unsafe_allow_html=True)


def install_client_behaviors() -> None:
    """Install all client-side niceties in one place, via one helper iframe.

    Injected JS only runs inside a Streamlit helper iframe, so we host
    everything in a single zero-height one (collapsed out of the layout via
    CSS). A flag on the parent document keeps everything installed exactly
    once, so Streamlit re-running this on every interaction never stacks
    duplicate listeners. The behaviors:

      • Auto-scroll — keep the newest message in view while it streams, but
        only while the user is already at the bottom; pause when they scroll
        up to re-read, resume when they return (the courtesy ChatGPT extends).
      • Jump-to-latest — a floating button that fades in when the user has
        scrolled away from the bottom; clicking it snaps back and re-pins.
        It reads the composer's real position so it centers on the message
        column and respects the sidebar offset, no hardcoded geometry.
      • Auto-focus — focus the question box once on first load only, so the
        user can type immediately without ever stealing focus on a rerun.
      • Copy — one delegated click handler copies an assistant answer's text
        (citations included; the button itself excluded). Because the button
        is part of Streamlit's rendered markdown, it survives every rerun.
    """
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const win = window.parent;
          // Version the guard so that when this script changes (e.g. a new
          // handler is added), an existing page reinstalls instead of being
          // blocked by a stale "already installed" flag. Bump on changes.
          const VERSION = 4;
          if (doc.__ragClientVersion === VERSION) return;  // already current
          doc.__ragClientVersion = VERSION;

          const NEAR = 140;                          // px tolerance for "at bottom"

          // The scroll container varies across Streamlit versions: the
          // main <section> in some, the document itself in others.
          function scroller() {
            return doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stMain"]')
                || doc.scrollingElement
                || doc.documentElement;
          }
          function atBottom(el) {
            if (!el) return true;
            return el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR;
          }
          function toBottom() {
            const el = scroller();
            if (el) el.scrollTop = el.scrollHeight;
          }

          /* ---------- jump-to-latest button ---------- */
          const jump = doc.createElement('button');
          jump.type = 'button';
          jump.className = 'jump-latest';
          jump.setAttribute('aria-label', 'Jump to latest');
          jump.innerHTML =
              '<svg viewBox="0 0 24 24" width="16" height="16" fill="none"'
            + ' stroke="currentColor" stroke-width="2" stroke-linecap="round"'
            + ' stroke-linejoin="round"><path d="M12 5v14"></path>'
            + '<path d="M19 12l-7 7-7-7"></path></svg>';
          doc.body.appendChild(jump);

          // Center the button on the composer (so it tracks the message
          // column / sidebar offset) and sit it just above the input.
          function placeJump() {
            const pill = doc.querySelector('[data-testid="stChatInput"]')
                      || doc.querySelector('[data-testid="stBottom"]');
            if (!pill) return;
            const r = pill.getBoundingClientRect();
            jump.style.left = (r.left + r.width / 2) + 'px';
            jump.style.bottom = (win.innerHeight - r.top + 10) + 'px';
          }
          jump.addEventListener('click', function () {
            pinned = true;
            toBottom();
            updateJump();
          });

          let pinned = true;
          function updateJump() {
            placeJump();
            jump.classList.toggle('is-visible', !pinned);
          }

          // Coalesce scroll events into one layout read/write per frame.
          let sQueued = false;
          function onScroll() {
            if (sQueued) return;
            sQueued = true;
            win.requestAnimationFrame(function () {
              sQueued = false;
              pinned = atBottom(scroller());
              updateJump();
            });
          }
          [scroller(), win].forEach(function (t) {
            if (t) t.addEventListener('scroll', onScroll, { passive: true });
          });
          win.addEventListener('resize', placeJump);

          /* ---------- follow the stream while pinned ---------- */
          let mQueued = false;
          const obs = new MutationObserver(function () {
            if (mQueued) return;
            mQueued = true;
            win.requestAnimationFrame(function () {
              mQueued = false;
              if (pinned) toBottom();
              updateJump();
            });
          });
          obs.observe(doc.body, { childList: true, subtree: true, characterData: true });

          /* ---------- copy an assistant answer (delegated) ---------- */
          doc.addEventListener('click', function (e) {
            const btn = e.target.closest && e.target.closest('.copy-btn');
            if (!btn) return;
            const content = btn.closest('[data-testid="stChatMessageContent"]')
                         || btn.closest('[data-testid="stChatMessage"]');
            if (!content) return;
            const clone = content.cloneNode(true);
            clone.querySelectorAll('.msg-actions').forEach(function (n) { n.remove(); });
            const text = (clone.innerText || '').trim();

            function flash() {
              btn.classList.add('copied');
              const label = btn.querySelector('.copy-label');
              const prev = label ? label.textContent : '';
              if (label) label.textContent = 'Copied';
              setTimeout(function () {
                btn.classList.remove('copied');
                if (label) label.textContent = prev || 'Copy';
              }, 1400);
            }
            if (win.navigator.clipboard && win.navigator.clipboard.writeText) {
              win.navigator.clipboard.writeText(text).then(flash).catch(function () {});
            } else {
              // Fallback for non-secure contexts without the async API.
              const ta = doc.createElement('textarea');
              ta.value = text;
              ta.style.position = 'fixed';
              ta.style.opacity = '0';
              doc.body.appendChild(ta);
              ta.select();
              try { doc.execCommand('copy'); flash(); } catch (err) {}
              doc.body.removeChild(ta);
            }
          }, true);

          /* ---------- toggle a sources disclosure (delegated) ---------- */
          doc.addEventListener('click', function (e) {
            const t = e.target.closest && e.target.closest('.sources-toggle');
            if (!t) return;
            const box = t.closest('.sources');
            if (!box) return;
            const open = box.classList.toggle('open');
            t.setAttribute('aria-expanded', open ? 'true' : 'false');
          }, true);

          /* ---------- citation chip click: open sources + highlight match ----
             Citations are the killer feature of a RAG UI — making them
             interactive (instead of decorative) is what turns the answer
             into something the user can verify. This handler opens the
             sources panel for the answer the chip belongs to, then pulses
             the matching source item so the user can see where it came
             from. A future enhancement would slide in the actual passage. */
          doc.addEventListener('click', function (e) {
            const chip = e.target.closest && e.target.closest('.cite');
            if (!chip) return;
            const docname = chip.dataset.doc;
            if (!docname) return;
            const root = chip.closest('[data-testid="stChatMessageContent"]')
                      || chip.closest('[data-testid="stChatMessage"]')
                      || doc;
            const box = root.querySelector('.sources');
            if (!box) return;
            // Open the panel if it isn't already.
            if (!box.classList.contains('open')) {
              box.classList.add('open');
              const tog = box.querySelector('.sources-toggle');
              if (tog) tog.setAttribute('aria-expanded', 'true');
            }
            // Pulse the matching source row. Small delay so the open
            // animation has started before the pulse lands.
            const items = box.querySelectorAll('.source-item');
            items.forEach(function (it) { it.classList.remove('is-pulsing'); });
            setTimeout(function () {
              let matched = null;
              items.forEach(function (it) {
                if (!matched && it.dataset.doc === docname) matched = it;
              });
              if (matched) {
                matched.classList.add('is-pulsing');
                setTimeout(function () {
                  matched.classList.remove('is-pulsing');
                }, 1800);
              }
            }, 320);
          }, true);
          // Keyboard parity: Enter / Space on a focused chip triggers click.
          doc.addEventListener('keydown', function (e) {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const chip = e.target.closest && e.target.closest('.cite');
            if (!chip) return;
            e.preventDefault();
            chip.click();
          });

          /* ---------- focus the question box once on first load ---------- */
          (function focusOnce(n) {
            const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
            if (ta) { ta.focus(); }
            else if (n > 0) { setTimeout(function () { focusOnce(n - 1); }, 150); }
          })(20);

          placeJump();
          toBottom();
        })();
        </script>
        """,
        height=0,
    )


@st.cache_resource(show_spinner=False)
def _build_pipeline() -> RAGPipeline:
    global _PIPELINE_BUILT
    pipeline = RAGPipeline()
    _PIPELINE_BUILT = True
    return pipeline


def get_pipeline() -> RAGPipeline:
    """Build the pipeline behind a custom centered loading state.

    Only shows the loader when a real build is about to occur. If the
    process-global cache is already warm, returns instantly with no flash.
    """
    if _PIPELINE_BUILT:
        return _build_pipeline()

    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class='loader-wrap'>
          <div class='loader-card'>
            <div class='loader-dots' aria-hidden='true'>
              <span></span><span></span><span></span>
            </div>
            <div class='loader-title'>Preparing your assistant</div>
            <div class='loader-sub'>Indexing memory and warming the retriever.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pipeline = _build_pipeline()
    placeholder.empty()
    return pipeline


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_conversation(pipeline: RAGPipeline) -> None:
    """Clear the UI conversation and the pipeline's memory.

    Prefers a pipeline-owned ``reset()`` so the UI doesn't depend on how
    memory is stored; falls back to ``memory.clear()`` and never crashes
    if neither exists.
    """
    st.session_state.messages = []

    reset = getattr(pipeline, "reset", None)
    if callable(reset):
        reset()
        return

    memory = getattr(pipeline, "memory", None)
    clear = getattr(memory, "clear", None)
    if callable(clear):
        clear()
    else:
        logger.warning("Pipeline exposes no reset() or memory.clear(); memory not cleared")


def _list_indexed_documents(pipeline: RAGPipeline) -> "tuple[list[dict], int]":
    """Introspect what's in the index for the sidebar Library section.

    Document names come from ``pipeline.bm25_by_source`` — a dict keyed by
    source filename with a synthetic ``"all"`` key excluded. Page counts are
    read from each per-source BM25 retriever's ``docs`` list (already in
    memory from startup) by counting distinct ``metadata['page']`` values.
    No extra Qdrant calls needed.

    Returns ``(documents, total_pages)``. Each document is a dict with
    ``name`` (str) and optionally ``pages`` (int). Returns ``([], 0)``
    if nothing is found — the sidebar renders a quiet empty placeholder.
    """
    bm25 = getattr(pipeline, "bm25_by_source", None)
    if not isinstance(bm25, dict):
        return [], 0

    names = sorted(k for k in bm25.keys() if k and k != "all")
    if not names:
        return [], 0

    normalized: "list[dict]" = []
    total_pages = 0
    for name in names:
        retriever = bm25.get(name)
        page_count = None
        docs = getattr(retriever, "docs", None)
        if isinstance(docs, list):
            pages = {
                d.metadata.get("page")
                for d in docs
                if d.metadata.get("page") is not None
            }
            page_count = len(pages) or None
        normalized.append({"name": name, "pages": page_count})
        if page_count:
            total_pages += page_count
    return normalized, total_pages


def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'>RAG Assistant</div>"
            "<div class='sidebar-sub'>"
            "Ask questions across your indexed documents."
            "</div>"
            "<div class='sidebar-rule'></div>",
            unsafe_allow_html=True,
        )
        if st.button("New chat", use_container_width=True, type="secondary"):
            reset_conversation(pipeline)
            st.rerun()

        # --- Library section --------------------------------------------------
        # Surfaces what corpus the assistant is grounded in. The single
        # biggest "what is this product?" signal in the whole UI — without
        # it, a first-time user can't tell the app apart from generic chat.
        docs, total_pages = _list_indexed_documents(pipeline)
        if docs:
            doc_count = len(docs)
            doc_noun = "document" if doc_count == 1 else "documents"
            page_meta = f" · {total_pages:,} pages" if total_pages else ""
            items_html: "list[str]" = []
            for d in docs[:40]:  # cap so a huge corpus doesn't fill the sidebar
                safe_name = html.escape(d["name"])
                pages = d.get("pages")
                meta_html = (
                    f"<span class='lib-meta'>{pages} pp.</span>"
                    if isinstance(pages, int)
                    else ""
                )
                items_html.append(
                    f"<li class='lib-doc' title='{safe_name}'>"
                    f"<span class='lib-name'>{safe_name}</span>{meta_html}</li>"
                )
            overflow_html = (
                f"<li class='lib-more'>+ {len(docs) - 40} more</li>"
                if len(docs) > 40
                else ""
            )
            st.markdown(
                "<div class='library'>"
                "<div class='lib-head'>"
                "<span class='lib-label'>Library</span>"
                f"<span class='lib-count'>{doc_count} {doc_noun}{page_meta}</span>"
                "</div>"
                f"<ul class='lib-list'>{''.join(items_html)}{overflow_html}</ul>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            # Quiet placeholder — nothing detected but the section is still
            # visible so it's obvious what *would* live here once wired.
            st.markdown(
                "<div class='library library--empty'>"
                "<div class='lib-head'>"
                "<span class='lib-label'>Library</span>"
                "<span class='lib-count'>No documents detected</span>"
                "</div>"
                "<div class='lib-empty-note'>"
                "Add documents to your index, then reload — they\u2019ll "
                "appear here automatically."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_empty_state() -> None:
    st.markdown(
        "<div class='welcome-wrap'>"
        "<div class='eyebrow'>Ready when you are</div>"
        "<div class='welcome-title'>What would you like to know?</div>"
        "<div class='welcome-sub'>"
        "Ask anything about your documents."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_user_message(text: str) -> None:
    """Render a user turn as a self-contained right-aligned bubble.

    We intentionally do NOT use st.chat_message here. Streamlit keeps
    renaming the internal avatar `data-testid`s between releases, which
    silently breaks the `:has(...)`-based right-alignment. Owning the
    markup makes the alignment bullet-proof across versions.
    """
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div class='user-row'>"
        f"<div class='user-label'>◉&nbsp;&nbsp;You</div>"
        f"<div class='user-bubble'>{safe}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _safe_partial_markdown(text: str) -> str:
    """Make in-progress markdown safe to render each frame.

    A stream can pause mid-code-block, leaving an unclosed ``` fence that
    would swallow the rest of the answer's formatting. If the number of
    fences is odd, append a temporary closing fence so the partial renders
    cleanly; the real final render (which has balanced fences) replaces it.
    """
    if text.count("```") % 2 == 1:
        return text + "\n```"
    return text


def stream_with_indicator(pipeline: RAGPipeline, prompt: str) -> str:
    """Stream the answer behind a 'Searching documents…' indicator.

    Paints a thinking state immediately, then swaps it for the answer as
    soon as the first token arrives, accumulating chunks so the text
    streams in live. Each repaint now renders markdown progressively (so
    headings, lists, bold and code format as they arrive instead of
    snapping in at the end), time-throttled to stay smooth. Any failure in
    the pipeline is caught and shown inline instead of crashing the turn.

    Returns the full answer text, or "" if nothing was produced / it failed.
    """
    placeholder = st.empty()
    placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

    chunks: list[str] = []
    last_paint = 0.0
    try:
        for chunk in pipeline.stream_answer(prompt):
            chunks.append(str(chunk))
            now = time.monotonic()
            if now - last_paint > _STREAM_REPAINT_INTERVAL:
                # Progressive markdown render + a thin streaming caret. No
                # unsafe_allow_html here: citations are chipped only in the
                # final paint, and partial HTML could render half-formed.
                partial = _safe_partial_markdown("".join(chunks))
                placeholder.markdown(partial + " ▌")
                last_paint = now
    except Exception:
        logger.exception("stream_answer failed for prompt=%r", prompt)
        placeholder.markdown(ERROR_HTML, unsafe_allow_html=True)
        return ""

    full = "".join(chunks).strip()
    if not full:
        logger.info("stream_answer produced no content for prompt=%r", prompt)
        placeholder.markdown(ERROR_HTML, unsafe_allow_html=True)
        return ""

    # Final clean render — drops the caret, parses markdown, turns source
    # citations into styled chips, appends the sources disclosure, then the
    # copy affordance.
    placeholder.markdown(
        chipify_citations(full) + sources_html(full) + ASSISTANT_ACTIONS_HTML,
        unsafe_allow_html=True,
    )
    return full


def render_history() -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    chipify_citations(msg["content"])
                    + sources_html(msg["content"])
                    + ASSISTANT_ACTIONS_HTML,
                    unsafe_allow_html=True,
                )


def handle_user_input(pipeline: RAGPipeline, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_user_message(prompt)

    with st.chat_message("assistant"):
        response = stream_with_indicator(pipeline, prompt)

    # Don't persist empty / failed turns — they'd re-render forever.
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    load_css()
    install_client_behaviors()
    init_session()
    pipeline = get_pipeline()

    render_sidebar(pipeline)

    # Read the prompt FIRST so we can decide which view to render. Without
    # this, on the rerun where the very first question is submitted, the
    # script sees `messages` still empty at the if-check below and renders
    # the welcome state — then handle_user_input() appends + renders the
    # user/assistant turns *underneath*, leaving the welcome stranded
    # above an active conversation. By reading prompt up front, we know to
    # skip the welcome on that rerun.
    prompt = st.chat_input("Ask a question…")

    if not st.session_state.messages and not prompt:
        render_empty_state()
    else:
        render_history()

    if prompt:
        handle_user_input(pipeline, prompt)


if __name__ == "__main__":
    main()
