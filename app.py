"""
Streamlit UI for the RAG system.
Run with:  streamlit run app.py

Styling lives in the sibling ``styles.css`` — this file is logic only.
"""
import html
import re
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
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


SUGGESTED_QUESTIONS = [
    "Give me an overview of the main topics covered.",
    "What are the key concepts I should understand?",
    "Summarize the most important findings.",
]

THINKING_HTML = (
    "<div class='thinking'>"
    "<span class='thinking-dots'><span></span><span></span><span></span></span>"
    "<span class='thinking-text'>Searching documents…</span>"
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
    """
    def repl(match: "re.Match[str]") -> str:
        filename = match.group(1).strip()
        page = match.group(2)
        safe = filename.replace("_", "&#95;").replace("*", "&#42;")
        return f"<span class='cite'>{safe} · p.{page}</span>"

    return _CITATION_RE.sub(repl, text)


def load_css() -> None:
    """Inject the external stylesheet. Degrades gracefully if missing."""
    css_path = Path(__file__).parent / "styles.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("styles.css not found at %s — running unstyled", css_path)
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def reset_conversation(pipeline: RAGPipeline) -> None:
    """Clear the UI conversation and the pipeline's memory.

    Prefers a pipeline-owned ``reset()`` so the UI doesn't depend on how
    memory is stored; falls back to ``memory.clear()`` and never crashes
    if neither exists.
    """
    st.session_state.messages = []
    st.session_state.pending_prompt = None

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


def render_empty_state() -> None:
    st.markdown(
        "<div class='welcome-wrap'>"
        "<div class='eyebrow'>Ready when you are</div>"
        "<div class='welcome-title'>What would you like to know?</div>"
        "<div class='welcome-sub'>"
        "Ask anything about your documents. Follow-up questions keep the "
        "conversation in context."
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


def stream_with_indicator(pipeline: RAGPipeline, prompt: str) -> str:
    """Stream the answer behind a 'Searching documents…' indicator.

    Paints a thinking state immediately, then swaps it for the answer as
    soon as the first token arrives, accumulating chunks so the text
    streams in live. Repaints are time-throttled, and any failure in the
    pipeline is caught and shown inline instead of crashing the turn.

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
                placeholder.markdown("".join(chunks) + " ▌")  # blinking caret
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

    # Final clean render — drops the caret, parses markdown, and turns
    # source citations into styled chips.
    placeholder.markdown(chipify_citations(full), unsafe_allow_html=True)
    return full


def render_history() -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chipify_citations(msg["content"]), unsafe_allow_html=True)


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
    init_session()
    pipeline = get_pipeline()

    render_sidebar(pipeline)

    if not st.session_state.messages:
        render_empty_state()
    else:
        render_history()

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        handle_user_input(pipeline, prompt)

    if prompt := st.chat_input("Ask a question…"):
        handle_user_input(pipeline, prompt)


if __name__ == "__main__":
    main()