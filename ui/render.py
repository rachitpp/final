import html
import logging
import time

import streamlit as st
import streamlit.components.v1 as components

from pipelines.rag_pipeline import RAGPipeline
from ui.constants import ASSISTANT_ACTIONS_HTML, ERROR_HTML, THINKING_HTML
from ui.citations import chipify_citations, sources_html

logger = logging.getLogger(__name__)

# Repaint at most this often while streaming (~20fps cap).
_STREAM_REPAINT_INTERVAL = 0.05


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
    """Render a user turn as a right-aligned bubble.

    Uses plain st.markdown instead of st.chat_message so the layout stays
    bullet-proof across Streamlit versions that rename internal testids.
    """
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div class='user-row'>"
        f"<div class='user-label'>◉&nbsp;&nbsp;You</div>"
        f"<div class='user-bubble'>{safe}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _snap_to_new_question() -> None:
    """Explicit belt-and-suspenders trigger for the snap-to-question scroll.

    The persistent MutationObserver in install_client_behaviors already
    detects new .user-row elements via count growth, so this is redundant
    in the happy path. Emitting it right after render_user_message gives a
    second deterministic trigger that doesn't depend on the observer timing.
    The snap function dedupes via activeSnapRow, so double-fires are no-ops.
    """
    components.html(
        """
        <script>
        (function () {
          const win = window.parent;
          function go() {
            if (typeof win.__ragSnapToLatest === 'function') {
              win.__ragSnapToLatest();
            } else {
              setTimeout(go, 60);
            }
          }
          go();
        })();
        </script>
        """,
        height=0,
    )


def _safe_partial_markdown(text: str) -> str:
    """Close any unclosed code fence so partial renders don't break layout."""
    if text.count("```") % 2 == 1:
        return text + "\n```"
    return text


def stream_with_indicator(pipeline: RAGPipeline, prompt: str) -> str:
    """Stream the answer behind a thinking indicator.

    Returns the full answer text, or "" on failure.
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
    _snap_to_new_question()

    with st.chat_message("assistant"):
        response = stream_with_indicator(pipeline, prompt)

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
