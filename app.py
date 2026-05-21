"""
Streamlit UI for the RAG system.

Wraps the existing RAGPipeline — does not modify backend logic.
Run with:  streamlit run app.py
"""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from pipelines.rag_pipeline import RAGPipeline
from config.settings import settings


# =============================================================
# Page config — must be the first Streamlit call.
# =============================================================
st.set_page_config(
    page_title="RAG System",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =============================================================
# Styling — minimal CSS for typography, spacing, and polish.
# We override only what Streamlit ships by default; we don't
# fight the theme system, so it stays dark/light agnostic.
# =============================================================
CSS = """
<style>
  /* Tighten the default top padding so the app feels app-like, not docs-like. */
  .block-container {
    padding-top: 3rem;
    padding-bottom: 6rem;
    max-width: 760px;
  }

  /* Modern system font stack — looks native on every OS. */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  /* Heading rhythm. */
  h1 { font-weight: 600; letter-spacing: -0.02em; }
  h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

  /* Chat message containers — gentler spacing, softer feel. */
  [data-testid="stChatMessage"] {
    padding: 0.75rem 0;
    border: none;
    background: transparent;
  }
  [data-testid="stChatMessageContent"] {
    line-height: 1.7;
    font-size: 0.97rem;
  }
  [data-testid="stChatMessageContent"] p { margin-bottom: 0.6rem; }

  /* Chat input — slightly more breathing room. */
  [data-testid="stChatInput"] textarea {
    font-size: 0.97rem;
    line-height: 1.5;
  }

  /* Sidebar — quieter, more documentation-like. */
  [data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.12);
  }
  [data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
  }
  [data-testid="stSidebar"] h1 {
    font-size: 1.25rem;
    margin-bottom: 0.25rem;
  }
  [data-testid="stSidebar"] hr {
    margin: 1.5rem 0;
    border-color: rgba(128, 128, 128, 0.12);
  }

  /* Pipeline flow display in sidebar — monospace, subtle. */
  .pipeline-flow {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    color: rgba(128, 128, 128, 0.95);
    padding: 0.75rem 0.9rem;
    border-radius: 8px;
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.1);
  }

  /* Key-value config rows in the sidebar. */
  .config-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.25rem 0;
    color: rgba(128, 128, 128, 0.95);
  }
  .config-row strong {
    color: inherit;
    font-weight: 500;
  }

  /* Subtle muted helper text. */
  .muted {
    color: rgba(128, 128, 128, 0.85);
    font-size: 0.85rem;
    line-height: 1.6;
  }

  /* Hide the default Streamlit chrome we don't need. */
  #MainMenu, footer { visibility: hidden; }
</style>
"""


# =============================================================
# Pipeline init — cached so the heavy objects (vector store,
# BM25 index, cross-encoder, LLM client) load exactly once.
# =============================================================
@st.cache_resource(show_spinner="Loading retrieval pipeline…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# =============================================================
# Session state — chat history for the UI layer.
# The backend keeps its own ConversationMemory (for query
# rewriting). They stay in sync: we clear both on New Chat.
# =============================================================
def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_conversation(pipeline: RAGPipeline) -> None:
    """Wipe UI history and the backend's rewrite memory."""
    st.session_state.messages = []
    pipeline.memory.clear()


# =============================================================
# Sidebar — project info, pipeline overview, config, actions.
# =============================================================
def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.markdown("# RAG System")
        st.markdown(
            "<div class='muted'>"
            "Hybrid retrieval with HYDE, cross-encoder reranking, "
            "and conversational query rewriting."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Pipeline flow ---
        st.markdown("##### Retrieval pipeline")
        st.markdown(
            "<div class='pipeline-flow'>"
            "1. Rewrite query<br>"
            "2. HYDE passage<br>"
            "3. BM25 + Vector (MMR)<br>"
            "4. Cross-encoder rerank<br>"
            "5. Confidence filter<br>"
            "6. Gemini answer"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Current configuration (read-only, transparent) ---
        st.markdown("##### Configuration")
        config_rows = [
            ("Embedding", settings.embedding_model),
            ("LLM", settings.llm_model),
            ("Vector store", "Qdrant Cloud"),
            ("Chunk size", str(settings.chunk_size)),
            ("Vector k / fetch_k", f"{settings.vector_k} / {settings.vector_fetch_k}"),
            ("BM25 k", str(settings.bm25_k)),
            ("Rerank threshold", f"{settings.rerank_score_threshold}"),
            ("HYDE", "on" if settings.hyde_enabled else "off"),
            ("Memory window", f"{settings.history_window} turn(s)"),
        ]
        for label, value in config_rows:
            st.markdown(
                f"<div class='config-row'>"
                f"<span>{label}</span><strong>{value}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Actions ---
        if st.button("New chat", use_container_width=True):
            reset_conversation(pipeline)
            st.rerun()

        # --- Debug (optional) ---
        with st.expander("Debug"):
            turns = pipeline.memory.turns()
            st.markdown(
                f"<div class='muted'>"
                f"Memory: {len(turns)} turn(s) stored "
                f"(max {settings.history_window})."
                f"</div>",
                unsafe_allow_html=True,
            )
            if turns:
                for i, (u, a) in enumerate(turns, 1):
                    st.markdown(
                        f"<div class='muted'><strong>{i}. user</strong> "
                        f"— {u[:80]}{'…' if len(u) > 80 else ''}</div>",
                        unsafe_allow_html=True,
                    )


# =============================================================
# Chat rendering helpers.
# =============================================================
def render_history() -> None:
    """Paint every previous turn from session_state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_empty_state() -> None:
    """Calm landing view when no conversation has started yet."""
    st.markdown(
        "<div style='padding: 3rem 0 1rem 0;'>"
        "<h2 style='margin-bottom: 0.5rem;'>How can I help?</h2>"
        "<p class='muted' style='margin: 0;'>"
        "Ask a question about your indexed documents. "
        "Follow-up questions remember the conversation context."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def handle_user_input(pipeline: RAGPipeline, prompt: str) -> None:
    """Append user message, stream assistant reply, persist both."""
    # 1. Persist + render the user turn.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream the assistant response.
    with st.chat_message("assistant"):
        # st.write_stream consumes the iterator and renders tokens live,
        # returning the concatenated final string when done.
        response = st.write_stream(pipeline.stream_answer(prompt))

    # 3. Persist the assistant turn for re-renders.
    st.session_state.messages.append({"role": "assistant", "content": response})


# =============================================================
# Main.
# =============================================================
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_session()
    pipeline = get_pipeline()

    render_sidebar(pipeline)

    # Empty-state vs conversation view.
    if not st.session_state.messages:
        render_empty_state()
    else:
        render_history()

    # Pinned chat input (Streamlit fixes it to the bottom by default).
    if prompt := st.chat_input("Ask a question…"):
        handle_user_input(pipeline, prompt)


if __name__ == "__main__":
    main()
