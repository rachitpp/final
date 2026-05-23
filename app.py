"""
Streamlit UI for the RAG system.
Run with:  streamlit run app.py
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import streamlit as st
from pipelines.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  /* ============================================================
     Tokens — one place, used everywhere
     ============================================================ */
  :root {
    --ink:        #0f1115;   /* primary text */
    --ink-soft:   #2a2d34;   /* secondary text */
    --ink-muted:  #6b6f78;   /* tertiary / helper */
    --ink-faint:  #9a9ea6;   /* placeholder, eyebrow */

    --paper:      #fdfcf9;   /* main background — warm off-white */
    --paper-2:    #f4f2ec;   /* sidebar */
    --paper-3:    #efece4;   /* hover fill */

    --rule:       #e6e3da;   /* hairlines */
    --rule-strong:#cfcbbf;

    --focus:      #0f1115;   /* focus ring color */
    --focus-halo: rgba(15, 17, 21, 0.06);

    --radius-sm:  8px;
    --radius-md:  12px;
    --radius-lg:  16px;

    --serif: "Charter", "Iowan Old Style", "Source Serif Pro",
             Georgia, "Times New Roman", serif;
    --sans:  -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
             Roboto, Helvetica, Arial, sans-serif;
  }

  /* ============================================================
     Hide Streamlit chrome
     ============================================================ */
  #MainMenu, footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stHeader"] .stDeployButton,
  [data-testid="stStatusWidget"] {
    display: none !important;
  }
  [data-testid="stHeader"] {
    background: transparent;
    height: 0;
  }

  /* ============================================================
     Base
     ============================================================ */
  .stApp { background: var(--paper); }

  html, body, [class*="css"], .stMarkdown, .stMarkdown p {
    font-family: var(--serif);
    -webkit-font-smoothing: antialiased;
    color: var(--ink);
  }
  button, input, textarea,
  [data-testid="stChatInput"] textarea,
  .ui-sans, .eyebrow, .muted, .sidebar-sub, .suggest-label {
    font-family: var(--sans) !important;
  }

  .block-container {
    padding-top: 2.5rem;
    padding-bottom: 9rem;
    max-width: 740px;
  }

  /* ============================================================
     Sidebar
     ============================================================ */
  [data-testid="stSidebar"] {
    background: var(--paper-2);
    border-right: 1px solid var(--rule);
  }
  [data-testid="stSidebar"] .block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
  }

  .sidebar-brand {
    font-family: var(--serif);
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin-bottom: 0.4rem;
  }
  .sidebar-sub {
    font-size: 0.82rem;
    color: var(--ink-muted);
    line-height: 1.55;
    margin-bottom: 1.5rem;
  }
  .sidebar-rule {
    height: 1px;
    background: var(--rule);
    margin: 0.25rem 0 1.25rem;
  }

  [data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: var(--ink);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.55rem 0.9rem;
    transition: border-color 140ms ease, background 140ms ease,
                transform 140ms ease;
    box-shadow: none;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--ink-soft);
    background: var(--paper-3);
    color: var(--ink);
  }
  [data-testid="stSidebar"] .stButton > button:focus,
  [data-testid="stSidebar"] .stButton > button:active {
    border-color: var(--ink) !important;
    background: var(--paper-3) !important;
    color: var(--ink) !important;
    box-shadow: 0 0 0 3px var(--focus-halo) !important;
    outline: none !important;
  }

  /* ============================================================
     Welcome / empty state
     ============================================================ */
  .welcome-wrap { padding: 4.5rem 0 1.5rem; }

  .eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 1.25rem;
  }

  .welcome-title {
    font-family: var(--serif);
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.08;
    color: var(--ink);
    margin: 0 0 0.85rem;
    text-wrap: pretty;
  }

  .welcome-sub {
    font-family: var(--serif);
    font-size: 1.05rem;
    color: var(--ink-soft);
    line-height: 1.6;
    max-width: 30rem;
    margin: 0 0 2.5rem;
    text-wrap: pretty;
  }

  /* ---- Suggestions ---- */
  .suggest-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 0.6rem 0.05rem;
  }

  /* Main-area suggestion buttons — quiet, list-like with hairline */
  .block-container .stButton > button {
    background: transparent;
    color: var(--ink-soft);
    border: none;
    border-top: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.95rem 0.25rem;
    font-family: var(--serif);
    font-size: 1rem;
    font-weight: 400;
    text-align: left;
    justify-content: flex-start;
    line-height: 1.5;
    transition: color 140ms ease, padding-left 180ms ease,
                background 140ms ease;
    box-shadow: none;
  }
  .block-container .stButton > button > div { justify-content: flex-start; }
  .block-container .stButton > button:last-of-type {
    border-bottom: 1px solid var(--rule);
  }
  .block-container .stButton > button:hover {
    color: var(--ink);
    background: transparent;
    padding-left: 0.65rem;
  }
  .block-container .stButton > button:focus,
  .block-container .stButton > button:active {
    color: var(--ink) !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    border-color: var(--rule) !important;
  }

  /* ============================================================
     Chat messages — base
     ============================================================ */
  [data-testid="stChatMessage"] {
    display: flex;
    flex-direction: column;
    padding: 0.6rem 0;
    border: none;
    background: transparent;
    gap: 0.28rem;
    padding-left: 0 !important;
  }
  [data-testid="stChatMessageContent"] {
    line-height: 1.78;
    font-size: 1.06rem;
    color: var(--ink);
    margin-left: 0 !important;
  }
  [data-testid="stChatMessageContent"] p { margin-bottom: 0.65rem; color: var(--ink); }
  [data-testid="stChatMessageContent"] strong { color: var(--ink); }
  [data-testid="stChatMessageContent"] code {
    background: var(--paper-3);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.92em;
  }

  /* Hide default avatars — selectors cover current + older Streamlit */
  [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
  [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
  [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
  [data-testid="stChatMessage"] > img:first-child,
  [data-testid="stChatMessage"] > div:first-child:has(svg) {
    display: none !important;
  }

  /* ============================================================
     User message — right-aligned bubble
     ============================================================ */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    align-items: flex-end;
    padding: 0.5rem 0 !important;
    margin-bottom: 0.2rem;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before,
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])::before {
    content: "◉  You";
    font-family: var(--sans);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-muted);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--paper-3);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.15rem !important;
    width: fit-content !important;
    max-width: 72% !important;
    align-self: flex-end !important;
    font-family: var(--sans) !important;
    font-size: 1.06rem !important;
    color: var(--ink-soft) !important;
    line-height: 1.6 !important;
  }

  /* ============================================================
     Assistant message — left-aligned, full-width content
     ============================================================ */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    align-items: flex-start;
    padding: 0.85rem 0 0.6rem !important;
    border-top: 1px solid var(--rule);
    margin-top: 0.1rem;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])::before,
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])::before {
    content: "◐  Assistant";
    font-family: var(--sans);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-soft);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    font-size: 1.06rem !important;
    line-height: 1.78 !important;
    max-width: 100%;
  }

  /* ============================================================
     Chat input — kill Streamlit's red, replace with ink focus
     ============================================================ */
  [data-testid="stChatInput"] { background: transparent; }
  [data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid var(--rule-strong) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 1px 0 rgba(15, 17, 21, 0.02) !important;
    transition: border-color 140ms ease, box-shadow 140ms ease;
  }
  [data-testid="stChatInput"] > div:hover {
    border-color: var(--ink-soft) !important;
  }
  [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--focus) !important;
    box-shadow: 0 0 0 3px var(--focus-halo) !important;
    outline: none !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--ink) !important;
    font-size: 0.98rem !important;
    caret-color: var(--ink);
  }
  [data-testid="stChatInput"] textarea::placeholder { color: var(--ink-faint) !important; }
  [data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
    outline: none !important;
    border: none !important;
  }
  [data-testid="stChatInput"] button {
    color: var(--ink) !important;
    background: transparent !important;
    transition: transform 140ms ease, color 140ms ease;
  }
  [data-testid="stChatInput"] button:hover {
    color: #000 !important;
    transform: translateY(-1px);
  }
  [data-testid="stChatInput"] button svg { fill: var(--ink); }

  /* ============================================================
     Misc
     ============================================================ */
  .muted {
    color: var(--ink-muted);
    font-size: 0.84rem;
    line-height: 1.6;
  }

  /* ============================================================
     Loading state — custom, centered, matches the design system
     ============================================================ */
  .loader-wrap {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--paper);
    z-index: 9999;
    animation: loader-fade-in 240ms ease both;
  }
  .loader-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.1rem;
    padding: 2rem 2.25rem;
    max-width: 22rem;
    text-align: center;
  }
  .loader-dots {
    display: inline-flex;
    gap: 0.45rem;
    align-items: center;
    height: 14px;
    margin-bottom: 0.4rem;
  }
  .loader-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ink);
    opacity: 0.25;
    animation: loader-bounce 1.1s ease-in-out infinite;
  }
  .loader-dots span:nth-child(2) { animation-delay: 0.15s; }
  .loader-dots span:nth-child(3) { animation-delay: 0.30s; }

  .loader-title {
    font-family: var(--serif);
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    color: var(--ink);
    line-height: 1.2;
  }
  .loader-sub {
    font-family: var(--sans);
    font-size: 0.85rem;
    color: var(--ink-muted);
    line-height: 1.55;
    max-width: 20rem;
  }

  @keyframes loader-bounce {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40%           { opacity: 1;    transform: translateY(-4px); }
  }
  @keyframes loader-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  /* ============================================================
     Bottom bar — kill every possible dark wrapper Streamlit uses
     ============================================================ */
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div,
  [data-testid="stBottom"] > div > div,
  [data-testid="stBottomBlockContainer"],
  .stChatFloatingInputContainer,
  .stChatFloatingInputContainer > div,
  .stChatFloatingInputContainer > div > div {
    background: var(--paper) !important;
    background-color: var(--paper) !important;
    border-top: none !important;
    box-shadow: none !important;
  }

  /* ============================================================
     Responsive — collapse extra padding on small screens
     ============================================================ */
  @media (max-width: 640px) {
    .block-container { padding-top: 1.5rem; padding-bottom: 7rem; }
    .welcome-wrap { padding: 2rem 0 1rem; }
    .welcome-title { font-size: 2rem; }
    .welcome-sub { font-size: 1rem; }
  }
</style>
"""

SUGGESTED_QUESTIONS = [
    "Give me an overview of the main topics covered.",
    "What are the key concepts I should understand?",
    "Summarize the most important findings.",
]


@st.cache_resource(show_spinner=False)
def _build_pipeline() -> RAGPipeline:
    return RAGPipeline()


def get_pipeline() -> RAGPipeline:
    """Build the pipeline behind a custom centered loading state."""
    # If already cached, skip the loader entirely.
    if "_pipeline_ready" in st.session_state:
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
    st.session_state["_pipeline_ready"] = True
    return pipeline


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def reset_conversation(pipeline: RAGPipeline) -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    pipeline.memory.clear()


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



def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_input(pipeline: RAGPipeline, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(pipeline.stream_answer(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
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