"""
Streamlit UI for the RAG system.
Run with:  streamlit run app.py

UI logic is split across ui/ modules — this file is the entry point only.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="expanded",
)

from ui.client_behaviors import load_css, install_client_behaviors
from ui.pipeline import get_pipeline
from ui.session import init_session
from ui.sidebar import render_sidebar
from ui.render import render_empty_state, render_history, handle_user_input


def main() -> None:
    load_css()
    install_client_behaviors()
    init_session()
    pipeline = get_pipeline()

    render_sidebar(pipeline)

    # Read the prompt before rendering history so we know whether to show
    # the welcome state or jump straight into the conversation on first submit.
    prompt = st.chat_input("Ask a question…")

    if not st.session_state.messages and not prompt:
        render_empty_state()
    else:
        render_history()

    if prompt:
        handle_user_input(pipeline, prompt)


if __name__ == "__main__":
    main()
