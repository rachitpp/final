import logging

import streamlit as st

from pipelines.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_conversation(pipeline: RAGPipeline) -> None:
    """Clear the UI conversation and the pipeline's memory."""
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
