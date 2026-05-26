import logging

import streamlit as st

from pipelines.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Tracks whether the pipeline has been built in this process (not just this
# browser session). cache_resource is process-global, so a module global
# mirrors it correctly: we only show the warm-up loader when a build is
# genuinely about to happen, avoiding a 1-frame flash for sessions that join
# after the cache is already warm.
_PIPELINE_BUILT = False


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
