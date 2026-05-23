"""
Chainlit UI for the RAG system.

Run with:  chainlit run app_chainlit.py -w

This is a drop-in alternative to app.py (Streamlit). It imports the same
RAGPipeline and calls the same public methods — stream_answer() and
last_sources() — so the pipeline code is untouched.

Install once:  pip install chainlit
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import asyncio
import queue
import threading

import chainlit as cl
from pipelines.rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Pipeline is built ONCE per process on first chat, then reused.
# Lazy init means a network hiccup at startup never crashes Chainlit —
# the error surfaces as a chat message instead.
# ---------------------------------------------------------------------------
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Starter prompts — these replace the Streamlit "suggested questions" cards
# and the welcome empty-state. They render as clickable chips on a fresh chat.
# ---------------------------------------------------------------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Overview of main topics",
            message="Give me an overview of the main topics covered.",
        ),
        cl.Starter(
            label="Key concepts to understand",
            message="What are the key concepts I should understand?",
        ),
        cl.Starter(
            label="Most important findings",
            message="Summarize the most important findings.",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _get_pipeline)  # warm up on first chat
    _get_pipeline().memory.clear()


# ---------------------------------------------------------------------------
# Bridge: run the SYNC generator stream_answer() in a worker thread and feed
# tokens to the async side through a queue. Keeps Chainlit's event loop free
# so tokens actually flush to the UI as they arrive.
# ---------------------------------------------------------------------------
async def _stream_tokens(query: str):
    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def _run():
        try:
            for token in _get_pipeline().stream_answer(query):
                q.put(token)
        except Exception as exc:  # surface pipeline errors to the UI
            q.put(exc)
        finally:
            q.put(SENTINEL)

    threading.Thread(target=_run, daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _format_sources() -> str | None:
    """Turn last_sources() into a compact markdown block, or None if empty."""
    rows = _get_pipeline().last_sources()
    if not rows:
        return None

    lines = ["**Sources**", ""]
    for source, page, _section in rows:
        lines.append(f"- `{source}`  ·  p. {page}")
    return "\n".join(lines)


@cl.on_message
async def on_message(message: cl.Message):
    answer = cl.Message(content="")
    await answer.send()

    try:
        async for token in _stream_tokens(message.content):
            await answer.stream_token(token)
    except Exception as exc:
        answer.content = f"Something went wrong while answering: `{exc}`"
        await answer.update()
        return

    await answer.update()

    # Surface grounding — this is what the Streamlit app never showed.
    sources = _format_sources()
    if sources:
        await cl.Message(content=sources, author="Retrieval").send()