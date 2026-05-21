import os
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from config.settings import settings


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Vertex AI text-embedding-004."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=settings.embedding_location,
        vertexai=True,
    )


def get_llm(
    streaming: bool = True,
    max_tokens: int | None = None,
) -> ChatGoogleGenerativeAI:
    """Gemini 2.5 Flash on Vertex AI. Streaming is on by default."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=settings.llm_location,
        vertexai=True,
        temperature=settings.llm_temperature,
        max_output_tokens=max_tokens or settings.llm_max_tokens,
        streaming=streaming,
    )
