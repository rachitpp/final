# =============================================================
# Interactive query loop.
# =============================================================
from dotenv import load_dotenv
load_dotenv()

from pipelines.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    print("=" * 64)
    print("RAG System Ready")
    print("Flow: Rewrite → Route → HYDE → Hybrid (BM25+Vector MMR) → Rerank+Filter → Gemini")
    print("Commands: '0' to exit, 'reset' to clear conversation memory")
    print("=" * 64)

    while True:
        try:
            user_query = input("\nUser:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue
        if user_query == "0":
            print("Goodbye!")
            break
        if user_query.lower() == "reset":
            pipeline.memory.clear()
            print("[memory cleared]")
            continue

        print("\nAI Assistant:")
        for piece in pipeline.stream_answer(user_query):
            print(piece, end="", flush=True)
        print()

        sources = pipeline.last_sources()
        if sources:
            print("\nSources:")
            for src, page, section in sources:
                section_str = f", §{section}" if section else ""
                print(f"  - {src}, p.{page}{section_str}")
 

if __name__ == "__main__":
    main()
