"""
Prints every stored chunk for a given source PDF.
Use this to inspect what the model actually receives as context.

Usage:
    python eval/inspect_chunks.py "foreign (1).pdf"
    python eval/inspect_chunks.py "domestic travel.pdf"
    python eval/inspect_chunks.py "foreign (1).pdf" --tables   # only table chunks
    python eval/inspect_chunks.py "foreign (1).pdf" --page 1   # only page 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from ingestion.vector_store import load_vector_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="PDF filename as stored (e.g. 'foreign (1).pdf')")
    parser.add_argument("--tables", action="store_true", help="Show only table chunks")
    parser.add_argument("--page", type=int, default=None, help="Filter to a specific page number")
    args = parser.parse_args()

    store = load_vector_store()

    # Scroll all points, filter by source in Python (simpler than Qdrant filter syntax)
    all_docs = []
    offset = None
    content_key = store.content_payload_key
    metadata_key = store.metadata_payload_key

    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            meta = payload.get(metadata_key, {}) or {}
            if meta.get("source") == args.source:
                all_docs.append({
                    "content": payload.get(content_key, ""),
                    "page": meta.get("page", "?"),
                    "section": meta.get("section"),
                    "clause_id": meta.get("clause_id"),
                    "is_table": meta.get("is_table", False),
                })
        if offset is None:
            break

    # Apply filters
    if args.tables:
        all_docs = [d for d in all_docs if d["is_table"]]
    if args.page is not None:
        all_docs = [d for d in all_docs if d["page"] == args.page]

    # Sort by page then by whether it's prose or table
    all_docs.sort(key=lambda d: (d["page"] if isinstance(d["page"], int) else 999, d["is_table"]))

    if not all_docs:
        print(f"No chunks found for source='{args.source}' with the given filters.")
        return

    print(f"\nSource: {args.source}  |  {len(all_docs)} chunk(s)\n")

    for i, doc in enumerate(all_docs, 1):
        tag = "TABLE" if doc["is_table"] else "PROSE"
        clause = f"  clause={doc['clause_id']}" if doc["clause_id"] else ""
        section = f"  §{doc['section']}" if doc["section"] else ""
        print(f"{'='*70}")
        print(f"[{i}] p.{doc['page']}  {tag}{clause}{section}")
        print(f"{'='*70}")
        print(doc["content"])
        print()


if __name__ == "__main__":
    main()
