# =============================================================
# Database creation entrypoint.
# Run once after dropping new PDFs in.
#
# Usage:
#   python create_db.py           # add / refresh current pdf/ folder
#   python create_db.py --wipe    # drop both Qdrant collections first,
#                                 # then ingest fresh (use when removing PDFs)
# =============================================================
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from qdrant_client import QdrantClient
from pipelines.ingestion_pipeline import run_ingestion
from config.settings import settings


def _wipe_collections() -> None:
    """Delete both Qdrant collections so re-ingestion starts from scratch."""
    import os
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    for name in (settings.qdrant_collection, settings.qdrant_summary_collection):
        if client.collection_exists(name):
            client.delete_collection(name)
            print(f"  Dropped collection '{name}'")
        else:
            print(f"  Collection '{name}' did not exist, skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete existing Qdrant collections before ingesting (required when removing PDFs from the corpus)",
    )
    args = parser.parse_args()

    if args.wipe:
        print("Wiping existing collections...")
        _wipe_collections()

    run_ingestion(settings.pdf_path)
    print("\n✓ Database creation complete. Run main.py to start querying.")
