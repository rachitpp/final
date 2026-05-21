# =============================================================
# Database creation entrypoint.
# Run once after dropping new PDFs in.
# =============================================================
from dotenv import load_dotenv
load_dotenv()

from pipelines.ingestion_pipeline import run_ingestion
from config.settings import settings


if __name__ == "__main__":
    run_ingestion(settings.pdf_path)
    print("\n✓ Database creation complete. Run main.py to start querying.")
