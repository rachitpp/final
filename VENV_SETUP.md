# Virtual Environment Setup

---

## Windows (PowerShell)

**1. Check your Python version (3.11+ required)**
```powershell
python --version
```

**2. Create the virtual environment**
```powershell
python -m venv venv
```

**3. Activate it**
```powershell
.\venv\Scripts\Activate.ps1
```

> If you get a script execution policy error, run this first, then retry step 3:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**4. Upgrade pip**
```powershell
python -m pip install --upgrade pip
```

**5. Install dependencies**
```powershell
pip install -r requirements.txt
```

**6. Deactivate when done**
```powershell
deactivate
```

---

## macOS / Linux (bash)

**1. Check your Python version (3.11+ required)**
```bash
python3 --version
```

**2. Create the virtual environment**
```bash
python3 -m venv venv
```

**3. Activate it**
```bash
source venv/bin/activate
```

**4. Upgrade pip**
```bash
python -m pip install --upgrade pip
```

**5. Install dependencies**
```bash
pip install -r requirements.txt
```

**6. Deactivate when done**
```bash
deactivate
```

---

## Environment Variables

This project requires the following environment variables to be set **before** running the app or ingestion script.

### Windows (PowerShell)
```powershell
$env:GOOGLE_CLOUD_PROJECT  = "your-gcp-project-id"
$env:QDRANT_API_KEY        = "your-qdrant-api-key"
$env:CLUSTER_ENDPOINT      = "https://your-cluster.qdrant.io"
$env:LANGCHAIN_API_KEY     = "your-langsmith-api-key"   # optional — for tracing
$env:LANGCHAIN_TRACING_V2  = "true"                     # optional — for tracing
```

### macOS / Linux (bash)
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export QDRANT_API_KEY="your-qdrant-api-key"
export CLUSTER_ENDPOINT="https://your-cluster.qdrant.io"
export LANGCHAIN_API_KEY="your-langsmith-api-key"   # optional — for tracing
export LANGCHAIN_TRACING_V2="true"                  # optional — for tracing
```

> **Tip:** create a `.env` file in the project root and load it with `python-dotenv`
> so you don't have to re-export variables every session.

---

## Verify the Setup

After activating the venv and setting env vars, confirm everything is wired up:

```bash
python -c "import streamlit, langchain, qdrant_client, fitz; print('All imports OK')"
```

Then run the app:
```bash
streamlit run app.py
```

---

## Notes

- The `venv/` folder is excluded from git via `.gitignore` — never commit it.
- Re-run `pip install -r requirements.txt` any time `requirements.txt` changes.
- If you add a new package, freeze the updated deps with:
  ```bash
  pip freeze > requirements.txt
  ```
