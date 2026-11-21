# Fraud Research Assistant

This is the working **application foundation**, not the completed research assistant. It contains six pages and an optional OpenRouter API connectivity test. It does **not** answer research questions until verified BM25, FAISS and metadata assets are connected.

## Setup on Windows (VS Code / PowerShell)

Use Python 3.11 or 3.12. Open a terminal in this folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` with your own OpenRouter key. You can run the app without a key; only the connection test is disabled. Then:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

If PowerShell execution policy blocks activation, use the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Next step: connect experiment CSVs

Copy files from Notebook 09 and Notebook 11 into `data/reports/`. Do not invent CSV schemas; inspect their columns first. Required reports include retrieval summary/timings and `system_summary.csv`, `metrics_by_question_type.csv`, `error_casebook.csv`, and `error_taxonomy_counts.csv`.

## Step after that: live retrieval

Collect the trusted section-aware BM25 index, FAISS index, vector-ID mapping (if separate), passage records, and corpus metadata. Verify they align before loading. A pickle file can execute arbitrary code; load only your own trusted BM25 pickle.

## Experimental integrity

Historical A–E scores come from local Qwen3:8B generation and judging. The live Streamlit app will use an OpenRouter model and needs its own small validation study. Do not present the old Qwen scores as new API-model benchmark results.

## Deployment

- Use Streamlit Community Cloud and its Secrets settings; never commit `secrets.toml`.
- Free-model request quotas can interrupt public demos.
- Index files and copyrighted document passages need storage and publication review.
