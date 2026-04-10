<<<<<<< Updated upstream
# Policy Compliance Pipeline

This project compares two policy documents clause by clause and shows where **Policy B** is compliant, partially compliant, non-compliant, or missing coverage against **Policy A** (source of truth).

It targets policy audit workflows where you need the verdict plus traceability: file, clause, sub-clause, section, page, and explanation.

The layout follows the same layered idea as [qa-bot-llm](https://github.com/archit1012/qa-bot-llm): **views** (HTTP/CLI) → **service** (orchestration) → **models** (parsers, segmenters, embedders) → **common** (schemas, LLM helpers).

## Repository layout (project root)

Keep these folders at the **main directory** (repository root):

| Folder | Purpose |
|--------|---------|
| **`tests/`** | Pytest suite. Convention is root-level `tests/`; run with `uv run pytest`. |
| **`samples/`** | Example PDFs or small fixtures for local runs and demos. Large binaries can stay untracked (see `.gitignore`). |
| **`output/`** | **Generated** artifacts: Excel reports, JSON, ChromaDB `vectordb/`, chunk exports. Listed in `.gitignore` so you do not commit build output. |

Application code lives under **`app/`** (not `src/`).

## What it implements today

- PDF parsing with **PyMuPDF**
- Optional **JSON** document path for the document-QA flow (see Flask `/upload`)
- Clause and sub-clause segmentation (regex-first, optional LLM fallback)
- Clause embeddings (**sentence-transformers**) and **ChromaDB** indexing
- Semantic top-k matching and **Mistral**-based compliance reasoning with gap detection
- Bidirectional scoring (A→B and B→A)
- **Excel** and **JSON** comparison outputs; segment manifest for audit
- **FastAPI** endpoint for policy comparison (Excel download)
- **Flask** endpoint for RAG Q&A over a single document (same pattern as qa-bot `/upload`)

## Installation (uv)

This project is set up for **[uv](https://github.com/astral-sh/uv)** and a local **`.venv/`**.

```powershell
cd "path\to\File Parsing"
uv sync --all-groups
```

Optional: activate the venv.

```powershell
.\.venv\Scripts\Activate.ps1
```

Dependencies are declared in `pyproject.toml`; lockfile is `uv.lock`.

Environment:

- **`MISTRAL_API_KEY`** — required for LLM segmentation, compliance reasoning, and RAG QA.
- **`MISTRAL_MODEL`** — optional override (default `mistral-small-latest`).
- **`EMBEDDING_MODEL`** — optional override for sentence-transformers (API layers).

## Configuration

Root **`config.yaml`** drives defaults for chunking, comparison thresholds, LLM model name, and output directory. Load programmatically via `PipelineRunner.from_config_file()` or the helpers used by the FastAPI comparison service.

## Main pipelines

### 1. Policy comparison (Python API)

```python
from pathlib import Path
from app.service.comparison_service import PolicyComparisonPipeline

pipeline = PolicyComparisonPipeline(output_dir="./output")
report_path = pipeline.compare(
    pdf_a="samples/policy_a.pdf",
    pdf_b="samples/policy_b.pdf",
)
print(report_path)
```

Or use the **views** layer with dependency injection:

```python
from app.views.pipeline_runner import PipelineRunner

runner = PipelineRunner()  # or PipelineRunner.from_config_file()
excel_path = runner.run_comparison_pipeline("samples/A.pdf", "samples/B.pdf")
```

### 2. Manifest-only (no LLM comparison cost)

Inspect segmentation and metadata before a full run:

```python
from app.service.comparison_service import PolicyComparisonPipeline

pipeline = PolicyComparisonPipeline(output_dir="./output", use_llm_segmentation=False)
manifest_path = pipeline.build_manifest("samples/A.pdf", "samples/B.pdf")
```

### 3. Single-document pipeline (CLI)

```powershell
uv run python text_extraction.py samples\your.pdf
uv run python text_extraction.py samples\your.pdf --advanced
```

### 4. HTTP APIs

**FastAPI — policy comparison** (default port 8000):

```powershell
uv run uvicorn app.fastapi_app:app --reload --port 8000
```

- `GET /health` — liveness
- `POST /compare/policies` — form fields `policy_a`, `policy_b` (PDFs); returns an **Excel** report when `MISTRAL_API_KEY` is set  
- `POST /tools/semantic-search` — form fields: `policy` (PDF), `query`, optional `top_k` (1–50). Indexes that PDF in a temp workspace and returns ranked chunk hits (for debugging retrieval).
- `POST /tools/questions-from-policy` — form field `policy` (PDF) plus optional: uploaded `tuning_json` (AutoTuner output), or `use_cached_tuning` + `tuning_model` + `output_dir` to load `output/tuning/tuning_results_<model>.json`. Uses **AdaptiveTuner** `best_params` / **score** for **QuestionGenerator** hyperparameters before you run a full compare.
- `GET /tools/tuning/available` — lists cached `tuning_results_*.json` files, or checks one model’s file and returns `cached_score` if present.

**Sessions — step-by-step comparison (debug each stage)**

1. `POST /sessions` — multipart `policy_a`, `policy_b` (PDFs) → `{ session_id }`.
2. `GET /sessions/{session_id}` — which steps have finished + paths.
3. `POST /sessions/{session_id}/steps/parse` → parsed JSON on disk.
4. `POST /.../steps/segment` → clause JSON.
5. `POST /.../steps/embed` → Chroma + `segment_manifest.json`.
6. `POST /.../steps/compare` — needs `MISTRAL_API_KEY` → comparison result JSON.
7. `POST /.../steps/score` → `comparison_scores.json`.
8. `POST /.../steps/export` → Excel + full comparison JSON.
9. `GET /.../exports/excel` / `GET /.../exports/json` — download artifacts.
10. `GET /.../artifacts/parsed/{a|b}` / `GET /.../artifacts/manifest` — inspect JSON.
11. `DELETE /sessions/{session_id}` — remove session folder.

- OpenAPI: `http://localhost:8000/docs`

**Flask — document Q&A** (RAG; default port 5000 in `app/app.py`):

```powershell
uv run python -m app.app
```

- `POST /upload` — `doc_file` + `question_file` (see `APIs.postman_collection.json`)

Postman variables: `base_url` (Flask), `fastapi_url` (FastAPI).

## Important output files (under `./output/`)

These are created at runtime (folder is gitignored except optional `.gitkeep`):

- `segment_manifest.json` — clauses and sub-clauses prepared for indexing  
- `comparison_<a>_vs_<b>.json` — structured comparison result  
- `compliance_report_*.xlsx` — human-readable report  
- `vectordb/` — persistent ChromaDB data for the pipeline  

## Vector DB traceability (Chroma metadata)

Each embedded clause/sub-clause is stored in Chroma with rich metadata so you can backtrack matches during audits and debugging. In addition to clause identifiers (e.g. `clause_id`, `sub_clause_id`, `parent_clause_id`) the stored metadata includes:

- `session_id` — ties embeddings to a specific FastAPI session (step-by-step flow) or one-shot compare run
- `indexed_at` — UTC ISO timestamp of indexing
- `source_policy` / `policy_label` — which policy the chunk came from (`A`/`B`)
- `source_pdf_name` / `filename` — original PDF name
- `source_page_start` / `source_page_end` (and `page`) — provenance in the source document
- `text_sha256` — hash of chunk text to detect drift / re-embedding changes

## Similarity and gap thresholds

`PolicyComparisonPipeline` accepts:

- `missing_similarity_threshold` — similarities below this tend toward **Missing** (default `0.30`)  
- `gap_similarity_threshold` — used with LLM output for **Non-Compliant** / gap logic (default `0.50`)  

You can set these in **`config.yaml`** (used when running via `PipelineRunner.from_config_file()` or the FastAPI comparison path that merges YAML defaults).

## Tests

From the repository root:

```powershell
uv run pytest tests -q
```

## Roadmap (high level)

- DOCX and additional ingest backends (beyond PDF/JSON)  
- Stronger segmentation on difficult PDFs; optional obligation-aware rules (`must` vs `should`)  
- Async or job-based comparison for very large policies  
- Richer scoring (e.g. critical vs non-critical gaps)  
- Optional PDF report export; separate frontend consumes Excel/JSON APIs  

## Low-cost tips

- Run **`build_manifest()`** before **`compare()`** to validate segmentation.  
- Set **`use_llm_segmentation=False`** when PDFs are clean and structured.  
- Prefer **`mistral-small-latest`** unless you have measured need for a larger model.  
=======
# 🛡️ Policy Compliance AI Backend (v2.1)

A professional, high-throughput backend engine for clause-by-clause policy comparison using **Semantic Vector Search (FAISS)** and **Hybrid LLM + Rule-Based Gap Analysis**.

---

## 🏗️ Architecture

- **Ingestion:** Powered by `PDFPlumber` (clean text extraction on Windows).
- **Segmentation:** Sophisticated regex-based state machine for identifying hierarchical clauses, sub-clauses, and bullet points.
- **Search Engine:** `Sentence-Transformers` generates 384-dimensional embeddings, indexed by `FAISS` for millisecond-latency similarity matching.
- **Comparison Engine:** Hybrid logic that detects clause **Strength** (Must/Shall vs Should) and **Category** (Security, Privacy, etc.).
- **Scoring:** Weighted compliance system with **Critical Gap** identification.
- **Export:** High-fidelity Multi-Sheet Excel and Legal-Style PDF reports.

---

## 🚀 Quick Start

1.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

2.  **Configure System (`config.json`):**
    Customize LLM backends, similarity thresholds, and precision levels.
    ```json
    {
      "llm": { "backend": "rules", "openai_model": "gpt-4o-mini" },
      "matching": { "top_k": 5 }
    }
    ```

3.  **Launch the Backend:**
    ```powershell
    uvicorn compliance_api:app --port 8000
    ```

---

## 📡 API Endpoints (v2)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check |
| `POST` | `/parse` | (Multipart) Upload PDF → Structured JSON |
| `POST` | `/parse/text` | (JSON) Raw text → Structured JSON |
| `POST` | `/compare` | (JSON) Compare two pre-parsed clause lists |
| `POST` | `/compare/pdf` | (Multipart) Upload 2 PDFs → Start Async Job |
| `GET` | `/compare/{job_id}` | Poll Job Status (queued | running | done) |
| `GET` | `/compare/{job_id}/report` | Get Full Comparison JSON |
| `GET` | `/compare/{job_id}/export` | Download `.xlsx` or `.pdf` report |

---

## 🧪 Testing the API

For "Backend Only" developers, use the provided tools:

- **Postman:** Import `Policy_Parser_API.postman_collection.json`.
- **Smoke Test:** Run `python test_compliance.py` for a full end-to-end trace.

---

## 🔧 LLM Integration
To enable OpenAI analysis for deeper insights:
1. Set your environment variable: `$env:OPENAI_API_KEY="sk-..."`
2. Change `config.json` → `"backend": "openai"`
3. The system will automatically use GPT-4o-mini for logical gap detection instead of similarity-only rules.
>>>>>>> Stashed changes
