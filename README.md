pipeline = PolicyComparisonPipeline(output_dir="./output")
report_path = pipeline.compare(
    pdf_a="samples/policy_a.pdf",
    pdf_b="samples/policy_b.pdf",
)
print(report_path)
pipeline = PolicyComparisonPipeline(output_dir="./output", use_llm_segmentation=False)
manifest_path = pipeline.build_manifest("samples/A.pdf", "samples/B.pdf")
=======
# 🛡️ Policy Compliance AI Backend (v2.1)

A professional, high-throughput backend engine for clause-by-clause policy comparison using **Semantic Vector Search (FAISS)** and **Hybrid LLM + Rule-Based Gap Analysis**.


## 🏗️ Architecture



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


## 🧪 Testing the API

For "Backend Only" developers, use the provided tools:



## 🔧 LLM Integration
To enable OpenAI analysis for deeper insights:
1. Set your environment variable: `$env:OPENAI_API_KEY="sk-..."`
2. Change `config.json` → `"backend": "openai"`
3. The system will automatically use GPT-4o-mini for logical gap detection instead of similarity-only rules.
>>>>>>> Stashed changes
