# Document Processing Pipeline

A fast PDF processing pipeline with **semantic search** and **AI-powered policy compliance analysis**.

## Features

| Feature | Standard Mode | AI Mode |
|---|---|---|
| PDF parsing & chunking | ✅ | ✅ |
| Semantic search (ChromaDB) | ✅ | ✅ |
| Excel searchable index | ✅ | ✅ |
| Knowledge graph | ✅ | ✅ |
| Policy statement extraction | ❌ | ✅ |
| Compliance question generation | ❌ | ✅ |
| Requirement extraction | ❌ | ✅ |

## Installation

```bash
# Standard mode
pip install pymupdf sentence-transformers chromadb networkx pydantic numpy tqdm pandas openpyxl

# AI mode (adds Mistral integration)
pip install mistralai python-dotenv
```

## Usage

```bash
# Standard processing
python text_extraction.py [path/to/document.pdf]

# AI-powered policy analysis
python text_extraction.py [path/to/document.pdf] --ai
```

### Python API

```python
from src.pipeline import DocumentPipeline

pipeline = DocumentPipeline(
    chunk_by="section",          # or "paragraph"
    embedding_model="all-MiniLM-L6-v2",
    output_dir="./output",
    max_chunk_chars=3000,
    min_chunk_chars=500
)

results = pipeline.process_document("document.pdf")

# Semantic search
search_results = pipeline.search("data security policy", n_results=5)
for result in search_results['results']:
    print(f"[{result['rank']}] Score: {result['score']}")
    print(f"Section: {result.get('section_title')}, Pages: {result.get('pages')}")
    print(result['text'][:200])

# Keyword search via Excel
excel_results = pipeline.search_in_excel(results['excel_path'], "security")
```

## Output Structure

```
output/
├── vectordb/                          # ChromaDB vector database
├── chunks/[document]/
│   ├── chunk_001.txt                  # Individual chunk files
│   └── index.json
├── [document]_searchable_index.xlsx   # Excel search interface
├── [document]_parsed.json             # Hierarchical structure
├── [document]_graph.json              # Knowledge graph
│
│   # AI mode only:
├── ai_compliance_questions.json
├── ai_compliance_questions.xlsx       # 4 sheets: All / By Category / By Type / Requirements
└── policy_statements.json
```

## Configuration

**Chunking strategies:**
```python
DocumentPipeline(chunk_by="section", max_chunk_chars=3000, min_chunk_chars=500)
DocumentPipeline(chunk_by="paragraph", max_chunk_chars=2000)
```

**Embedding models:**
```python
DocumentPipeline(embedding_model="all-MiniLM-L6-v2")   # Default — fast, 384-dim
DocumentPipeline(embedding_model="all-mpnet-base-v2")   # Better quality, 768-dim
```

## Performance

| Document Size | Standard | AI Mode |
|---|---|---|
| 10 pages | ~5s | +10–15s |
| 25 pages | ~10s | +20–30s |
| 50 pages | ~15s | +45–60s |
| 100 pages | ~30s | +90–120s |

> First run is slower due to embedding model download (~90MB). AI mode speed depends on Mistral API rate limits.

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Re-run the `pip install` command for your mode |
| API key error | Create `.env` with `MISTRAL_API_KEY=your_key` ([get key](https://console.mistral.ai/)) |
| Quota exceeded | Wait or upgrade Mistral tier |
| No PDFs found | Place PDFs in `./samples/` |
| Excel file locked | Close the file before re-running |
| Memory issues | Use `chunk_by="paragraph"` for large documents |

## Demo

**Standard mode:**

```
$ python text_extraction.py

================================================================================
DOCUMENT PROCESSING PIPELINE
================================================================================
Mode: Standard Processing
File: samples/Information Security & Management Policy v3.pdf
================================================================================

Step 1: Parsing PDF...
  Pages: 23 | Sections: 6

Step 2: Intelligent chunking (max 3000 chars per chunk)...
  Created 18 chunks from 6 sections
  - Average chunk size: 2401 characters
  - Largest chunk:     2994 characters
  - Smallest chunk:      91 characters

Step 3: Generating Excel searchable index...
  output\Information Security & Management Policy v3_searchable_index.xlsx

Step 4: Writing chunks to individual txt files...
  18 files in output\chunks\Information Security & Management Policy v3\

Step 5: Creating chunk index...
  output\chunks\Information Security & Management Policy v3\index.json

Step 6: Generating embeddings and indexing...
  Batches: 100%|████████████| 1/1 [00:00<00:00,  1.79it/s]
  Indexed 18 chunks

Step 7: Building knowledge graph...
  Saved JSON outputs to output\

PROCESSING STATISTICS:
  Total Pages:    23
  Total Sections:  6
  Total Chunks:   18
  Avg Chunk Size: 2401 chars
  Files Created:  19
```

**AI mode** (`--ai` flag) — adds policy analysis and compliance question generation:

```
$ python text_extraction.py --ai

================================================================================
AI-POWERED POLICY DOCUMENT PROCESSING
================================================================================
Mode: AI + Compliance Analysis
File: samples/Information Security & Management Policy v3.pdf
================================================================================

... (same 7 processing steps as above) ...

================================================================================
AI POLICY ANALYSIS
================================================================================

[1/3] Analyzing policy statements...
  Found 112 policy statements
  Identified 45 requirement statements

  Requirements by category:
    General:            12
    Data Security:      11
    Access Control:      6
    Compliance:          6
    Privacy:             3
    Vendor Management:   3
    Incident Response:   1
    Risk Management:     1
    Training & Awareness:1
    Audit & Monitoring:  1

[2/3] Generating AI compliance questions...
  Model: mistral-small-latest | Statements: 45 | Batch size: 5

  Processing batch 1... 15 questions
  Processing batch 2... 15 questions
  Processing batch 3... 15 questions
  ...
  Total questions generated: 138

[3/3] Exporting AI results...
  output\ai_compliance_questions.json
  output\policy_statements.json
  output\ai_compliance_questions.xlsx

  Sample questions (first 3):
  1. What specific procedures and controls have been implemented to protect
     the confidentiality and integrity of information?
     Category: General | Type: implementation

  2. How often are these procedures and controls audited to ensure
     their effectiveness?
     Category: General | Type: audit

  3. What mechanisms are in place to verify that information is only
     accessible to authorized persons?
     Category: General | Type: verification
```
