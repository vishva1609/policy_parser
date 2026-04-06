# Document Processing Pipeline

A PDF processing pipeline for semantic search and AI-powered policy compliance analysis. Built with LangChain and Mistral for intelligent question generation from policy documents.

## Features

| Feature | Standard Mode | Advanced Mode |
|---|---|---|
| PDF parsing and chunking | Yes | Yes |
| Semantic search (ChromaDB) | Yes | Yes |
| Excel searchable index | Yes | Yes |
| Knowledge graph generation | Yes | Yes |
| Policy statement extraction | -- | Yes |
| Compliance question generation | -- | Yes |
| Requirement extraction | -- | Yes |

## Prerequisites

- Python 3.9 or higher
- Mistral API key (required for advanced mode only)

## Installation

```bash
pip install -e .
```

For environment variable support:

```bash
pip install python-dotenv
```

## Usage

### Command Line

```bash
# Standard processing
python text_extraction.py [path/to/document.pdf]

# Advanced mode with compliance analysis
python text_extraction.py [path/to/document.pdf] --advanced
```

If no PDF path is provided, the pipeline defaults to `samples/Information Security & Management Policy v3.pdf`.

### Advanced Mode Options

The following parameters control LLM behavior during question generation:

```bash
python text_extraction.py document.pdf --advanced [options]
```

| Option | Default | Description |
|---|---|---|
| `--temperature=X` | 0.3 | Controls output randomness. 0.0 = deterministic, 1.0 = creative. Lower values produce more focused, consistent compliance questions. |
| `--max-tokens=N` | 2048 | Maximum response length in tokens. Increase if responses are getting truncated. |
| `--top-p=X` | 1.0 | Nucleus sampling threshold. Lower values restrict output to higher-probability tokens. Avoid changing both temperature and top_p simultaneously. |
| `--model=NAME` | mistral-small-latest | Mistral model to use. Options: `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`, and open-source variants. |

### Interactive Mode

Configure hyperparameters at runtime with prompts:

```bash
# Launch interactive configuration
python text_extraction.py document.pdf --advanced --interactive
# or
python text_extraction.py document.pdf --advanced -i
```

This will prompt you for each hyperparameter with explanations and allow you to select a model from a numbered list.

### Auto-Tuning

Automatically find optimal hyperparameters for your document:

```bash
python text_extraction.py document.pdf --advanced --auto-tune
```

Auto-tuning tests multiple hyperparameter combinations and scores results based on:
- Coverage: Does it generate questions for all statements?
- Specificity: Are questions detailed enough?
- Diversity: Are different question types generated?
- Actionability: Do questions contain action-oriented language?

Results are saved to model-specific files under `output/tuning/`, for example `output/tuning/tuning_results_mistral-small-latest.json`.

### Model Comparison

Compare different models to find the most efficient one for your use case:

```bash
# Compare all available models
python text_extraction.py document.pdf --advanced --compare-models

# List available models with descriptions
python text_extraction.py --list-models
```

This generates a comparison report showing:
- Question quality scores
- Generation time per model
- Recommendations for different use cases (production, development, cost-optimized)

### Examples

```bash
# Use a larger model for higher quality questions
python text_extraction.py document.pdf --advanced --model=mistral-large-latest

# More focused output with lower temperature
python text_extraction.py document.pdf --advanced --temperature=0.1

# Increase token limit for longer responses
python text_extraction.py document.pdf --advanced --max-tokens=4096

# Interactive mode - prompts for all settings
python text_extraction.py document.pdf --advanced -i

# Find best hyperparameters automatically
python text_extraction.py document.pdf --advanced --auto-tune

# Compare models and use the best one
python text_extraction.py document.pdf --advanced --compare-models
```

### Available Models

| Model | Speed | Quality | Cost | Best For |
|---|---|---|---|---|
| mistral-small-latest | Fast | Good | Low | Development, testing, prototyping |
| mistral-medium-latest | Medium | Better | Medium | Balanced production use |
| mistral-large-latest | Slower | Best | High | Critical compliance applications |
| open-mistral-7b | Fast | Good | Free | Self-hosting, cost-sensitive |
| open-mixtral-8x7b | Medium | Better | Free | Open-source balanced option |
| open-mixtral-8x22b | Slower | Best (open) | Free | Highest quality open-source |

### Python API

```python
from src.pipeline import DocumentPipeline

pipeline = DocumentPipeline(
    chunk_by="section",
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
    print(f"Section: {result.get('section')}, Pages: {result.get('pages')}")
    print(result['text'][:200])

# Keyword search via Excel
excel_results = pipeline.search_in_excel(results['excel_path'], "security")
```

### Adaptive Tuning for Deployment

For end-user deployment where users shouldn't need to manually tune hyperparameters:

```python
from src.question_generator import QuestionGenerator, AdaptiveTuner

# Initialize adaptive tuner (loads pre-tuned configs)
tuner = AdaptiveTuner(config_path="output/tuning/tuning_results_mistral-small-latest.json")

# Get optimal parameters based on document type detection
optimal_params = tuner.get_optimal_params(chunks=your_chunks)

# Use detected/optimal parameters
generator = QuestionGenerator(
    temperature=optimal_params['temperature'],
    max_tokens=optimal_params['max_tokens'],
    top_p=optimal_params['top_p']
)

# Generate questions with optimal settings
questions = generator.generate_questions_from_statements(statements)

# Collect user feedback for future improvements
AdaptiveTuner.save_feedback(questions, {'quality': 'good', 'coverage': 'complete'})
```

**Deployment workflow:**
1. Run `--auto-tune` during initial setup to find optimal parameters
2. Results are saved to `output/tuning/tuning_results_<model>.json`
3. `AdaptiveTuner` loads these results automatically for future runs
4. Document type detection provides fallback presets (policy, technical, general)
5. User feedback is collected to improve tuning over time
```

## Output Structure

```
output/
├── vectordb/                          # ChromaDB vector database
├── chunks/[document]/
│   ├── chunk_001.txt                  # Individual chunk files
│   └── index.json
├── [document]_searchable_index.xlsx   # Excel searchable index
├── [document]_parsed.json             # Parsed document structure
├── [document]_graph.json              # Knowledge graph
│
│   # Advanced mode only:
├── compliance_questions.json
├── compliance_questions.xlsx          # Sheets: Questions / By Category / By Type / Requirements
└── policy_statements.json
```

## Configuration

### Chunking Strategies

```python
# Section-based (default) — groups content by document sections
DocumentPipeline(chunk_by="section", max_chunk_chars=3000, min_chunk_chars=500)

# Paragraph-based — finer granularity, useful for large documents
DocumentPipeline(chunk_by="paragraph", max_chunk_chars=2000)
```

### Embedding Models

```python
# Default — fast, 384 dimensions
DocumentPipeline(embedding_model="all-MiniLM-L6-v2")

# Higher quality, 768 dimensions
DocumentPipeline(embedding_model="all-mpnet-base-v2")
```

## Project Structure

```
├── text_extraction.py          # CLI entry point
├── pyproject.toml              # Project configuration and dependencies
├── src/
│   ├── pipeline.py             # Main pipeline orchestrator
│   ├── parser.py               # PDF parsing (PyMuPDF)
│   ├── chunker.py              # Document chunking with boundary detection
│   ├── embedder.py             # Embedding generation and ChromaDB indexing
│   ├── knowledge_graph.py      # Document relationship graph (NetworkX)
│   ├── chunk_file_writer.py    # Individual chunk file output
│   ├── excel_generator.py      # Excel index and compliance report generation
│   ├── schemas.py              # Pydantic data models
│   ├── policy_analyzer.py      # Policy statement extraction and categorization
│   ├── question_generator.py   # LangChain + Mistral compliance question generation
│   │                           # Includes AutoTuner and AdaptiveTuner classes
│   └── model_comparison.py     # Model comparison and benchmarking tool
├── samples/                    # Input PDF documents
└── output/                     # Generated outputs (gitignored)
```

## How It Works

### Standard Pipeline (7 steps)

1. **Parse PDF** — Extracts text with font metadata, detects headings and section boundaries
2. **Chunk** — Splits content by section or paragraph, respecting sentence boundaries
3. **Excel Index** — Generates a searchable `.xlsx` with content, page numbers, and keywords
4. **Chunk Files** — Writes individual `.txt` / `.json` / `.md` files per chunk
5. **Chunk Index** — Creates a JSON index of all chunks with metadata
6. **Embed and Index** — Generates sentence-transformer embeddings, stores in ChromaDB
7. **Knowledge Graph** — Builds a directed graph linking documents, sections, and chunks

### Advanced Pipeline (adds 3 steps)

1. **Policy Statement Extraction** — Identifies and categorizes policy statements across 14 categories (Access Control, Data Security, Compliance, Privacy, etc.)
2. **Compliance Question Generation** — Uses LangChain with Mistral to generate audit, implementation, verification, and requirement questions from extracted policy statements
3. **Report Export** — Outputs questions and statements to JSON and a multi-sheet Excel report

Non-content pages (cover pages, table of contents) are automatically detected and excluded from analysis.

## Environment Setup

Create a `.env` file in the project root for advanced mode:

```
MISTRAL_API_KEY=your_api_key_here
```

Get an API key from [Mistral Console](https://console.mistral.ai/).

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Run `pip install -e .` to install all dependencies |
| API key error | Create `.env` with `MISTRAL_API_KEY=your_key` |
| Quota exceeded | Wait for rate limit reset or upgrade Mistral plan |
| No PDF found | Provide a path or place PDFs in `./samples/` |
| Excel file locked | Close the file in Excel before re-running |
| Memory issues | Use `chunk_by="paragraph"` for large documents |
| Truncated responses | Increase `--max-tokens` value |

## License

MIT
