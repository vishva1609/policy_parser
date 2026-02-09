# Document Processing Pipeline

A simple, fast, and reliable PDF processing pipeline with semantic search capabilities.

## Features

- **PDF Parsing**: Extract text and structure from PDFs using PyMuPDF
- **Smart Chunking**: Split documents into sections with hierarchical context
- **Semantic Embeddings**: Generate vector embeddings for semantic search
- **Vector Database**: Index and search documents using ChromaDB
- **Knowledge Graph**: Track document relationships and structure

## Installation

```bash
pip install pymupdf sentence-transformers chromadb networkx pydantic numpy tqdm
```

That's it! No complex dependencies or external tools required.

## Quick Start

### 1. Place PDFs in the samples folder

```bash
# Copy your PDF files
cp your_document.pdf samples/
```

### 2. Run the pipeline

```bash
python text_extraction.py
```

### 3. Search your documents

The pipeline will prompt you to enter a search query after processing.

## Usage

### Process a Single PDF

```bash
python text_extraction.py path/to/document.pdf
```

### Process All PDFs in samples/

```bash
python text_extraction.py
```

### Python API

```python
from src.pipeline import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline(
    chunk_by="section",              # or "paragraph"
    embedding_model="all-MiniLM-L6-v2",
    output_dir="./output"
)

# Process a document
results = pipeline.process_document("document.pdf")

# Access results
print(f"Sections: {len(results['document'].sections)}")
print(f"Chunks: {len(results['chunks'])}")

# Search across all indexed documents
search_results = pipeline.search("data security", n_results=5)

# Display results
for result in search_results['results']:
    print(f"[{result['rank']}] Score: {result['score']}")
    print(f"Document: {result['document']}")
    print(f"Section: {result['section']}")
    print(f"Text: {result['text'][:200]}...")
    print()

# Get statistics
stats = pipeline.get_stats()
print(f"Total indexed chunks: {stats['vector_db']['total_chunks']}")
print(f"Knowledge graph nodes: {stats['knowledge_graph']['nodes']}")
```

## Pipeline Stages

### Stage 1: Parse PDF
- Extracts text from all pages
- Identifies sections and headings
- Preserves document structure
- Saves to `output/*_parsed.json`

### Stage 2: Chunk & Enrich
- Splits document into semantic chunks
- Adds parent context (breadcrumb trail)
- Preserves page references
- Saves to `output/*_chunks.json`

### Stage 3: Embed & Index
- Generates semantic embeddings
- Indexes in ChromaDB vector database
- Enables semantic search

### Stage 4: Knowledge Graph
- Builds document relationship graph
- Tracks sections and chunks
- Saves to `output/*_graph.json`

## Output Files

After processing, the pipeline generates:

```
output/
├── vectordb/                      # ChromaDB vector database
├── document_parsed.json           # Hierarchical document structure
├── document_chunks.json           # Chunks with metadata
└── document_graph.json            # Knowledge graph
```

## Configuration Options

### Chunking Strategy

```python
# Section-level chunking (default)
pipeline = DocumentPipeline(chunk_by="section")

# Paragraph-level chunking (more granular)
pipeline = DocumentPipeline(chunk_by="paragraph")
```

### Embedding Model

```python
# Fast, smaller model (default)
pipeline = DocumentPipeline(
    embedding_model="all-MiniLM-L6-v2"
)

# Better quality, larger model
pipeline = DocumentPipeline(
    embedding_model="all-mpnet-base-v2"
)
```

## Project Structure

```
file-parsing/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── schemas.py           # Pydantic data models
│   ├── parser.py            # PDF parsing with PyMuPDF
│   ├── chunker.py           # Document chunking logic
│   ├── embedder.py          # Embeddings and vector DB
│   ├── knowledge_graph.py   # Graph management
│   └── pipeline.py          # Main orchestrator
├── samples/                 # Input PDFs (place your PDFs here)
├── output/                  # Generated files
├── text_extraction.py       # CLI entry point
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Example Output

```
============================================================
Document Processing Pipeline
============================================================
Loading embedding model: all-MiniLM-L6-v2
Vector database ready (0 existing chunks)

Configuration:
  - Chunking: section
  - Embedding: all-MiniLM-L6-v2
  - Output: ./output
============================================================

Found 1 PDF file(s)

============================================================
Processing: research_paper.pdf
============================================================

STAGE 1: Parse PDF
------------------------------------------------------------
Pages: 23
Extracted 6 sections
✓ Parsing complete

STAGE 2: Chunk & Enrich
------------------------------------------------------------
Created 17 chunks

STAGE 3: Embed & Index
------------------------------------------------------------
Generating embeddings for 17 chunks...
Indexing 17 chunks...
✓ Embedding and indexing complete

STAGE 4: Knowledge Graph
------------------------------------------------------------
✓ Knowledge graph built (24 nodes)

============================================================
✓ Processing Complete!
============================================================
Document: research_paper.pdf
  - Sections: 6
  - Chunks: 17
  - Embeddings: 17
  - Graph nodes: 24
============================================================

Enter search query: machine learning

Top results for: 'machine learning'
------------------------------------------------------------

[1] Score: 0.742
    Document: research_paper.pdf
    Section: Introduction
    Text: Machine learning algorithms have revolutionized...
```

## Why This Implementation?

### Simple & Reliable
- Uses PyMuPDF (battle-tested, no external dependencies)
- Minimal dependencies (just 7 core packages)
- No complex model downloads or setup
- Works out of the box on Windows, Mac, Linux

### Fast & Efficient
- PyMuPDF is one of the fastest PDF parsers
- Batch embedding generation
- Efficient vector storage with ChromaDB
- Processes typical documents in seconds

### Complete Solution
- PDF parsing and structure extraction
- Semantic chunking with context
- Vector embeddings for search
- Knowledge graph for relationships
- Ready for RAG applications

## Dependencies

- **pymupdf** - Fast, reliable PDF parsing
- **sentence-transformers** - Semantic embeddings
- **chromadb** - Vector database for search
- **networkx** - Knowledge graph management
- **pydantic** - Data validation and schemas
- **numpy** - Numerical operations
- **tqdm** - Progress bars

## Requirements

- Python 3.9 or higher
- 2GB RAM minimum
- Internet connection (for first-time model download)

## Use Cases

- **Document Q&A Systems**: Build chatbots that answer questions from your documents
- **Semantic Search**: Find relevant content using natural language queries
- **Document Analysis**: Extract and analyze document structure and content
- **RAG Applications**: Retrieval-Augmented Generation for LLMs
- **Knowledge Management**: Index and search large document collections

## Performance

Typical processing times (on standard hardware):

| Document Size | Processing Time |
|--------------|----------------|
| 10 pages     | ~5 seconds     |
| 50 pages     | ~15 seconds    |
| 100 pages    | ~30 seconds    |

*First run will be slower due to model download (~90MB)*

## Troubleshooting

### Import Error
```bash
pip install pymupdf sentence-transformers chromadb networkx pydantic numpy tqdm
```

### No PDFs Found
Place PDF files in the `./samples/` directory

### Memory Issues
Use paragraph-level chunking or process documents one at a time:
```python
pipeline = DocumentPipeline(chunk_by="paragraph")
```

### Slow First Run
The embedding model (~90MB) downloads on first use. Subsequent runs are fast.

## Advanced Usage

### Custom Chunking Logic

```python
from src.chunker import DocumentChunker

chunker = DocumentChunker(
    chunk_by="section",
    max_words=500  # Maximum words per chunk
)
```

### Search with Metadata Filters

```python
# Search only specific document
results = pipeline.search(
    "security policy",
    n_results=5,
    where={"document": "policy.pdf"}
)
```

### Access Raw Data

```python
# Get parsed document structure
doc = results['document']
for section in doc.sections:
    print(f"Section: {section.title}")
    print(f"Pages: {section.page_start}-{section.page_end}")
    print(f"Elements: {len(section.elements)}")

# Get chunks
for chunk in results['chunks']:
    print(f"Type: {chunk.chunk_type}")
    print(f"Context: {chunk.parent_context}")
    print(f"Pages: {chunk.pages}")
```

## License

MIT

## Contributing

Contributions welcome! This is a simplified implementation focused on reliability and ease of use.

## Support

For issues or questions, please check the troubleshooting section above or review the code - it's simple and well-commented.

---

**Built with simplicity and reliability in mind. Just works.** 🚀
