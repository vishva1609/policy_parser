# Policy Parser 📄🔍

A robust Python-based tool and API designed to extract, structure, and segment hierarchical information from bank policy PDF documents. It transforms unstructured PDF text into a standardized JSON format, identifying clauses, sub-clauses, and bullet points with high precision.

---

## ✨ Features

- **PDF Text Extraction**: Uses Apache Tika for reliable text extraction from complex PDF layouts.
- **Hierarchical Segmentation**: Intelligently identifies section headings (e.g., "1", "1.1", "Section 2") and nesting levels.
- **Keyword Extraction**: Uses NLTK for frequency-based keyword extraction from each segment.
- **Context Generation**: Provides short summaries/context for each clause based on leading sentences.
- **FastAPI Integration**: A production-ready REST API for remote parsing.
- **CLI Utility**: A simple command-line interface for local processing.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Java (JRE)**: Required to run the Apache Tika server (the JAR is included in the repo).

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd policy_parser
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **NLTK Data**:
   The script will automatically attempt to download the required NLTK data (`punkt`, `stopwords`) on first run.

---

## 🛠 Usage

### 1. Command Line Interface (CLI)
Run the parser directly on a PDF file:

```bash
python policy_parser.py path/to/your/policy.pdf
```

- **Output**: By default, it creates a `<filename>_output.json` in the same directory.
- **Custom Output**: `python policy_parser.py policy.pdf --output result.json`

### 2. REST API Server
Start the FastAPI server:

```bash
uvicorn api_server:app --reload --port 8000
```

The API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Endpoints:
- `GET /health`: Check service status.
- `POST /parse`: Upload a PDF file to receive structured JSON.
- `POST /parse/text`: Send raw text in a JSON body to parse into segments.

---

## 📊 Sample Output Format

The parser produces a JSON array of segments, each containing:

```json
{
  "file_name": "example.pdf",
  "clause_id": "1.2",
  "level": "sub-clause",
  "parent_clause": "1",
  "title": "Governance Framework",
  "text": "The bank shall maintain a robust governance framework...",
  "keywords": ["governance", "bank", "framework", "robust"],
  "context": "The bank shall maintain a robust governance framework. This includes regular audits."
}
```

---

## 📁 Project Structure

- `policy_parser.py`: The core logic for parsing, segmentation, and NLP.
- `api_server.py`: FastAPI implementation with endpoints for PDF and text parsing.
- `tika-server-standard-3.1.0.jar`: Local binary for Apache Tika (ensures no internet dependency for parsing).
- `run_api_tests.py`: Utility script for validating API endpoints.
- `requirements.txt`: Python package dependencies.

---

## 🧠 Technical Highlights

- **Regex-based Parsing**: Custom regular expressions handle various bank policy formatting styles (e.g., "Section 1", "A-1", "1.1.2").
- **Noise Filtering**: Automatically filters out OCR artefacts, page numbers, and calendar years from being misidentified as clauses.
- **Tika Integration**: Configured to run against a local JAR for performance and stability.

---

## 📝 License
This project is for internal use for bank policy analysis.
