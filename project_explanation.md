# Policy Parser — Complete Project Explanation

## 🏗️ Project Structure (2 main files)

```
c:\policy_parser\
├── policy_parser.py          ← CORE ENGINE (all logic lives here)
├── api_server.py             ← WEB API WRAPPER (exposes engine via HTTP)
├── tika-server-standard-3.1.0.jar  ← PDF extraction engine (Java)
├── bank_policy.pdf           ← Sample input PDF
└── bank_output.json          ← Sample output JSON
```

---

## 🔄 How the Whole System Works (Big Picture)

```
User/Postman
     │
     │  HTTP Request (PDF file or raw text)
     ▼
┌─────────────────────────────────────────┐
│           api_server.py                 │  ← FastAPI Web Server
│   Receives the request, validates it    │    (runs on port 8000)
└──────────────┬──────────────────────────┘
               │ calls functions from
               ▼
┌─────────────────────────────────────────┐
│          policy_parser.py               │  ← Core Engine
│                                         │
│  Step 1: extract_text_with_tika()       │  ← PDF → Raw Text
│  Step 2: clean_text()                   │  ← Raw Text → Clean Text
│  Step 3: segment_clauses()              │  ← Clean Text → JSON Segments
│  Step 4: extract_keywords()             │  ← Per-clause keywords
│  Step 5: generate_context()             │  ← Per-clause 1-2 sentence summary
└─────────────────────────────────────────┘
               │
               ▼
        JSON Response sent back to user
```

---

## 📄 FILE 1: `policy_parser.py` — The Core Engine

This file has **5 independent functions**, each doing one job.

---

### STEP 1 — PDF Text Extraction (`extract_text_with_tika`)
**Lines 26–30 in policy_parser.py**

```python
def extract_text_with_tika(file_path):
    parsed = tika_parser.from_file(file_path)   # ← Apache Tika reads the PDF
    content = parsed.get('content') or ''        # ← gets the raw text from it
    return content
```

**What happens here:**
- Apache **Tika** (a Java library) opens the PDF file
- It reads every page and extracts all the text as a single big string
- This is like copy-pasting all text from a PDF manually, but automatic
- The local JAR file `tika-server-standard-3.1.0.jar` does the actual PDF reading
- Tika is set up at lines 10–16 using the local JAR so it never downloads anything from internet

**Input:** Path to a PDF file (e.g. `bank_policy.pdf`)  
**Output:** One big string of all text from the PDF

---

### STEP 2 — Text Cleaning (`clean_text`)
**Lines 36–41 in policy_parser.py**

```python
def clean_text(text):
    text = re.sub(r'\r\n|\r', '\n', text)      # fix Windows line endings
    text = re.sub(r'\n{3,}', '\n\n', text)     # collapse 3+ blank lines → 2
    text = re.sub(r'[ \t]+', ' ', text)        # collapse multiple spaces → 1
    return text.strip()
```

**What happens here:**
- PDFs extracted by Tika have messy whitespace, extra blank lines, mixed line endings
- This function normalizes all of that into clean consistent text
- Makes the next step (segmentation) much more reliable

**Input:** Raw messy text from Tika  
**Output:** Clean normalized text

---

### STEP 3 — Clause Segmentation (`segment_clauses`)
**Lines 103–253 in policy_parser.py** ← THIS IS THE MOST IMPORTANT FUNCTION

This is the brain of the project. It reads through every line and decides:
- Is this line a **section heading** (like `3.1 Introduction`)? → Start a new clause
- Is this line a **bullet point** (like `- All employees must comply`)? → Add to current clause
- Is this line **body text**? → Add to current clause

```
Text input:
─────────────────────────────────
3.1 Introduction                  ← Heading detected → new segment starts
This policy applies to all staff. ← Body text → added to 3.1's text
- All employees must comply.      ← Bullet → added to 3.1's text
3.2 Data Retention                ← New heading → flush 3.1, start 3.2
Data must be kept for 7 years.    ← Body text → added to 3.2's text
─────────────────────────────────
```

**How headings are detected — `_HEADING_RE` (Lines 59–70):**

The regex matches patterns like:
| Pattern | Example | Meaning |
|---------|---------|---------|
| `\d+\.\d+` | `3.1`, `3.1.2` | Sub-clause with dot |
| `\d+` | `3` | Top-level clause number |
| `Section \d+` | `Section 3` | Named section |
| `[A-Z]-\d+` | `A-1`, `B-2` | Lettered section |

**Anti-false-positive guards (Lines 185–235):**

The code has 7 safety checks to avoid treating normal text as a heading:

| Guard | Example it blocks |
|-------|-------------------|
| Lowercase after number | `3. the bank shall...` |
| 4-digit years | `2024` being a clause ID |
| Long sentence titles | `2 The data officer shall perform...` (>7 words) |
| Colon-ending titles | `2. Key elements are:` |
| Bare integer no title | `24` alone on a line = page number |
| OCR noise | `32 e.se. 4O` = garbled text |

**Hierarchy tracking — `clause_stack` (Lines 113, 152–155):**

```
When 3.1 is found → stack = [(3.1, depth=2)]
When 3.2 is found → pop 3.1, push 3.2
When 3.2.1 is found → stack = [(3.2, depth=2), (3.2.1, depth=3)]
```
This is how `parent_clause` is set correctly for each segment.

**Output:** List of structured dicts, one per clause:
```json
{
  "file_name": "bank_policy.pdf",
  "clause_id": "3.1",
  "level": "sub-clause",
  "title": "Introduction",
  "text": "This policy applies to all staff.\n- All employees must comply.",
  "parent_clause": "3",
  "keywords": ["policy", "staff", "comply"],
  "context": "This policy applies to all staff."
}
```

---

### STEP 4 — Keyword Extraction (`extract_keywords`)
**Lines 259–265 in policy_parser.py**

```python
def extract_keywords(text, top_n=5):
    stop_words = set(stopwords.words('english'))    # ← removes "the", "a", "is"...
    words = word_tokenize(text)                      # ← splits into individual words
    words = [w.lower() for w in words               # ← lowercase
             if w.isalpha()                          # ← only real words (no numbers)
             and len(w) > 2                          # ← no short words like "it"
             and w.lower() not in stop_words]        # ← exclude stopwords
    freq = Counter(words)                            # ← count frequency of each word
    return [w for w, _ in freq.most_common(top_n)]  # ← return top 5 most frequent
```

**What it does:** Finds the 5 most important words in each clause by frequency.  
Uses NLTK (Natural Language Toolkit) library.

---

### STEP 5 — Context Generation (`generate_context`)
**Lines 271–276 in policy_parser.py**

```python
def generate_context(text):
    sentences = sent_tokenize(text)    # ← splits text into sentences
    return ' '.join(sentences[:2])     # ← returns just the first 2 sentences
```

**What it does:** Creates a short 1-2 sentence summary of each clause.  
This becomes the `"context"` field in the JSON output.

---

## 🌐 FILE 2: `api_server.py` — The Web API Wrapper

This file **wraps** the core engine functions into HTTP endpoints using **FastAPI**.

### How FastAPI works here:
```python
from policy_parser import extract_text_with_tika, clean_text, segment_clauses
```
Line 21 imports the 3 key functions from policy_parser.py.

---

### Endpoint 1: `GET /health` (Lines 48–51)
```python
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "policy-parser"}
```
**Purpose:** Just checks if the server is alive. Always returns 200 OK.  
**No logic** — just a ping/pong.

---

### Endpoint 2: `POST /parse` — PDF Upload (Lines 54–119)

```
User sends PDF file
        ↓
[1] Validate: must be .pdf extension  (line 73)
        ↓
[2] Save to temp file with unique name (line 80-88)
    _tmp_a3f9b2c1_bank_policy.pdf
        ↓
[3] Call extract_text_with_tika()     (line 91)  ← PDF TEXT EXTRACTION
        ↓
[4] Check text is not empty           (line 92-97)
        ↓
[5] Call clean_text()                 (line 99)  ← CLEAN TEXT
        ↓
[6] Call segment_clauses()            (line 100) ← BUILD JSON SEGMENTS
        ↓
[7] Return JSON response              (line 102)
        ↓
[8] Delete temp file (always, even on error) (line 118)
```

**Why temp file?** Because the uploaded file comes as bytes in memory. Tika needs a real file on disk to read. The `uuid4()` makes the filename unique so 2 users uploading at the same time don't overwrite each other.

---

### Endpoint 3: `POST /parse/text` — Raw Text (Lines 122–152)

```
User sends JSON: {"text": "3.1 Introduction\n...", "file_name": "test.txt"}
        ↓
[1] Validate: text must not be empty   (line 135)
        ↓
[2] Call clean_text()                  (line 139) ← CLEAN TEXT
        ↓
[3] Call segment_clauses()             (line 140) ← BUILD JSON SEGMENTS
        ↓
[4] Return JSON response               (line 142)
```

**Simpler than PDF** — no Tika needed since text is already provided.  
Skips Step 1 (extraction) and goes straight to Step 2 (cleaning).

---

## 📊 Complete Data Flow Diagram

```
                     ┌─────────────────────────────────────┐
                     │           api_server.py              │
                     │  uvicorn runs this on port 8000      │
                     └──────┬────────────┬─────────────────┘
                            │            │
              ┌─────────────┘            └──────────────────┐
              ▼                                             ▼
   POST /parse (PDF)                          POST /parse/text (raw text)
              │                                             │
              │ 1. Save to temp file                        │
              │                                             │
              ▼                                             │
   extract_text_with_tika()                                 │
   [policy_parser.py line 26]                               │
   Uses: tika-server-standard-3.1.0.jar                     │
              │                                             │
              ▼                                             ▼
         clean_text()  ◄─────────────────────── clean_text()
   [policy_parser.py line 36]                [policy_parser.py line 36]
              │                                             │
              └─────────────────┬───────────────────────────┘
                                ▼
                      segment_clauses()
                   [policy_parser.py line 103]
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
             For each clause detected:
             extract_keywords()    generate_context()
           [line 259]              [line 271]
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌──────────────────────┐
                    │   JSON Response      │
                    │ {                    │
                    │   "file_name": ...,  │
                    │   "total_segments":  │
                    │   "segments": [...]  │
                    │ }                    │
                    └──────────────────────┘
```

---

## 🎯 Where Exactly PDF Extraction Happens

| What | File | Line | Function |
|------|------|------|----------|
| **PDF → Text** | `policy_parser.py` | **28** | `tika_parser.from_file(file_path)` |
| **Text cleanup** | `policy_parser.py` | **38–41** | `clean_text()` |
| **Text → Segments** | `policy_parser.py` | **108–252** | `segment_clauses()` |
| **API receives PDF** | `api_server.py` | **87–88** | `shutil.copyfileobj()` saves to disk |
| **API calls Tika** | `api_server.py` | **91** | `extract_text_with_tika(tmp_path)` |
| **API calls segmenter** | `api_server.py` | **100** | `segment_clauses(cleaned, ...)` |

---

## 🔧 Libraries Used and Why

| Library | Purpose |
|---------|---------|
| **Apache Tika** | PDF text extraction — industry standard, handles complex PDFs |
| **NLTK** | Natural language: word tokenization, sentence splitting, stopwords |
| **FastAPI** | Web framework — auto-generates Swagger docs at `/docs` |
| **Uvicorn** | ASGI web server that runs the FastAPI app |
| **re (regex)** | Pattern matching for detecting section headings |
| **uuid** | Generates unique temp filenames for concurrent PDF uploads |
