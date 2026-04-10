import os
import re
import json
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import argparse
import pdfplumber

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# 1. PDF Parsing using PDFPlumber
# ---------------------------------------------------------------------------
def extract_text_with_pdfplumber(file_path):
    """Extracts raw text from a PDF file using PDFPlumber.
    This replaces Tika to ensure full Windows compatibility and cleaner text
    without needing a background Java process or MD5 checks.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error parsing PDF with pdfplumber: {e}")
    return text

# Legacy alias for backward compatibility for the API server or tests
def extract_text_with_tika(file_path):
    return extract_text_with_pdfplumber(file_path)


# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    """Normalises whitespace and newlines."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Clause Segmentation – improved
# ---------------------------------------------------------------------------

# A *real* section heading must have a **word** after the numbering,
# not just end the line or be followed by another digit.
# Accepted patterns:
#   "3"            – top-level section number appearing alone on a line
#   "3.1"          – sub-section: digit(s).digit(s)
#   "3.1.2"        – sub-sub-section
#   "Section 3"    – explicit keyword
#   "A-1", "B-2"   – letter-dash-digit
#
# Critically we REJECT lines where the number is immediately followed by
# ". " and then more lowercase prose (those are inline list items inside body text).
_HEADING_RE = re.compile(
    r'^'
    r'('
    r'Section\s+\d[\d.]*'           # "Section 3" / "Section 3.1"
    r'|[A-Z]-\d[\d.]*'              # "A-1", "B-2"
    r'|\d+(?:\.\d+)+'               # "3.1", "3.1.2" – MUST have at least one dot
    r'|\d+'                         # single integer  – only when standalone
    r')'
    r'(?:\s*[.:]?\s+|\s*$)'         # followed by optional "."/":"  then whitespace, or end-of-line
    r'(?![\da-z])',                  # NOT immediately followed by digit or lowercase letter
    re.MULTILINE,
)

# Additional guard: headings at the single-integer level must span the
# whole line (i.e. the rest of the line is the title, not inline prose).
# We detect "fake" headings as: digit followed by ". " and a lowercase word.
_FAKE_HEADING_RE = re.compile(r'^\d+\.\s+[a-z]')

# 4-digit years (e.g. "2024", "2022") must never be clause IDs.
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')

# Garbled / noise line: contains non-word chars or uncommon letter patterns
# suggesting OCR junk rather than a real section number.
_NOISE_RE = re.compile(r'[^\w\s.:\-]')

# Bullet patterns
_BULLET_RE = re.compile(
    r'^(\s*)'
    r'([\u2022\u2013\u2014\*\-\–]'   # symbol bullets: •, –, —, *, -, –
    r'|[a-z]\)'                        # a) b) c)
    r'|[ivxlc]+\)'                     # i) ii) iii) (roman numerals)
    r'|\([0-9]+\)'                     # (1) (2)
    r')'
    r'\s+',
)


def _heading_level(clause_id: str) -> int:
    """Return nesting depth: 'Section 3' or '3' → 1, '3.1' → 2, '3.1.2' → 3 …"""
    if clause_id.startswith('Section') or re.match(r'^[A-Z]-', clause_id):
        return 1
    return len(clause_id.split('.'))


def segment_clauses(text: str, file_name: str = '') -> list:
    """
    Segments the text into clauses, sub-clauses and bullet points.
    Returns a list of dicts conforming to the expected output schema.
    """
    lines = text.split('\n')
    results = []

    # Stack tracks open clause IDs so we can derive parent correctly.
    # Each entry: (clause_id, depth)
    clause_stack: list[tuple[str, int]] = []

    current_id = ''
    current_title = ''
    current_text_lines: list[str] = []

    def _parent_of(dep: int) -> str | None:
        """Find the nearest ancestor with depth < dep."""
        for cid, cdep in reversed(clause_stack):
            if cdep < dep:
                return cid
        return None

    def _flush():
        nonlocal current_id, current_title, current_text_lines
        if not current_id:
            current_text_lines = []
            return
        full_text = '\n'.join(current_text_lines).strip()
        if not full_text:
            current_text_lines = []
            return

        dep = _heading_level(current_id)
        parent = _parent_of(dep)

        entry: dict = {
            'file_name': file_name,
            'clause_id': current_id,
            'level': _depth_to_level(dep),
            'title': current_title.strip(),
            'text': full_text,
        }
        if parent:
            entry['parent_clause'] = parent
        entry['keywords'] = extract_keywords(full_text)
        entry['context'] = generate_context(full_text)
        results.append(entry)

        # Update stack: pop everything at same or deeper level, then push current
        while clause_stack and clause_stack[-1][1] >= dep:
            clause_stack.pop()
        clause_stack.append((current_id, dep))

        current_id = ''
        current_title = ''
        current_text_lines = []

    def _depth_to_level(d: int) -> str:
        if d == 1:
            return 'clause'
        if d == 2:
            return 'sub-clause'
        return 'sub-sub-clause'

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # ── Bullet? ──────────────────────────────────────────────────────────
        bullet_m = _BULLET_RE.match(line)
        if bullet_m:
            # Bullets accumulate into the current clause's text body so that
            # they are associated with it rather than creating orphan entries.
            # Only split them out if it would help context.
            current_text_lines.append(stripped)
            continue

        # ── Heading? ─────────────────────────────────────────────────────────
        heading_m = _HEADING_RE.match(stripped)

        # Reject if it looks like inline prose: "3. The bank shall …"
        if heading_m and _FAKE_HEADING_RE.match(stripped):
            heading_m = None

        # Reject single-integer "headings" whose title starts with lowercase
        # (almost certainly an inline numbered list item, not a real section)
        if heading_m:
            cid_candidate = heading_m.group(1)
            rest = stripped[heading_m.end():].strip()

            is_single_int = bool(re.match(r'^\d+$', cid_candidate))

            # Reject inline prose: "3. the bank shall …"
            if is_single_int and rest and rest[0].islower():
                heading_m = None

            # Reject 4-digit calendar years
            if heading_m and _YEAR_RE.match(cid_candidate):
                heading_m = None

            # Reject bare large integers (> 30) with no useful title —
            # these are almost always page numbers or OCR artefacts.
            if heading_m and is_single_int and int(cid_candidate) > 30 and not rest:
                heading_m = None

            # Single integers with a non-empty title must start with
            # a capital letter, digit, or quote — not OCR junk characters.
            if heading_m and is_single_int and rest:
                first_char = rest[0]
                if not (first_char.isupper() or first_char.isdigit() or first_char in '"\'('):
                    heading_m = None

            # If the title after the number is a long sentence (> 7 words) it's
            # almost certainly inline body text, not a section heading.
            if heading_m and is_single_int and rest:
                if len(rest.split()) > 7:
                    heading_m = None

            # Titles ending with ":" are typically intro sentences, not headings.
            # e.g. "2. The key elements for breach management are:"
            if heading_m and is_single_int and rest and rest.rstrip().endswith(':'):
                heading_m = None

            # Bare single integer with no title is a page number reference
            # (from the TOC or elsewhere). Drop it entirely.
            if heading_m and is_single_int and not rest:
                heading_m = None

            # Reject if the remainder of the line looks like OCR junk
            if heading_m and rest and _NOISE_RE.search(rest[:20]):
                heading_m = None

        if heading_m:
            _flush()
            current_id = heading_m.group(1)
            current_title = stripped[heading_m.end():].strip()
            current_text_lines = []
            continue

        # ── Body text ────────────────────────────────────────────────────────
        current_text_lines.append(stripped)
        # If there's no active clause yet, use first non-empty line as title
        if not current_id:
            # treat as a preamble block labelled by its first words
            current_id = f'PREAMBLE-{idx}'
            current_title = stripped[:80]

    _flush()
    return results


# ---------------------------------------------------------------------------
# 4. Keyword extraction
# ---------------------------------------------------------------------------
def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Frequency-based keyword extraction, stopwords removed."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha() and len(w) > 2 and w.lower() not in stop_words]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


# ---------------------------------------------------------------------------
# 5. Context generation
# ---------------------------------------------------------------------------
def generate_context(text: str) -> str:
    """Returns the first 1-2 sentences as a short summary."""
    sentences = sent_tokenize(text)
    if not sentences:
        return ''
    return ' '.join(sentences[:2])


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser_ = argparse.ArgumentParser(description="Policy PDF → structured JSON")
    parser_.add_argument('pdf_path', type=str, help='Path to the policy PDF file')
    parser_.add_argument('--output', type=str, default=None,
                         help='Output JSON file (default: <pdf_name>_output.json)')
    args = parser_.parse_args()

    pdf_path = args.pdf_path
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    file_name = os.path.basename(pdf_path)
    print(f"Extracting text from {file_name} …")
    raw_text = extract_text_with_tika(pdf_path)
    cleaned = clean_text(raw_text)

    print("Segmenting clauses …")
    segments = segment_clauses(cleaned, file_name=file_name)

    output_path = args.output or os.path.splitext(pdf_path)[0] + '_output.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(segments)} segments written to {output_path}")


if __name__ == "__main__":
    main()
