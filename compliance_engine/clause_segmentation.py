import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalises whitespace and newlines."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Clause Segmentation
# ---------------------------------------------------------------------------
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

_FAKE_HEADING_RE = re.compile(r'^\d+\.\s+[a-z]')
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')
_NOISE_RE = re.compile(r'[^\w\s.:\-]')

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
    clause_stack: list[tuple[str, int]] = []

    current_id = ''
    current_title = ''
    current_text_lines: list[str] = []

    def _parent_of(dep: int) -> str | None:
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

        bullet_m = _BULLET_RE.match(line)
        if bullet_m:
            current_text_lines.append(stripped)
            continue

        heading_m = _HEADING_RE.match(stripped)

        if heading_m and _FAKE_HEADING_RE.match(stripped):
            heading_m = None

        if heading_m:
            cid_candidate = heading_m.group(1)
            rest = stripped[heading_m.end():].strip()

            is_single_int = bool(re.match(r'^\d+$', cid_candidate))

            if is_single_int and rest and rest[0].islower():
                heading_m = None
            if heading_m and _YEAR_RE.match(cid_candidate):
                heading_m = None
            if heading_m and is_single_int and int(cid_candidate) > 30 and not rest:
                heading_m = None
            if heading_m and is_single_int and rest:
                first_char = rest[0]
                if not (first_char.isupper() or first_char.isdigit() or first_char in '"\'('):
                    heading_m = None
            if heading_m and is_single_int and rest:
                if len(rest.split()) > 7:
                    heading_m = None
            if heading_m and is_single_int and rest and rest.rstrip().endswith(':'):
                heading_m = None
            if heading_m and is_single_int and not rest:
                heading_m = None
            if heading_m and rest and _NOISE_RE.search(rest[:20]):
                heading_m = None

        if heading_m:
            _flush()
            current_id = heading_m.group(1)
            current_title = stripped[heading_m.end():].strip()
            current_text_lines = []
            continue

        current_text_lines.append(stripped)
        if not current_id:
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
