"""
ClauseSegmenter — Concrete implementation of BaseClauseSegmenter.

Extracts structured policy clauses from parsed documents using:
    1. Regex detection of numbered clauses, headings, and bullet points.
    2. LLM fallback (Mistral via MistralClient) for large unstructured blocks.

Now uses MistralClient from app/common/llm_client.py — eliminating the
duplicated LLM chain setup that previously existed here and in clause_comparator.py.
"""
import re
import os
import time
from typing import List, Optional, Tuple

from .base_clause_segmenter import BaseClauseSegmenter
from ..common.schemas import PolicyClause, ParsedDocument, DocumentElement, DocumentSection
from ..common.llm_client import MistralConfig, LANGCHAIN_AVAILABLE

try:
    from langchain_core.output_parsers import JsonOutputParser
except ImportError:
    pass


# ── Regex patterns ────────────────────────────────────────────────────────────

NUMBERED_CLAUSE_PATTERNS: List[Tuple[str, str]] = [
    (r'^(\d+(?:\.\d+){2,})\s+(.+)',                    'deep_sub'),
    (r'^(\d+(?:\.\d+)+)\s+(.+)',                       'sub_section'),
    (r'^(\d+\.)\s+([A-Z].+)',                          'section'),
    (r'^([A-Z]\.)\s+(.+)',                             'alpha'),
    (r'^(?:Section|Clause|Article)\s+(\d+[\.\d]*)\s*[:\-]?\s*(.+)', 'named'),
    (r'^(\d{1,2})\s+([A-Z][A-Za-z ]{2,59})$',         'bare_num'),
]

BULLET_PATTERNS = [
    r'^[•\u2022\u2023\u25E6]\s+',
    r'^[-–—]\s+',
    r'^\*\s+',
    r'^[a-z]\)\s+',
    r'^\([a-z]\)\s+',
    r'^\([ivxlcdm]+\)\s+',
    r'^\d+\)\s+',
]

REQUIREMENT_KEYWORDS = ['must', 'shall', 'required', 'mandatory', 'will', 'need to', 'have to']

LLM_FALLBACK_CHARS = 1500

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'that', 'this', 'these', 'those', 'it', 'its', 'all', 'any',
    'each', 'every', 'both', 'either', 'neither', 'not', 'no',
    'such', 'other', 'than', 'then', 'so', 'if', 'when', 'where',
    'which', 'who', 'how', 'as', 'by', 'from', 'into', 'through',
}

_NOISE_RE = re.compile(
    r'(©|all rights reserved|^reference\s+[a-z0-9/()+_-]+'
    r'|^revision\s+\d|^public$|^policy$|^table of contents$|\.{5,})',
    re.IGNORECASE,
)

_LONE_BULLETS: frozenset = frozenset([
    '•', '\u2022', '\u2023', '\u25e6',
    '-', '–', '—',
    '*',
])

_STRUCTURAL_MARKER_RE = re.compile(
    r'^(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[ivxlcdm]+\.?)$',
    re.IGNORECASE,
)

_TOC_TITLE_RE = re.compile(r'^(?:table of contents|contents)$', re.IGNORECASE)


class ClauseSegmenter(BaseClauseSegmenter):
    """
    Segments a ParsedDocument into a flat list of PolicyClause objects.

    Concrete implementation of BaseClauseSegmenter:
        segment()          — public API (full document → clauses)
        _segment_section() — internal (one section → clauses)

    LLM chain is now built via MistralClient (app/common/llm_client.py)
    instead of duplicating initialisation code.
    """

    def __init__(
        self,
        policy_label: str = "A",
        policy_id: str = "",
        api_key: str = None,
        model: str = "mistral-small-latest",
        use_llm_fallback: bool = True,
        config: Optional[MistralConfig] = None,
    ):
        self.policy_label = policy_label
        self.policy_id = policy_id or f"policy_{policy_label.lower()}"
        self.use_llm_fallback = use_llm_fallback
        self._llm_chain = None

        if use_llm_fallback and LANGCHAIN_AVAILABLE:
            if config is not None:
                self._llm_chain = config.build_chain(
                    system_prompt=(
                        "You are a policy document analyst. Extract distinct clauses from "
                        "the given policy text. Respond ONLY with valid JSON, no extra text."
                    ),
                    output_parser=JsonOutputParser(),
                )
                print(
                    f"  ClauseSegmenter LLM fallback (shared MistralConfig): {config.model_name}"
                )
            else:
                api_key = api_key or os.getenv("MISTRAL_API_KEY")
                if api_key:
                    self._setup_llm(api_key, model)
                    print(f"  ClauseSegmenter LLM fallback enabled: {model}")
                else:
                    print("  ClauseSegmenter: no API key — LLM fallback disabled")

    # ── LLM setup via MistralClient ────────────────────────────────────────────

    def _setup_llm(self, api_key: str, model: str):
        """
        Build the LLM chain using the shared MistralConfig.

        Uses MistralConfig.build_chain() — the config object is injected
        instead of created internally. This mirrors how OpenAIConfig is
        created once in views and passed down to the service and models.
        """
        config = MistralConfig(
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
        self._llm_chain = config.build_chain(
            system_prompt=(
                "You are a policy document analyst. Extract distinct clauses from "
                "the given policy text. Respond ONLY with valid JSON, no extra text."
            ),
            output_parser=JsonOutputParser(),
        )

    # ── BaseClauseSegmenter contract ──────────────────────────────────────────

    def segment(self, document: ParsedDocument) -> List[PolicyClause]:
        """
        Convert a ParsedDocument into an ordered flat list of PolicyClause objects.

        Implements BaseClauseSegmenter.segment().

        Args:
            document: Output of SimplePDFParser.parse_pdf()

        Returns:
            List[PolicyClause] with auto-assigned clause_ids like "A-001", "A-002" …
        """
        print(f"\n[Segmenter] Policy {self.policy_label}: {document.filename}")
        print(f"  Sections to process: {len(document.sections)}")

        all_clauses: List[PolicyClause] = []

        for section in document.sections:
            section_clauses = self._segment_section(section)
            all_clauses.extend(section_clauses)

        file_id = document.document_id
        for i, clause in enumerate(all_clauses, 1):
            clause.clause_id = f"{self.policy_label}-{i:03d}"
            clause.filename = document.filename
            clause.policy_id = self.policy_id
            clause.file_id = file_id
            for j, sub in enumerate(clause.sub_clauses, 1):
                sub.clause_id = f"{self.policy_label}-{i:03d}-S{j:02d}"
                sub.filename = document.filename
                sub.policy_id = self.policy_id
                sub.file_id = file_id

        print(f"  → {len(all_clauses)} clauses extracted")
        return all_clauses

    def _segment_section(self, section: DocumentSection) -> List[PolicyClause]:
        """
        Segment a single DocumentSection into PolicyClause objects.

        Implements BaseClauseSegmenter._segment_section().
        """
        elements = self._preprocess_elements(section.elements)

        clauses: List[PolicyClause] = []
        current: Optional[PolicyClause] = None
        body_lines: List[str] = []

        for elem in elements:
            text = elem.text.strip()
            if not text:
                continue

            matched = self._match_numbered(text)
            if matched:
                if current:
                    current.text = "\n".join(body_lines).strip() or current.title
                    clauses.append(self._finalize(current))
                clause_num, clause_title = matched
                current = self._new_clause(clause_title, section.title, elem.page,
                                           clause_number=clause_num)
                body_lines = [clause_title]
                continue

            if self._is_bullet(text):
                bullet_text = self._strip_bullet(text)
                if current:
                    sub = self._new_clause(bullet_text, current.title, elem.page)
                    sub.text = bullet_text
                    sub.is_requirement = self._is_requirement(bullet_text)
                    sub.keywords = self._keywords(bullet_text)
                    current.sub_clauses.append(self._finalize(sub))
                    body_lines.append(bullet_text)
                else:
                    orphan = self._new_clause(bullet_text, section.title, elem.page)
                    orphan.text = bullet_text
                    clauses.append(self._finalize(orphan))
                continue

            if self._is_heading(text, elem):
                if current:
                    current.text = "\n".join(body_lines).strip() or current.title
                    clauses.append(self._finalize(current))
                current = self._new_clause(text, section.title, elem.page)
                body_lines = []
                continue

            if self._is_structural_marker(text):
                continue

            if current:
                body_lines.append(text)
                if self._is_requirement(text):
                    current.is_requirement = True
            else:
                if len(text) >= 50:
                    current = self._new_clause(
                        text[:80] + ("…" if len(text) > 80 else ""),
                        section.title, elem.page
                    )
                    body_lines = [text]

        if current:
            current.text = "\n".join(body_lines).strip() or current.title
            clauses.append(self._finalize(current))

        clauses = [c for c in clauses if not self._is_discardable_clause(c)]

        if self.use_llm_fallback and self._llm_chain:
            clauses = self._llm_refine(clauses, section)

        return clauses

    # ── Pattern helpers ────────────────────────────────────────────────────────

    def _match_numbered(self, text: str) -> Optional[Tuple[str, str]]:
        for pattern, _ in NUMBERED_CLAUSE_PATTERNS:
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                g = m.groups()
                return (g[0].strip(), g[1].strip()) if len(g) >= 2 else (g[0].strip(), text[len(g[0]):].strip())
        return None

    def _is_bullet(self, text: str) -> bool:
        return any(re.match(p, text) for p in BULLET_PATTERNS)

    def _strip_bullet(self, text: str) -> str:
        for p in BULLET_PATTERNS:
            cleaned = re.sub(p, '', text, count=1).strip()
            if cleaned != text:
                return cleaned
        return text.lstrip('•-–—*').strip()

    def _is_heading(self, text: str, elem: DocumentElement) -> bool:
        font_size = elem.metadata.get('font_size', 12)
        words = text.split()
        word_count = len(words)
        is_single_word_heading = (
            word_count == 1 and text[0].isupper()
            and len(text) >= 4 and not text.isdigit()
        )
        return (
            (font_size > 13 or text.isupper() or is_single_word_heading)
            and len(text) < 120
            and 1 <= word_count <= 10
            and not text.endswith('.')
        )

    def _is_structural_marker(self, text: str) -> bool:
        return bool(_STRUCTURAL_MARKER_RE.match(text))

    def _is_requirement(self, text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in REQUIREMENT_KEYWORDS)

    def _keywords(self, text: str, max_kw: int = 8) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        seen: set = set()
        kws: List[str] = []
        for w in words:
            w_lo = w.lower()
            if w_lo not in STOPWORDS and w_lo not in seen:
                kws.append(w_lo)
                seen.add(w_lo)
            if len(kws) >= max_kw:
                break
        return kws

    # ── PDF element preprocessing ──────────────────────────────────────────────

    def _preprocess_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        out: List[DocumentElement] = []
        i = 0
        n = len(elements)

        while i < n:
            elem = elements[i]
            text = elem.text.strip()

            if not text or _NOISE_RE.search(text):
                i += 1
                continue

            j = i + 1
            while j < n and not elements[j].text.strip():
                j += 1
            next_elem = elements[j] if j < n else None
            next_text = next_elem.text.strip() if next_elem else ""

            if text in _LONE_BULLETS:
                result = self._merge_bullet(elem, next_elem, next_text)
                if result:
                    out.append(result)
                    i = j + 1
                else:
                    i += 1
                continue

            if re.match(r'^\d{1,2}$', text) and next_elem:
                result, consumed = self._merge_section_number(text, elem, next_elem, next_text)
                if result:
                    out.append(result)
                i = (j + 1) if consumed else (i + 1)
                continue

            elements, n, did_merge = self._merge_fragment(
                elements, i, j, elem, text, next_elem, next_text, n
            )
            if did_merge:
                continue

            out.append(elem)
            i += 1

        return out

    @staticmethod
    def _merge_bullet(elem, next_elem, next_text) -> Optional[DocumentElement]:
        if next_elem and next_text:
            return DocumentElement(
                text=f"\u2022 {next_text}",
                element_type=elem.element_type,
                page=elem.page,
                metadata=dict(elem.metadata),
            )
        return None

    @staticmethod
    def _merge_section_number(text, elem, next_elem, next_text):
        if not next_text or _NOISE_RE.search(next_text):
            return None, False
        is_heading = (
            next_text[0].isupper()
            and len(next_text) < 80
            and len(next_text.split()) <= 6
            and not next_text.endswith('.')
        )
        if is_heading:
            return DocumentElement(
                text=f"{text}. {next_text}",
                element_type=next_elem.element_type,
                page=elem.page,
                metadata=dict(next_elem.metadata),
            ), True
        return None, False

    @staticmethod
    def _merge_fragment(elements, i, j, elem, text, next_elem, next_text, n):
        is_fragment = (
            next_elem and next_text and text
            and text[-1] not in '.!?:;'
            and text not in _LONE_BULLETS
            and next_text[0].islower()
            and next_text not in _LONE_BULLETS
            and not _NOISE_RE.search(next_text)
            and not re.match(r'^\d{1,2}$', next_text)
        )
        if not is_fragment:
            return elements, n, False
        merged = DocumentElement(
            text=f"{text} {next_text}",
            element_type=elem.element_type,
            page=elem.page,
            metadata=dict(elem.metadata),
        )
        elements = list(elements[:i]) + [merged] + list(elements[j + 1:])
        return elements, len(elements), True

    def _is_discardable_clause(self, clause: PolicyClause) -> bool:
        title = clause.title.strip()
        text = clause.text.strip()
        if not title and not text:
            return True
        if _TOC_TITLE_RE.match(title):
            return True
        if self._is_structural_marker(title) and len(text) <= 10:
            return True
        if self._is_structural_marker(text):
            return True
        if len(text) <= 3 and not clause.sub_clauses:
            return True
        return False

    def _new_clause(self, title, context, page, clause_number="") -> PolicyClause:
        return PolicyClause(
            clause_id="",
            title=title[:150],
            text="",
            keywords=[],
            context=context,
            page=page,
            category="General",
            is_requirement=False,
            clause_number=clause_number,
            policy_label=self.policy_label,
            filename="",
            policy_id=self.policy_id,
        )

    def _finalize(self, clause: PolicyClause) -> PolicyClause:
        if not clause.text:
            clause.text = clause.title
        if not clause.keywords:
            clause.keywords = self._keywords(clause.text)
        if not clause.is_requirement:
            clause.is_requirement = self._is_requirement(clause.text)
        return clause

    # ── LLM fallback ──────────────────────────────────────────────────────────

    def _llm_refine(self, clauses, section):
        refined = []
        for clause in clauses:
            if len(clause.text) > LLM_FALLBACK_CHARS and not clause.sub_clauses:
                llm_clauses = self._llm_segment(clause, section)
                refined.extend(llm_clauses if llm_clauses else [clause])
            else:
                refined.append(clause)
        return refined

    def _llm_segment(self, clause, section, max_retries=2):
        excerpt = clause.text[:2000]
        prompt = (
            f'Section: "{section.title}"\n\n'
            f'Policy text:\n"""\n{excerpt}\n"""\n\n'
            'Identify distinct policy clauses in this text.\n'
            'Return a JSON array:\n'
            '[\n'
            '  {"title": "short title", "text": "full clause text", "is_requirement": true}\n'
            ']\n'
            'Rules:\n'
            '- Only split where there are truly separate topics.\n'
            '- If it is already one clause, return a single-item array.\n'
            '- Do not invent content not present in the text.'
        )
        for attempt in range(max_retries):
            try:
                result = self._llm_chain.invoke({"input": prompt})
                if not isinstance(result, list):
                    return []
                new_clauses = []
                for item in result:
                    if isinstance(item, dict) and item.get('text'):
                        c = self._new_clause(
                            item.get('title', item['text'][:80]),
                            section.title,
                            clause.page,
                            clause.clause_number,
                        )
                        c.text = item['text']
                        c.is_requirement = item.get('is_requirement', self._is_requirement(item['text']))
                        new_clauses.append(self._finalize(c))
                if new_clauses:
                    print(f"    LLM split 1 clause → {len(new_clauses)}")
                    return new_clauses
            except Exception as e:
                print(f"    LLM fallback error (attempt {attempt+1}): {str(e)[:80]}")
                time.sleep(3 * (attempt + 1))
        return []
