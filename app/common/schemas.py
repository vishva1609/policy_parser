"""
Simple document schemas using Pydantic.

Canonical location: app/common/schemas.py
Also re-exported via src/schemas.py for backward compatibility.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentElement(BaseModel):
    """A document element (paragraph, heading, etc.)."""
    text: str
    element_type: str  # "heading", "paragraph", "list", "table"
    page: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A document section with elements."""
    title: str
    level: int
    elements: List[DocumentElement]
    page_start: int
    page_end: int


class ParsedDocument(BaseModel):
    """Complete parsed document."""
    document_id: str
    filename: str
    total_pages: int
    sections: List[DocumentSection]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    """Enriched document chunk."""
    chunk_id: str
    document_id: str
    text: str
    chunk_type: str
    parent_context: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[int] = Field(default_factory=list)


class EmbeddedChunk(BaseModel):
    """Chunk with embedding vector."""
    chunk_id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Policy Comparison Schemas
# ============================================================

class ComplianceStatus(str, Enum):
    """Result of comparing one clause against another policy."""
    COMPLIANT = "Compliant"
    PARTIAL = "Partial"
    NON_COMPLIANT = "Non-Compliant"
    MISSING = "Missing"


class PolicyClause(BaseModel):
    """A single structured policy clause extracted from a document."""
    clause_id: str = ""
    title: str
    text: str
    sub_clauses: List['PolicyClause'] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    context: str = ""          # Parent section title
    page: int = 0
    category: str = "General"
    is_requirement: bool = False   # Contains must/shall?
    clause_number: str = ""        # Detected number e.g. "5.1.2"
    policy_label: str = ""         # "A" or "B"
    filename: str = ""             # Source document filename e.g. "Policy_A.pdf"
    policy_id: str = ""            # Stable policy identifier e.g. "policy_a"
    file_id: str = ""              # ParsedDocument.document_id — stable file instance id

    model_config = {"arbitrary_types_allowed": True}


PolicyClause.model_rebuild()


class ClauseVectorMetadata(BaseModel):
    """
    Metadata persisted with each clause/sub-clause embedding for retrieval and audit.

    Single place to define Chroma-facing fields (OOP): build from :class:`PolicyClause`
    via ``from_parent_clause`` / ``from_sub_clause``, then ``to_flat_metadata()`` for
    :class:`DocumentChunk` / indexing.
    """

    file_id: str = ""
    filename: str = ""
    policy_id: str = ""
    policy_label: str = ""
    clause_id: str = ""
    sub_clause_id: str = ""
    parent_clause_id: str = ""
    is_sub_clause: bool = False
    clause_number: str = ""
    clause_title: str = ""
    section_title: str = ""
    category: str = ""
    is_requirement: bool = False
    keywords: List[str] = Field(default_factory=list)
    page: int = 0
    chunk_type: str = "clause"
    sub_clause_count: int = 0
    # Traceability (run/session scoped)
    session_id: str = ""
    indexed_at: str = ""  # ISO timestamp
    source_policy: str = ""  # "A" / "B" (redundant but convenient filter)
    source_pdf_name: str = ""
    source_page_start: int = 0
    source_page_end: int = 0
    text_sha256: str = ""

    @classmethod
    def from_parent_clause(cls, clause: PolicyClause) -> "ClauseVectorMetadata":
        return cls(
            file_id=clause.file_id or "",
            filename=clause.filename or "",
            policy_id=clause.policy_id or "",
            policy_label=clause.policy_label or "",
            clause_id=clause.clause_id or "",
            sub_clause_id="",
            parent_clause_id="",
            is_sub_clause=False,
            clause_number=clause.clause_number or "",
            clause_title=clause.title or "",
            section_title=clause.context or "",
            category=clause.category or "General",
            is_requirement=bool(clause.is_requirement),
            keywords=list(clause.keywords),
            page=int(clause.page or 0),
            chunk_type="clause",
            sub_clause_count=len(clause.sub_clauses),
        )

    @classmethod
    def from_sub_clause(cls, parent: PolicyClause, sub: PolicyClause) -> "ClauseVectorMetadata":
        return cls(
            file_id=sub.file_id or parent.file_id or "",
            filename=sub.filename or parent.filename or "",
            policy_id=sub.policy_id or parent.policy_id or "",
            policy_label=sub.policy_label or parent.policy_label or "",
            clause_id=sub.clause_id or "",
            sub_clause_id=sub.clause_id or "",
            parent_clause_id=parent.clause_id or "",
            is_sub_clause=True,
            clause_number=sub.clause_number or parent.clause_number or "",
            clause_title=sub.title or "",
            section_title=parent.context or "",
            category=sub.category or parent.category or "General",
            is_requirement=bool(sub.is_requirement),
            keywords=list(sub.keywords),
            page=int(sub.page or parent.page or 0),
            chunk_type="sub_clause",
            sub_clause_count=0,
        )

    def to_flat_metadata(self) -> Dict[str, Any]:
        """Dict merged into ``DocumentChunk.metadata`` (Chroma-safe primitive values)."""
        import json

        d = self.model_dump()
        d["keywords"] = json.dumps(d["keywords"])
        return d


class ClauseMatch(BaseModel):
    """A candidate matching clause found via semantic search."""
    text: str
    clause_id: str = ""
    similarity_score: float
    section: str = ""
    page: int = 0
    filename: str = ""            # Source document filename of the matched clause
    clause_title: str = ""        # Title of the matched clause
    policy_label: str = ""        # "A" or "B" — which policy this match came from
    file_id: str = ""             # ParsedDocument.document_id of the matched file
    sub_clause_id: str = ""       # Empty if match is a parent clause row
    parent_clause_id: str = ""   # Parent clause id when match is a sub-clause
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceResult(BaseModel):
    """Full comparison result for a single source clause."""
    clause_a: PolicyClause
    best_match: Optional[ClauseMatch] = None
    all_candidates: List[ClauseMatch] = Field(default_factory=list)
    status: ComplianceStatus = ComplianceStatus.MISSING
    reason: str = ""
    gap_description: str = ""
    confidence: float = 0.0
    evidence_quote: str = ""
    direction: str = "A→B"


class ComplianceScore(BaseModel):
    """Aggregated compliance score for one direction."""
    overall_pct: float
    compliant_count: int
    partial_count: int
    non_compliant_count: int
    missing_count: int
    critical_gaps: int          # Gaps in mandatory (must/shall) clauses
    scores_by_category: Dict[str, float] = Field(default_factory=dict)
    total_clauses: int
    direction: str = "A→B"


class PolicyComparisonReport(BaseModel):
    """Full bidirectional compliance comparison report."""
    policy_a_name: str
    policy_b_name: str
    timestamp: str
    results_a_to_b: List[ComplianceResult] = Field(default_factory=list)
    results_b_to_a: List[ComplianceResult] = Field(default_factory=list)
    score_a_to_b: Optional[ComplianceScore] = None
    score_b_to_a: Optional[ComplianceScore] = None
