"""
BaseClauseSegmenter — Abstract base class for policy clause segmentation.

Defines the interface that all concrete segmenter implementations must satisfy.
A future LLM-only segmenter, rule-only segmenter, or document-type-specific
segmenter can be added by inheriting this class.

Inheritance hierarchy:
    BaseClauseSegmenter (ABC)
        └── ClauseSegmenter  — Regex-first + Mistral LLM fallback implementation

Usage::

    class MyLLMOnlySegmenter(BaseClauseSegmenter):
        def segment(self, document):
            ...
        def _segment_section(self, section):
            ...
"""
from abc import ABC, abstractmethod
from typing import List, Any


class BaseClauseSegmenter(ABC):
    """
    Abstract base class for clause segmentation.

    Defines the two-method contract:
        segment()          — Public API: convert a full document into clauses.
        _segment_section() — Internal: segment a single document section.

    The two-level design (document → sections → clauses) gives subclasses
    the flexibility to override only section-level logic while inheriting
    document-level orchestration.
    """

    @abstractmethod
    def segment(self, document: Any) -> List[Any]:
        """
        Convert a ParsedDocument into an ordered flat list of PolicyClause objects.

        Args:
            document: Output of SimplePDFParser.parse_pdf() — a ParsedDocument.

        Returns:
            List[PolicyClause] with ``file_id`` (``ParsedDocument.document_id``),
            ``clause_id`` / sub-clause ids, ``filename``, ``policy_id``, and fields
            consumed by :class:`app.common.schemas.ClauseVectorMetadata` for indexing.
        """
        pass

    @abstractmethod
    def _segment_section(self, section: Any) -> List[Any]:
        """
        Segment a single DocumentSection into a list of PolicyClause objects.

        This is the core segmentation unit. Subclasses implement their
        detection strategy (regex, LLM, hybrid) here.

        Args:
            section: A DocumentSection from the parsed document.

        Returns:
            List[PolicyClause] extracted from this section.
        """
        pass
