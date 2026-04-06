"""
Models package — Abstract base classes (ABCs) and concrete implementations.

OOP Hierarchy:
    BaseDocumentProcessor (ABC)
        L-- SimplePDFParser       (loads PDFs)
        L-- DocumentChunker       (splits into chunks)

    BaseEmbedder (ABC)
        L-- EmbeddingPipeline     (ChromaDB + sentence-transformers)

    BaseClauseSegmenter (ABC)
        L-- ClauseSegmenter       (regex + LLM fallback)

    PolicyAnalyzer                (keyword-based policy categorization)
    QuestionGenerator             (LLM compliance question generation)
    AutoTuner / AdaptiveTuner     (hyperparameter optimization)

Clause / vector metadata types live in ``app.common.schemas`` (e.g. ClauseVectorMetadata).
"""
from .base_document_processor import BaseDocumentProcessor
from .base_embedder import BaseEmbedder
from .base_clause_segmenter import BaseClauseSegmenter
from .document_processor import SimplePDFParser, DocumentChunker
from .json_document_processor import SimpleJSONParser
from .embedder import EmbeddingPipeline
from .clause_segmenter import ClauseSegmenter
from .clause_comparator import ClauseMatcher, ComplianceReasoner, GapDetector
from .compliance_scorer import ComplianceScorer
from .policy_analyzer import PolicyAnalyzer, PolicyStatement, PolicyCategory
from .question_generator import QuestionGenerator, AutoTuner, AdaptiveTuner

__all__ = [
    # ABCs
    "BaseDocumentProcessor",
    "BaseEmbedder",
    "BaseClauseSegmenter",
    # Document processing
    "SimplePDFParser",
    "SimpleJSONParser",
    "DocumentChunker",
    "EmbeddingPipeline",
    # Clause pipeline
    "ClauseSegmenter",
    "ClauseMatcher",
    "ComplianceReasoner",
    "GapDetector",
    "ComplianceScorer",
    # Policy analysis
    "PolicyAnalyzer",
    "PolicyStatement",
    "PolicyCategory",
    # Question generation & tuning
    "QuestionGenerator",
    "AutoTuner",
    "AdaptiveTuner",
]
