"""
Common package — shared schemas, LLM configuration, and Excel utilities.

Mirrors common/openapi.py from archit1012/qa-bot-llm:
    OpenAIConfig → MistralConfig (shared LLM config object)
"""
from .schemas import (
    DocumentElement, DocumentSection, ParsedDocument,
    DocumentChunk, EmbeddedChunk,
    ComplianceStatus, PolicyClause, ClauseVectorMetadata, ClauseMatch,
    ComplianceResult, ComplianceScore, PolicyComparisonReport,
)
# MistralConfig is the primary name (mirrors OpenAIConfig from reference repo)
# MistralClient is kept as a backward-compat alias
from .llm_client import MistralConfig, MistralClient
from .excel_utils import (
    ExcelGenerator,
    ComparisonReportGenerator,
    clean_text_for_excel,
    export_compliance_questions_to_excel,
)

__all__ = [
    # Schemas
    "DocumentElement", "DocumentSection", "ParsedDocument",
    "DocumentChunk", "EmbeddedChunk",
    "ComplianceStatus", "PolicyClause", "ClauseVectorMetadata", "ClauseMatch",
    "ComplianceResult", "ComplianceScore", "PolicyComparisonReport",
    # LLM config — MistralConfig is the canonical name (mirrors OpenAIConfig)
    "MistralConfig",
    "MistralClient",   # backward-compat alias
    # Excel
    "ExcelGenerator", "ComparisonReportGenerator",
    "clean_text_for_excel", "export_compliance_questions_to_excel",
]
