"""
Service package — Orchestration and factory layer.

Mirrors the service/ layer in archit1012/qa-bot-llm:
    qa_apis_service.py  →  document_service.py + comparison_service.py

The service layer:
    1. Picks the right model implementation at runtime (factory pattern)
    2. Orchestrates the full pipeline (parse → segment → embed → compare → report)
    3. Has no HTTP/CLI concerns — those belong in views/
"""
from .document_service import DocumentService
from .comparison_service import PolicyComparisonPipeline
from .comparison_http_service import run_policy_compare, default_pipeline_options_from_config

__all__ = [
    "DocumentService",
    "PolicyComparisonPipeline",
    "run_policy_compare",
    "default_pipeline_options_from_config",
]
