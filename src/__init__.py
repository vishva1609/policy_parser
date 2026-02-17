"""Simplified file parsing package."""
from .pipeline import DocumentPipeline
from .excel_generator import export_compliance_questions_to_excel, clean_text_for_excel

__version__ = "0.1.0"
__all__ = ["DocumentPipeline", "export_compliance_questions_to_excel", "clean_text_for_excel"]
