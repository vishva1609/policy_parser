"""
Views package — Entry-point / runner layer.

Mirrors the views/ layer in archit1012/qa-bot-llm:
    qa_apis.py         →  pipeline_runner.py (CLI) + qa_apis.py (Flask /upload)
    (FastAPI compare)  →  comparison_api.py (POST /compare/policies)
    (FastAPI sessions) →  sessions_api.py (POST /sessions, steps/parse|segment|…)

The views layer:
    1. Handles CLI arguments / script entry-points
    2. Calls the service layer
    3. Has no business logic of its own
"""
from .pipeline_runner import PipelineRunner

__all__ = ["PipelineRunner"]
