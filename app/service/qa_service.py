"""
QA upload orchestration — mirrors qa_apis_service.py from archit1012/qa-bot-llm.

Factory by extension, then document_loader → text_splitter → prepare_vectordb,
then RAG answers for each question in the uploaded questions JSON.
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..common.rag_qa import get_response_from_query
from ..common.llm_client import MistralConfig
from ..models.embedder import EmbeddingPipeline
from .document_service import DocumentService


def _uploads_dir() -> Path:
    base = Path(__file__).resolve().parents[1] / "data" / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def prepare_file_paths(request) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """
    Save multipart ``doc_file`` and ``question_file`` under app/data/uploads/.

    Returns:
        (doc_path, question_path, error_message). Paths are None on failure.
    """
    try:
        if "doc_file" not in request.files or "question_file" not in request.files:
            return None, None, "Both doc_file and question_file must be provided"
        doc = request.files["doc_file"]
        qf = request.files["question_file"]
        if not doc.filename or not qf.filename:
            return None, None, "Both files must have a filename"

        session = uuid.uuid4().hex
        dest = _uploads_dir() / session
        dest.mkdir(parents=True, exist_ok=True)

        doc_path = dest / Path(doc.filename).name
        q_path = dest / Path(qf.filename).name
        doc.save(str(doc_path))
        qf.save(str(q_path))
        return doc_path, q_path, None
    except Exception as exc:
        return None, None, str(exc)


def _load_questions(question_path: Path) -> List[str]:
    with open(question_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        nested = data.get("questions")
        if isinstance(nested, list):
            data = nested
        else:
            return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            q = item.get("question") or item.get("q")
            if q:
                out.append(str(q))
    return out


def process_qa_upload(
    request,
    llm_config: MistralConfig,
    embedding_model: Optional[str] = None,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Run the reference-repo flow: index document, answer each question via RAG.

    Args:
        request:    Flask request (multipart).
        llm_config: Injected MistralConfig (same role as ``chat`` in reference).
        embedding_model: Overrides default sentence-transformers model name.

    Returns:
        (answers_dict, error_message). answers_dict maps question → answer string.
    """
    doc_path, question_path, err = prepare_file_paths(request)
    if err:
        return None, err

    assert doc_path is not None and question_path is not None
    questions = _load_questions(question_path)
    if not questions:
        return None, "No questions found in question_file (expected JSON list or objects with 'question')"

    ext = doc_path.suffix.lower()
    if ext not in DocumentService._SUPPORTED_FORMATS:
        return None, f"Unsupported document extension: {ext}"

    session_dir = doc_path.parent
    db_path = session_dir / "vectordb"
    model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    embedder = EmbeddingPipeline(model_name=model_name, db_path=str(db_path))
    service = DocumentService(
        output_dir=str(session_dir),
        embedding_model=model_name,
        embedder=embedder,
    )

    try:
        proc = service.process(str(doc_path), output_format="txt", overwrite_files=True)
    except Exception as exc:
        return None, f"Document processing failed: {exc}"

    db = proc.get("vector_db")
    if db is None:
        return None, "Vector store not produced"

    answers: Dict[str, str] = {}
    for q in questions:
        try:
            response, _ = get_response_from_query(db, q, llm_config, depth=4)
            answers[q] = response
        except Exception as exc:
            answers[q] = f"Error generating answer: {exc}"

    return answers, None


def mistral_config_from_env() -> Optional[MistralConfig]:
    """Build shared MistralConfig from config.yaml + env (same as FastAPI lifespan)."""
    from ..common.llm_client import load_mistral_config_optional

    return load_mistral_config_optional()
