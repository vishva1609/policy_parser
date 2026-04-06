"""
Service helpers for FastAPI tool endpoints: semantic search + question generation.

Keeps routers thin; paths are validated against the repository root where needed.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .document_service import DocumentService
from ..models.embedder import EmbeddingPipeline
from ..models.policy_analyzer import PolicyAnalyzer
from ..models.question_generator import (
    QuestionGenerator,
    AdaptiveTuner,
    AutoTuner,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_output_dir(user_relative: str) -> Path:
    """
    Resolve a directory under the repo root (e.g. ``output``).
    Rejects path traversal.
    """
    root = _repo_root()
    base = (root / user_relative.strip().replace("\\", "/")).resolve()
    root_r = root.resolve()
    if not str(base).startswith(str(root_r)):
        raise ValueError("output_dir must stay within the project repository")
    return base


def run_semantic_search_on_pdf(
    pdf_path: Path,
    query: str,
    n_results: int,
    work_dir: Path,
) -> Dict[str, Any]:
    """
    Parse + chunk + embed one PDF in ``work_dir``, then run semantic search.

    Returns the same shape as :meth:`DocumentService.search`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingPipeline(
        db_path=str(work_dir / "vectordb"),
        collection_name="policy_search",
    )
    service = DocumentService(
        output_dir=str(work_dir),
        embedder=embedder,
    )
    service.process(str(pdf_path), output_format="txt", overwrite_files=True)
    return service.search(query.strip(), n_results=max(1, min(n_results, 50)))


def resolve_tuning_config_path(
    *,
    tuning_json_path: Optional[Path],
    use_cached_tuning: bool,
    tuning_model: Optional[str],
    output_dir_relative: str,
) -> Optional[Path]:
    """
    Prefer an explicit uploaded tuning JSON; else optional cache under output/tuning/.
    """
    if tuning_json_path is not None and tuning_json_path.is_file():
        return tuning_json_path
    if use_cached_tuning and tuning_model:
        out = safe_output_dir(output_dir_relative)
        p = AutoTuner.get_tuning_output_path(str(out), tuning_model)
        if p.is_file():
            return p
    return None


def run_questions_from_policy(
    pdf_path: Path,
    work_dir: Path,
    *,
    tuning_config_path: Optional[Path] = None,
    model_override: Optional[str] = None,
    questions_per_statement: int = 3,
    max_statements: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Chunk policy → analyze → generate compliance questions.

    If ``tuning_config_path`` points to AutoTuner JSON (``best_params`` + ``score``),
    :class:`AdaptiveTuner` supplies hyperparameters for :class:`QuestionGenerator`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    service = DocumentService(output_dir=str(work_dir))
    proc = service.process(str(pdf_path), output_format="txt", overwrite_files=True)
    chunks = proc["chunks"]

    analyzer = PolicyAnalyzer()
    statements = analyzer.analyze_document(chunks)
    requirements = analyzer.get_requirements(statements, min_confidence=0.5)

    hp: Dict[str, Any] = {}
    tuning_score: Optional[float] = None
    model = model_override or os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    if tuning_config_path and tuning_config_path.is_file():
        with open(tuning_config_path, encoding="utf-8") as f:
            raw = json.load(f)
        model = raw.get("model") or model
        adaptive = AdaptiveTuner(config_path=str(tuning_config_path))
        hp = adaptive.get_optimal_params(chunks=chunks)
        tuning_score = hp.get("score")

    generator = QuestionGenerator(
        model=model,
        temperature=float(hp.get("temperature", 0.3)),
        max_tokens=int(hp.get("max_tokens", 2048)),
        top_p=float(hp.get("top_p", 1.0)),
        frequency_penalty=float(hp.get("frequency_penalty", 0.0)),
        presence_penalty=float(hp.get("presence_penalty", 0.0)),
        seed=hp.get("seed"),
        top_k=hp.get("top_k"),
    )

    stmts = requirements
    if max_statements is not None and max_statements > 0:
        stmts = requirements[: max_statements]

    questions = generator.generate_questions_from_statements(
        stmts,
        questions_per_statement=questions_per_statement,
        batch_size=10,
    )
    exported = generator.export_questions(questions)

    out_json = work_dir / "questions_api.json"
    generator.save_questions(questions, str(out_json))

    return {
        "model": model,
        "tuning_config_used": str(tuning_config_path) if tuning_config_path else None,
        "tuning_score_from_file": tuning_score,
        "hyperparameters_used": {
            "temperature": generator.temperature,
            "max_tokens": generator.max_tokens,
            "top_p": generator.top_p,
            "frequency_penalty": generator.frequency_penalty,
            "presence_penalty": generator.presence_penalty,
            "seed": generator.seed,
            "top_k": generator.top_k,
        },
        "statements_analyzed": len(statements),
        "requirements_used": len(stmts),
        "questions_export": exported,
        "questions_json_path": str(out_json),
    }
