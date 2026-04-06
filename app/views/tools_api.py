"""
Tool endpoints: semantic search over one policy PDF; question generation with optional tuning.
"""
import json
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..common.llm_client import MistralConfig
from ..service.tools_http_service import (
    run_semantic_search_on_pdf,
    run_questions_from_policy,
    resolve_tuning_config_path,
    safe_output_dir,
)
from .deps import require_mistral_config
from .upload_utils import require_pdf_upload

router = APIRouter(prefix="/tools", tags=["tools"])

_TOOLS_ROOT = Path(__file__).resolve().parents[1] / "data" / "uploads" / "tools"


@router.post("/semantic-search")
async def semantic_search(
    policy: UploadFile = File(..., description="Policy PDF to index and search"),
    query: str = Form(..., description="Natural-language search query"),
    top_k: int = Form(5, ge=1, le=50, description="Number of chunks to return"),
):
    """
    Parse, chunk, and embed the PDF in an isolated workspace, then run semantic search.

    Use this to verify retrieval quality before or after a comparison run.
    """
    require_pdf_upload(policy, "policy")
    if not query or not query.strip():
        raise HTTPException(422, detail="query must be non-empty")

    session = _TOOLS_ROOT / "search" / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=True)
    pdf_path = session / Path(policy.filename).name
    pdf_path.write_bytes(await policy.read())

    try:
        return run_semantic_search_on_pdf(
            pdf_path, query.strip(), top_k, session / "work"
        )
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.post("/questions-from-policy")
async def questions_from_policy(
    _: Annotated[MistralConfig, Depends(require_mistral_config)],
    policy: UploadFile = File(..., description="Single policy PDF"),
    tuning_json: Optional[UploadFile] = File(
        None,
        description="Optional tuning_results_*.json from a previous AutoTuner run (best_params + score)",
    ),
    use_cached_tuning: bool = Form(
        False,
        description="If true, load tuning from output_dir/tuning/tuning_results_{tuning_model}.json",
    ),
    tuning_model: Optional[str] = Form(
        None,
        description="Model name slug for cache lookup (e.g. mistral-small-latest)",
    ),
    output_dir: str = Form(
        "output",
        description="Project-relative folder containing tuning/ (must stay inside repo)",
    ),
    model: Optional[str] = Form(None, description="Override Mistral model when not set by tuning file"),
    questions_per_statement: int = Form(3, ge=1, le=10),
    max_statements: Optional[int] = Form(
        None,
        ge=1,
        description="Cap requirement statements for cost control (omit = all)",
    ),
):
    """
    Generate compliance questions from one policy using PolicyAnalyzer + QuestionGenerator.

    Hyperparameters come from (in order): uploaded ``tuning_json`` (previous score /
    ``best_params``), or cached AutoTuner file if ``use_cached_tuning`` + ``tuning_model``,
    else defaults. Intended to run **before** a full A↔B compare when you want tuned
    question quality aligned with a prior tuning run.
    """
    require_pdf_upload(policy, "policy")

    try:
        safe_output_dir(output_dir)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e

    session = _TOOLS_ROOT / "questions" / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=True)
    pdf_path = session / Path(policy.filename).name
    pdf_path.write_bytes(await policy.read())

    tuning_path: Optional[Path] = None
    if tuning_json is not None and tuning_json.filename:
        if not tuning_json.filename.lower().endswith(".json"):
            raise HTTPException(422, detail="tuning_json must be a .json file")
        raw = await tuning_json.read()
        tuning_path = session / "uploaded_tuning.json"
        tuning_path.write_bytes(raw)

    tuning_path = resolve_tuning_config_path(
        tuning_json_path=tuning_path,
        use_cached_tuning=use_cached_tuning,
        tuning_model=tuning_model,
        output_dir_relative=output_dir,
    )

    try:
        return run_questions_from_policy(
            pdf_path,
            session / "work",
            tuning_config_path=tuning_path,
            model_override=model,
            questions_per_statement=questions_per_statement,
            max_statements=max_statements,
        )
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/tuning/available")
async def tuning_available(
    output_dir: str = "output",
    tuning_model: Optional[str] = None,
):
    """
    Check whether a cached tuning file exists for ``tuning_model`` under ``output_dir/tuning/``.
    """
    try:
        base = safe_output_dir(output_dir)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e

    from ..models.question_generator import AutoTuner

    if not tuning_model:
        tuning_dir = base / "tuning"
        files = list(tuning_dir.glob("tuning_results_*.json")) if tuning_dir.is_dir() else []
        return {
            "output_dir": str(base),
            "tuning_files": [f.name for f in files],
        }

    p = AutoTuner.get_tuning_output_path(str(base), tuning_model)
    exists = p.is_file()
    score = None
    model_in_file = None
    if exists:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        score = (data.get("best_params") or {}).get("score")
        model_in_file = data.get("model")
    return {
        "path": str(p),
        "exists": exists,
        "cached_score": score,
        "model_in_file": model_in_file,
    }
