"""
FastAPI routes for policy A↔B comparison (views layer).

Validates multipart uploads, persists files under app/data/uploads/compare/,
then calls :func:`app.service.comparison_http_service.run_policy_compare`.
"""
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..common.llm_client import MistralConfig
from ..service.comparison_http_service import run_policy_compare
from .deps import require_mistral_config
from .http_constants import EXCEL_MEDIA_TYPE
from .upload_utils import require_pdf_upload

router = APIRouter(tags=["comparison"])

_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "data" / "uploads" / "compare"


@router.post("/compare/policies")
async def compare_policies(
    _cfg: Annotated[MistralConfig, Depends(require_mistral_config)],
    policy_a: UploadFile = File(..., description="Policy A PDF (source of truth)"),
    policy_b: UploadFile = File(..., description="Policy B PDF (to validate)"),
):
    """
    Run the full compliance pipeline and return the Excel report as a download.
    """
    require_pdf_upload(policy_a, "policy_a")
    require_pdf_upload(policy_b, "policy_b")

    session = _UPLOAD_ROOT / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=True)
    path_a = session / Path(policy_a.filename).name
    path_b = session / Path(policy_b.filename).name

    path_a.write_bytes(await policy_a.read())
    path_b.write_bytes(await policy_b.read())

    out_dir = session / "output"

    try:
        report_path = run_policy_compare(
            path_a,
            path_b,
            output_dir=out_dir,
            config=_cfg,
            api_key=os.getenv("MISTRAL_API_KEY"),
            session_id=session.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=str(report_path),
        filename=report_path.name,
        media_type=EXCEL_MEDIA_TYPE,
    )
