"""
Session-scoped comparison pipeline — one HTTP step at a time (parse → segment → embed → compare → score → export).

Upload both PDFs once, then call ``/sessions/{id}/steps/...`` in order.
"""
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from ..common.llm_client import MistralConfig
from ..service.comparison_http_service import build_comparison_pipeline
from ..service.comparison_service import PolicyComparisonPipeline
from .deps import require_mistral_config
from .http_constants import EXCEL_MEDIA_TYPE
from .upload_utils import require_pdf_upload

router = APIRouter(prefix="/sessions", tags=["sessions"])

_SESSIONS_ROOT = Path(__file__).resolve().parents[1] / "data" / "uploads" / "sessions"

# Only allow hex UUID folder names (no path traversal)
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _session_dir(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    d = _SESSIONS_ROOT / session_id.lower()
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    return d


def _work_dir(session_id: str) -> Path:
    return _session_dir(session_id) / "work"


def _state_path(session_id: str) -> Path:
    return _session_dir(session_id) / "state.json"


def _load_state(session_id: str) -> Dict[str, Any]:
    p = _state_path(session_id)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(session_id: str, data: Dict[str, Any]) -> None:
    p = _state_path(session_id)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _artifact_status(work: Path) -> Dict[str, Any]:
    """Which step outputs exist on disk."""
    return {
        "parse": {
            "done": (work / "parsed_policy_a.json").is_file()
            and (work / "parsed_policy_b.json").is_file(),
        },
        "segment": {
            "done": (work / "clauses_policy_a.json").is_file()
            and (work / "clauses_policy_b.json").is_file(),
        },
        "embed": {
            "done": _vectordb_nonempty(work / "vectordb" / "policy_a")
            and _vectordb_nonempty(work / "vectordb" / "policy_b"),
        },
        "compare": {
            "done": (work / "comparison_results_a_to_b.json").is_file()
            and (work / "comparison_results_b_to_a.json").is_file(),
        },
        "score": {"done": (work / "comparison_scores.json").is_file()},
        "export": {
            "done": bool(list(work.glob("compliance_report_*.xlsx")))
            or bool(list(work.glob("comparison_*_vs_*.json"))),
        },
    }


def _vectordb_nonempty(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def _mistral_cfg(request: Request) -> Optional[MistralConfig]:
    return getattr(request.app.state, "mistral_config", None)


def _require_full_session_for_compare_flow(session_id: str) -> None:
    """Compare / score / export need two real PDFs."""
    st = _load_state(session_id)
    if st.get("session_mode") == "single_policy_a":
        raise HTTPException(
            status_code=400,
            detail=(
                "This session was created with only policy A (cost-saving test mode). "
                "Create a new session uploading both policy_a and policy_b to run compare, score, or export."
            ),
        )


def require_full_compare_session(session_id: str = PathParam(..., description="32-char hex session id")) -> None:
    """Runs before LLM dependency so single-file sessions fail fast without API calls."""
    _require_full_session_for_compare_flow(session_id)


def _run_step_core(
    session_id: str,
    request: Request,
    work: Path,
    step_name: str,
    run: Callable[[PolicyComparisonPipeline], Any],
    *,
    post: Optional[Callable[[Any, PolicyComparisonPipeline, Path], None]] = None,
) -> Any:
    try:
        pl = build_comparison_pipeline(
            work,
            _mistral_cfg(request),
            api_key=None,
            session_id=session_id,
        )
        out = run(pl)
        if post:
            post(out, pl, work)
        st = _load_state(session_id)
        st["last_step"] = step_name
        _save_state(session_id, st)
        return out
    except FileNotFoundError as e:
        raise HTTPException(400, detail=str(e)) from e
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.post("")
async def create_session(
    policy_a: UploadFile = File(...),
    policy_b: Optional[UploadFile] = File(
        None,
        description="Optional second PDF. Omit to test parse → segment → embed on policy A only (no compare API cost).",
    ),
):
    """
    Upload policy A and optionally policy B.

    **Single-file mode:** omit ``policy_b`` to run steps through ``embed`` on one PDF only;
    ``compare`` / ``score`` / ``export`` return an error until you create a full two-PDF session.
    """
    require_pdf_upload(policy_a, "policy_a")
    single = policy_b is None or not policy_b.filename
    if not single:
        require_pdf_upload(policy_b, "policy_b")

    sid = uuid.uuid4().hex
    base = _SESSIONS_ROOT / sid
    base.mkdir(parents=True, exist_ok=True)
    (base / "work").mkdir(exist_ok=True)

    (base / "policy_a.pdf").write_bytes(await policy_a.read())
    if not single:
        (base / "policy_b.pdf").write_bytes(await policy_b.read())

    state = {
        "session_id": sid,
        "session_mode": "single_policy_a" if single else "full",
        "policy_a_original_name": policy_a.filename,
        "policy_b_original_name": None if single else policy_b.filename,
    }
    _save_state(sid, state)

    out = {
        "session_id": sid,
        "policy_a_saved_as": "policy_a.pdf",
        "next": "POST /sessions/{session_id}/steps/parse",
    }
    if single:
        out["session_mode"] = "single_policy_a"
        out["note"] = (
            "Only policy A uploaded — use steps through embed to test cheaply; "
            "compare/score/export need a new session with both PDFs."
        )
    else:
        out["policy_b_saved_as"] = "policy_b.pdf"
    return out


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Session metadata + which pipeline steps have produced artifacts."""
    d = _session_dir(session_id)
    work = d / "work"
    work.mkdir(exist_ok=True)
    state = _load_state(session_id)
    pb = d / "policy_b.pdf"
    return {
        "session_id": session_id,
        "state": state,
        "steps": _artifact_status(work),
        "paths": {
            "session_dir": str(d),
            "work_dir": str(work),
            "policy_a": str(d / "policy_a.pdf"),
            "policy_b": str(pb) if pb.is_file() else None,
        },
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    shutil.rmtree(_session_dir(session_id), ignore_errors=True)
    return {"deleted": session_id}


@router.post("/{session_id}/steps/parse")
async def step_parse(session_id: str, request: Request):
    """Step 1: PDF → structured ``parsed_policy_*.json``."""
    d = _session_dir(session_id)
    work = _work_dir(session_id)
    work.mkdir(parents=True, exist_ok=True)
    st = _load_state(session_id)
    single = st.get("session_mode") == "single_policy_a"
    pdf_b = None if single else str(d / "policy_b.pdf")
    if not single and not (d / "policy_b.pdf").is_file():
        raise HTTPException(status_code=400, detail="policy_b.pdf missing for this session")

    return _run_step_core(
        session_id,
        request,
        work,
        "parse",
        lambda pl: pl.step_parse(str(d / "policy_a.pdf"), pdf_b),
    )


@router.post("/{session_id}/steps/segment")
async def step_segment(session_id: str, request: Request):
    """Step 2: parsed JSON → ``clauses_policy_*.json``."""
    work = _work_dir(session_id)
    work.mkdir(parents=True, exist_ok=True)
    return _run_step_core(session_id, request, work, "segment", lambda pl: pl.step_segment())


@router.post("/{session_id}/steps/embed")
async def step_embed(session_id: str, request: Request):
    """Step 3: clauses → Chroma ``vectordb`` + ``segment_manifest.json``."""
    work = _work_dir(session_id)

    def _enrich_counts(out: Any, pl: PolicyComparisonPipeline, w: Path) -> None:
        try:
            from ..models.embedder import EmbeddingPipeline

            ea = EmbeddingPipeline(
                model_name=pl.embedding_model,
                db_path=str(w / "vectordb" / "policy_a"),
                collection_name="policy_a",
            )
            eb = EmbeddingPipeline(
                model_name=pl.embedding_model,
                db_path=str(w / "vectordb" / "policy_b"),
                collection_name="policy_b",
            )
            out["indexed_chunks_a"] = ea.get_stats().get(
                "total_chunks", out.get("indexed_chunks_a")
            )
            out["indexed_chunks_b"] = eb.get_stats().get(
                "total_chunks", out.get("indexed_chunks_b")
            )
        except Exception:
            pass

    return _run_step_core(
        session_id,
        request,
        work,
        "embed",
        lambda pl: pl.step_embed(),
        post=_enrich_counts,
    )


@router.post(
    "/{session_id}/steps/compare",
    dependencies=[Depends(require_full_compare_session)],
)
async def step_compare(
    session_id: str,
    request: Request,
    _: MistralConfig = Depends(require_mistral_config),
):
    """Step 4: semantic match + LLM + gap rules → ``comparison_results_*.json``."""
    work = _work_dir(session_id)
    return _run_step_core(session_id, request, work, "compare", lambda pl: pl.step_compare())


@router.post(
    "/{session_id}/steps/score",
    dependencies=[Depends(require_full_compare_session)],
)
async def step_score(session_id: str, request: Request):
    """Step 5: aggregate ``comparison_scores.json``."""
    work = _work_dir(session_id)
    return _run_step_core(session_id, request, work, "score", lambda pl: pl.step_score())


@router.post(
    "/{session_id}/steps/export",
    dependencies=[Depends(require_full_compare_session)],
)
async def step_export(
    session_id: str,
    request: Request,
    report_filename: Optional[str] = Query(None, description="Optional .xlsx filename"),
):
    """Step 6: Excel + full comparison JSON (same as monolithic compare)."""
    work = _work_dir(session_id)

    def _persist_excel_path(out: dict, _pl: PolicyComparisonPipeline, _w: Path) -> None:
        st = _load_state(session_id)
        st["excel_report"] = out["excel_path"]
        _save_state(session_id, st)

    return _run_step_core(
        session_id,
        request,
        work,
        "export",
        lambda pl: {
            "step": "export",
            "excel_path": pl.step_export_excel(report_filename=report_filename),
        },
        post=_persist_excel_path,
    )


@router.get("/{session_id}/exports/excel")
async def download_excel(session_id: str):
    """Download the latest ``compliance_report_*.xlsx`` in the session work dir."""
    work = _work_dir(session_id)
    files = sorted(work.glob("compliance_report_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(404, detail="No Excel report; run POST .../steps/export first.")
    f = files[0]
    return FileResponse(path=str(f), filename=f.name, media_type=EXCEL_MEDIA_TYPE)


@router.get("/{session_id}/exports/json")
async def download_comparison_json(session_id: str):
    """Download ``comparison_<a>_vs_<b>.json`` from the work dir."""
    work = _work_dir(session_id)
    files = list(work.glob("comparison_*_vs_*.json"))
    if not files:
        raise HTTPException(404, detail="No comparison JSON; run POST .../steps/export first.")
    f = max(files, key=lambda p: p.stat().st_mtime)
    return FileResponse(path=str(f), filename=f.name, media_type="application/json")


@router.get("/{session_id}/artifacts/parsed/{which}")
async def get_parsed_json(session_id: str, which: str):
    """Inspect raw parsed document JSON (``which`` = ``a`` or ``b``)."""
    if which not in ("a", "b"):
        raise HTTPException(422, detail="which must be 'a' or 'b'")
    work = _work_dir(session_id)
    name = "parsed_policy_a.json" if which == "a" else "parsed_policy_b.json"
    p = work / name
    if not p.is_file():
        raise HTTPException(404, detail="Run steps/parse first.")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/{session_id}/artifacts/manifest")
async def get_segment_manifest(session_id: str):
    """Return ``segment_manifest.json`` if embed step has run."""
    work = _work_dir(session_id)
    p = work / "segment_manifest.json"
    if not p.is_file():
        raise HTTPException(404, detail="Run steps/embed first.")
    return json.loads(p.read_text(encoding="utf-8"))
