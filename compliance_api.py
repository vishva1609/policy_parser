"""
compliance_api.py  –  Session-Based Policy Compliance API
============================================================
Refactored to match granular step-by-step session architecture.

Run with:
  uvicorn compliance_api:app --reload --port 8000
"""

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import (
    BackgroundTasks, Body, FastAPI, File, HTTPException,
    Query, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from policy_parser import extract_text_with_tika as extract_text, clean_text, segment_clauses
from compliance_engine.pipeline import run_comparison
from compliance_engine.database import engine, SessionLocal, Base
from compliance_engine import models

# Initialize Database
Base.metadata.create_all(bind=engine)

_EXPORTS_DIR = Path(__file__).parent / "_exports"
_EXPORTS_DIR.mkdir(exist_ok=True)

# In-memory job tracker
_JOBS = {}

# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Policy Compliance Comparison API",
    description=(
        "Parse policy documents and compare them clause-by-clause "
        "using vector embeddings + hybrid LLM/rule-based analysis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class CompareJSONRequest(BaseModel):
    clauses_a: list[dict] = Field(..., description="Parsed clauses from Policy A")
    clauses_b: list[dict] = Field(..., description="Parsed clauses from Policy B")
    policy_a_name: str = Field("Policy A", description="Label for Policy A")
    policy_b_name: str = Field("Policy B", description="Label for Policy B")
    top_k: int = Field(5, ge=1, le=20, description="Number of nearest neighbours to retrieve")
    llm_backend: Literal["rules", "openai", "ollama"] = Field(
        "rules",
        description=(
            "'rules' (default, no key needed) | "
            "'openai' (needs OPENAI_API_KEY env var) | "
            "'ollama' (needs local Ollama running on :11434)"
        ),
    )
    llm_model: Optional[str] = Field(
        None,
        description="Model name override, e.g. 'gpt-4o', 'llama3'. Defaults per backend.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok", "service": "policy-compliance-api", "version": "2.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
#  TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/tools/semantic-search", tags=["tools"])
async def semantic_search(query: str, top_k: int = 5):
    """Perform a semantic search across the default indexed bank policy."""
    from compliance_engine.embedder import PolicyIndex
    # Load default bank policy index for demo
    index_path = Path("bank_policy.pdf.index")
    if not Path(str(index_path) + ".faiss").exists():
        return {"message": "Index not found. Please run Step 3 (Embed) first.", "results": []}
    
    idx = PolicyIndex.load(str(index_path))
    matches = idx.search({"text": query}, top_k=top_k)
    return {"query": query, "results": matches}

@app.post("/tools/questions-from-policy", tags=["tools"])
async def questions_from_policy(policy_id: str):
    """Generate potential audit questions from a given policy."""
    return {"message": "Questions generated", "questions": []}

@app.get("/tools/tuning/available", tags=["tools"])
async def tuning_available():
    """Get list of available tuning models or parameters."""
    return {"available_models": ["gpt-4o", "llama3", "rules-v1"]}

# ─────────────────────────────────────────────────────────────────────────────
#  SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/sessions", tags=["sessions"])
async def create_session(name_a: str = "Policy A", name_b: str = "Policy B"):
    """Initialize a new comparison session."""
    session_id = uuid.uuid4().hex
    db = SessionLocal()
    new_job = models.ComparisonJob(
        id=session_id,
        status="active",
        current_step="init",
        name_a=name_a,
        name_b=name_b
    )
    db.add(new_job)
    db.commit()
    db.close()
    return {"session_id": session_id, "status": "created"}

@app.get("/sessions/{session_id}", tags=["sessions"])
async def get_session(session_id: str):
    """Get metadata for a specific session."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    db.close()
    if not job: raise HTTPException(404, "Session not found")
    return job

@app.delete("/sessions/{session_id}", tags=["sessions"])
async def delete_session(session_id: str):
    """Cleanup and delete a session."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    if job:
        db.delete(job)
        db.commit()
    db.close()
    return {"status": "deleted"}

@app.post("/sessions/{session_id}/steps/parse", tags=["sessions"])
async def step_parse(session_id: str, file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    """STEP 1: Extract raw text from both PDFs."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    if not job: raise HTTPException(404, "Session not found")
    
    # Simple extraction for demo
    path_a = _EXPORTS_DIR / f"{session_id}_a.pdf"
    path_b = _EXPORTS_DIR / f"{session_id}_b.pdf"
    with open(path_a, "wb") as f: shutil.copyfileobj(file_a.file, f)
    with open(path_b, "wb") as f: shutil.copyfileobj(file_b.file, f)
    
    job.raw_text_a = extract_text(str(path_a))
    job.raw_text_b = extract_text(str(path_b))
    job.current_step = "parse"
    db.commit()
    db.close()
    return {"status": "success", "step": "parse"}

@app.post("/sessions/{session_id}/steps/segment", tags=["sessions"])
async def step_segment(session_id: str):
    """STEP 2: Segment raw text into clauses/sub-clauses."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    if not job or not job.raw_text_a: raise HTTPException(400, "Parse step needed first")
    
    job.segments_a = segment_clauses(clean_text(job.raw_text_a), file_name=job.name_a)
    job.segments_b = segment_clauses(clean_text(job.raw_text_b), file_name=job.name_b)
    job.current_step = "segment"
    db.commit()
    db.close()
    return {"status": "success", "step": "segment", "segments_count_a": len(job.segments_a)}

@app.post("/sessions/{session_id}/steps/embed", tags=["sessions"])
async def step_embed(session_id: str):
    """STEP 3: Vectorize segments for comparison."""
    # (In real implementation this would trigger FAISS building)
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    job.current_step = "embed"
    db.commit()
    db.close()
    return {"status": "success", "step": "embed"}

@app.post("/sessions/{session_id}/steps/compare", tags=["sessions"])
async def step_compare(session_id: str):
    """STEP 4: Match and compare clauses from A to B."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    
    from compliance_engine.pipeline import run_comparison_sync
    report = run_comparison_sync(job.segments_a, job.segments_b, job.name_a, job.name_b)
    
    job.results_json = report
    job.overall_score = report.get("overall_compliance", 0)
    job.current_step = "compare"
    db.commit()
    db.close()
    return {"status": "success", "step": "compare"}

@app.post("/sessions/{session_id}/steps/score", tags=["sessions"])
async def step_score(session_id: str):
    """STEP 5: Calculate final compliance scores."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    job.current_step = "score"
    db.commit()
    db.close()
    return {"status": "success", "step": "score", "overall_score": job.overall_score}

@app.post("/sessions/{session_id}/steps/export", tags=["sessions"])
async def step_export(session_id: str):
    """STEP 6: Generate final export report."""
    db = SessionLocal()
    job = db.query(models.ComparisonJob).filter(models.ComparisonJob.id == session_id).first()
    job.status = "done"
    job.current_step = "export"
    db.commit()
    db.close()
    return {"status": "success", "step": "export", "report_url": f"/compare/{session_id}/export"}


# ─────────────────────────────────────────────────────────────────────────────
#  Synchronous JSON-to-JSON comparison
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/compare", tags=["Comparison"])
async def compare_json(req: CompareJSONRequest):
    """
    Compare two pre-parsed clause lists.

    **Tip**: Use `/parse` to get clauses lists from PDFs first, then pass them here.

    Returns full compliance report with gap analysis and scores.
    """
    if not req.clauses_a:
        raise HTTPException(400, "clauses_a must not be empty.")
    if not req.clauses_b:
        raise HTTPException(400, "clauses_b must not be empty.")

    try:
        report = await run_comparison(
            clauses_a=req.clauses_a,
            clauses_b=req.clauses_b,
            policy_a_name=req.policy_a_name,
            policy_b_name=req.policy_b_name,
            top_k=req.top_k,
            llm_backend=req.llm_backend,
            llm_model=req.llm_model,
        )
        return JSONResponse(report)
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Async PDF-to-PDF comparison (background job)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pdf_job(job_id: str, path_a: Path, path_b: Path,
                 name_a: str, name_b: str,
                 top_k: int, backend: str, model: str | None):
    """Background worker – runs in thread pool."""
    try:
        # Use aliased extract_text (pdfplumber)
        raw_a = extract_text(str(path_a))
        clauses_a = segment_clauses(clean_text(raw_a), file_name=name_a)

        _JOBS[job_id]["progress"] = "Extracting text from Policy B …"
        raw_b = extract_text(str(path_b))
        clauses_b = segment_clauses(clean_text(raw_b), file_name=name_b)

        _JOBS[job_id]["progress"] = "Running comparison …"

        # run_comparison is async; call sync version directly from thread
        from compliance_engine.pipeline import run_comparison_sync
        report = run_comparison_sync(
            clauses_a, clauses_b,
            policy_a_name=name_a,
            policy_b_name=name_b,
            top_k=top_k,
            llm_backend=backend,
            llm_model=model,
        )

        # ── Database Persistence ──────────────────────────────────────────────
        try:
            db_session = SessionLocal()
            # 1. Save Job Result
            db_job = models.ComparisonJob(
                id=job_id,
                status="done",
                name_a=name_a,
                name_b=name_b,
                overall_score=report.get("overall_compliance", 0),
                results_json=report
            )
            db_session.add(db_job)
            db_session.commit()
            db_session.close()
        except Exception as db_err:
            print(f"Database Save Error: {db_err}")

        _JOBS[job_id]["report"] = report
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["progress"] = "Complete"
    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(exc)
    finally:
        if path_a.exists():
            path_a.unlink()
        if path_b.exists():
            path_b.unlink()


@app.post("/compare/pdf", tags=["Comparison"], status_code=202)
async def compare_pdfs(
    background_tasks: BackgroundTasks,
    file_a: UploadFile = File(..., description="Policy A PDF"),
    file_b: UploadFile = File(..., description="Policy B PDF"),
    top_k: int = Query(5, ge=1, le=20),
    llm_backend: str = Query("rules", description="rules | openai | ollama"),
    llm_model: Optional[str] = Query(None),
):
    """
    Upload two PDFs and start an async comparison job.
    Returns a `job_id` – poll `GET /compare/{job_id}` for status.
    """
    for f in [file_a, file_b]:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename} is not a PDF.")

    job_id = uuid.uuid4().hex
    path_a = _EXPORTS_DIR / f"{job_id}_a_{file_a.filename}"
    path_b = _EXPORTS_DIR / f"{job_id}_b_{file_b.filename}"

    with open(path_a, "wb") as f:
        shutil.copyfileobj(file_a.file, f)
    with open(path_b, "wb") as f:
        shutil.copyfileobj(file_b.file, f)

    _JOBS[job_id] = {
        "status": "queued",
        "progress": "Waiting …",
        "report": None,
        "error": None,
        "policy_a": file_a.filename,
        "policy_b": file_b.filename,
    }

    background_tasks.add_task(
        asyncio.to_thread,
        _run_pdf_job,
        job_id, path_a, path_b,
        file_a.filename, file_b.filename,
        top_k, llm_backend, llm_model,
    )

    return {"job_id": job_id, "status": "queued",
            "poll_url": f"/compare/{job_id}"}


@app.get("/compare/{job_id}", tags=["Comparison"])
def poll_job(job_id: str):
    """Poll the status of an async comparison job – checks memory first then DB."""
    # 1. Check in-memory cache first (for real-time status)
    job = _JOBS.get(job_id)
    if job:
        resp: dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress"),
            "policy_a": job.get("policy_a"),
            "policy_b": job.get("policy_b"),
        }
        if job["status"] == "done":
            resp["overall_compliance"] = job["report"].get("overall_compliance")
            resp["export_url"] = f"/compare/{job_id}/export"
        if job["status"] == "error":
            resp["error"] = job.get("error")
        return resp

    # 2. Check Database (PostgreSQL)
    db_session = SessionLocal()
    db_job = db_session.query(models.ComparisonJob).filter(models.ComparisonJob.id == job_id).first()
    db_session.close()
    
    if not db_job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
        
    return {
        "job_id": db_job.id,
        "status": db_job.status,
        "overall_compliance": db_job.overall_score,
        "policy_a": db_job.name_a,
        "policy_b": db_job.name_b,
        "history": True
    }

@app.get("/history", tags=["Database"])
def get_job_history():
    """Get complete check history from PostgreSQL."""
    db = SessionLocal()
    history = db.query(models.ComparisonJob).order_by(models.ComparisonJob.created_at.desc()).all()
    db.close()
    return [{
        "job_id": h.id,
        "name_a": h.name_a,
        "name_b": h.name_b,
        "date": h.created_at,
        "score": h.overall_score
    } for h in history]

@app.post("/compare/{job_id}/override", tags=["Feedback"])
def override_result(job_id: str, clause_a_id: str, new_status: str):
    """User feedback loop: Override an AI decision manually."""
    db_session = SessionLocal()
    job = db_session.query(models.ComparisonJob).filter(models.ComparisonJob.id == job_id).first()
    if not job:
        db_session.close()
        raise HTTPException(404, "Job not found")
    
    # Update JSON report in DB
    report = job.results_json
    for comparison in report.get("comparisons", []):
        if comparison.get("clause_a_id") == clause_a_id:
            comparison["status"] = new_status
            comparison["reason"] = f"User override to {new_status}"
            break
            
    job.results_json = report
    db_session.commit()
    db_session.close()
    return {"message": "Result updated successfully"}


@app.get("/compare/{job_id}/report", tags=["Comparison"])
def get_full_report(job_id: str):
    """Retrieve the full comparison report JSON for a completed job."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "done":
        raise HTTPException(409, f"Job status is '{job['status']}', not done yet.")
    return JSONResponse(job["report"])


# ─────────────────────────────────────────────────────────────────────────────
#  Export endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/compare/{job_id}/export", tags=["Export"])
def export_report(
    job_id: str,
    format: Literal["excel", "pdf"] = Query("excel"),
):
    """
    Download the compliance report as Excel or PDF.
    `format=excel` (default) or `format=pdf`
    """
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "done":
        raise HTTPException(409, "Job not complete yet.")

    report = job["report"]
    pa = job.get("policy_a", "PolicyA")
    pb = job.get("policy_b", "PolicyB")

    try:
        if format == "excel":
            from compliance_engine.exporter import export_excel
            fname = _EXPORTS_DIR / f"{job_id}_report.xlsx"
            export_excel(report, str(fname))
            return FileResponse(str(fname),
                                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                filename="compliance_report.xlsx")
        else:
            from compliance_engine.exporter import export_pdf
            fname = _EXPORTS_DIR / f"{job_id}_report.pdf"
            export_pdf(report, pa, pb, str(fname))
            return FileResponse(str(fname),
                                media_type="application/pdf",
                                filename="compliance_report.pdf")
    except ImportError as exc:
        raise HTTPException(500, f"Missing export dependency: {exc}")
    except Exception as exc:
        raise HTTPException(500, str(exc))
