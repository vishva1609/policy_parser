"""
api_server.py  –  FastAPI REST API for policy_parser
=====================================================
Endpoints:
  GET  /health          → server health check
  POST /parse           → upload a PDF, get structured JSON back
  POST /parse/text      → send raw text body, get structured JSON back

Run with:
  uvicorn api_server:app --reload --port 8000
"""

import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from policy_parser import extract_text_with_tika, clean_text, segment_clauses

# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Policy Parser API",
    description=(
        "Upload a policy PDF and receive a structured JSON with "
        "clauses, sub-clauses, bullet points, keywords and context."
    ),
    version="1.0.0",
)

# Allow Postman / browser access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    """Quick liveness probe – always returns 200 OK."""
    return {"status": "ok", "service": "policy-parser"}


@app.post("/parse", tags=["Parsing"])
async def parse_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and receive a structured JSON representation of
    all clauses, sub-clauses and bullet points found in the document.

    **Request** (multipart/form-data):
    - `file` – the PDF file

    **Response** (200 OK):
    ```json
    {
      "file_name": "policy.pdf",
      "total_segments": 42,
      "segments": [ { "clause_id": "1", ... }, ... ]
    }
    ```
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Write upload to a unique temp file so concurrent requests don't collide
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"_tmp_{uuid.uuid4().hex}_{file.filename}",
    )

    try:
        # Save uploaded bytes to disk
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        # Pipeline: extract → clean → segment
        raw_text = extract_text_with_tika(tmp_path)
        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Tika could not extract any text from the uploaded PDF. "
                       "The file may be scanned/image-only.",
            )

        cleaned  = clean_text(raw_text)
        segments = segment_clauses(cleaned, file_name=file.filename)

        return JSONResponse(
            status_code=200,
            content={
                "file_name":       file.filename,
                "total_segments":  len(segments),
                "segments":        segments,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/parse/text", tags=["Parsing"])
async def parse_text(
    text: str = Body(..., embed=True, example="1. Introduction\nThis policy applies to..."),
    file_name: str = Body("inline.txt", embed=True),
):
    """
    Parse raw policy text (no PDF upload required).
    Useful for quick Postman tests with `application/json` body:

    ```json
    { "text": "1. Introduction\\n...", "file_name": "test.txt" }
    ```
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` field must not be empty.")

    try:
        cleaned  = clean_text(text)
        segments = segment_clauses(cleaned, file_name=file_name)

        return JSONResponse(
            status_code=200,
            content={
                "file_name":      file_name,
                "total_segments": len(segments),
                "segments":       segments,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
