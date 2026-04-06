"""Multipart upload validation shared across comparison, tools, and sessions."""
from pathlib import Path

from fastapi import HTTPException, UploadFile


def require_pdf_upload(upload: UploadFile, field: str) -> None:
    if not upload.filename:
        raise HTTPException(status_code=422, detail=f"{field}: missing filename")
    if Path(upload.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=422,
            detail=f"{field}: expected a .pdf file, got {upload.filename!r}",
        )
