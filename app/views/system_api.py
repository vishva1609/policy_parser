"""Liveness / readiness — separate from domain routers (comparison, tools, sessions)."""
from fastapi import APIRouter, Request

from .deps import get_mistral_config_optional

router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(request: Request):
    """
    Whether the app constructed a shared ``MistralConfig`` at startup
    (API key present and langchain-mistralai usable).
    """
    cfg = get_mistral_config_optional(request)
    return {
        "ready": cfg is not None,
        "llm": "configured" if cfg is not None else "not_configured",
    }
