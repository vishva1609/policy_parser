"""FastAPI dependencies: LLM config from app.state (lifespan)."""
from typing import Optional

from fastapi import HTTPException, Request

from ..common.llm_client import MistralConfig
from .http_constants import LLM_UNCONFIGURED_DETAIL


def get_mistral_config_optional(request: Request) -> Optional[MistralConfig]:
    return getattr(request.app.state, "mistral_config", None)


def require_mistral_config(request: Request) -> MistralConfig:
    cfg = get_mistral_config_optional(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail=LLM_UNCONFIGURED_DETAIL)
    return cfg
