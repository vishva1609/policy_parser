"""
FastAPI ASGI entry — policy comparison API on port 8000 by default.

Coexists with Flask QA RAG in app/app.py (port 5000). Run::

    uv run uvicorn app.fastapi_app:app --reload --port 8000

OpenAPI: GET /docs
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one shared MistralConfig when the API key is available (DI)."""
    from .common.llm_client import load_mistral_config_optional

    app.state.mistral_config = load_mistral_config_optional()
    yield


def create_fastapi_app() -> FastAPI:
    application = FastAPI(
        title="File Parsing API",
        description=(
            "Policy A↔B comparison, semantic search on one PDF, and tuned question generation."
        ),
        lifespan=lifespan,
    )
    from .views.system_api import router as system_router
    from .views.comparison_api import router as comparison_router
    from .views.tools_api import router as tools_router
    from .views.sessions_api import router as sessions_router

    application.include_router(system_router)
    application.include_router(comparison_router)
    application.include_router(tools_router)
    application.include_router(sessions_router)
    return application


app = create_fastapi_app()
