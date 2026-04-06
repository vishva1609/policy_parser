"""
HTTP-oriented wrapper for policy comparison — keeps FastAPI routers thin.

Delegates to PolicyComparisonPipeline (service layer), mirroring qa-bot-llm’s
pattern of views → service → models.
"""
from pathlib import Path
from typing import Any, Dict, Optional

from ..common.llm_client import MistralConfig
from .comparison_service import PolicyComparisonPipeline


def build_comparison_pipeline(
    output_dir: str | Path,
    config: Optional[MistralConfig] = None,
    *,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    **pipeline_kwargs: Any,
) -> PolicyComparisonPipeline:
    """Construct a pipeline with YAML-backed defaults (sessions + one-shot compare)."""
    opts = {**default_pipeline_options_from_config(), **pipeline_kwargs}
    return PolicyComparisonPipeline(
        output_dir=str(output_dir),
        config=config,
        api_key=api_key,
        session_id=session_id,
        **opts,
    )


def run_policy_compare(
    path_a: Path,
    path_b: Path,
    *,
    output_dir: Path,
    config: Optional[MistralConfig] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    **pipeline_kwargs: Any,
) -> Path:
    """
    Run A↔B comparison on saved PDF paths; return path to the Excel report.

    Args:
        path_a:       Policy A (source of truth) on disk.
        path_b:       Policy B to validate.
        output_dir:   Directory for vectordb, reports, and JSON sidecars.
        config:       Injected MistralConfig (shared LLM), optional.
        api_key:      Mistral API key for segmenter / fallback when config omitted.
        **pipeline_kwargs: Passed to PolicyComparisonPipeline (e.g. top_k_candidates).

    Returns:
        Path to generated ``.xlsx`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_comparison_pipeline(
        output_dir,
        config=config,
        api_key=api_key,
        session_id=session_id,
        **pipeline_kwargs,
    )
    report_path = pipeline.compare(str(path_a), str(path_b))
    return Path(report_path)


def default_pipeline_options_from_config() -> Dict[str, Any]:
    """Optional YAML defaults for PolicyComparisonPipeline (non-breaking)."""
    try:
        from ..common.config_loader import load_pipeline_config
        raw = load_pipeline_config()
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    comp = raw.get("comparison") or {}
    seg = raw.get("segmentation") or {}
    chunk = raw.get("chunking") or {}
    emb = raw.get("embeddings") or {}
    llm = raw.get("llm") or {}
    if comp.get("top_k_candidates") is not None:
        out["top_k_candidates"] = int(comp["top_k_candidates"])
    if comp.get("missing_threshold") is not None:
        out["missing_similarity_threshold"] = float(comp["missing_threshold"])
    if comp.get("gap_threshold") is not None:
        out["gap_similarity_threshold"] = float(comp["gap_threshold"])
    if seg.get("use_llm_fallback") is not None:
        out["use_llm_segmentation"] = bool(seg["use_llm_fallback"])
    if chunk.get("max_chars") is not None:
        out["max_chunk_chars"] = int(chunk["max_chars"])
    if chunk.get("min_chars") is not None:
        out["min_chunk_chars"] = int(chunk["min_chars"])
    if chunk.get("strategy") is not None:
        out["chunk_by"] = str(chunk["strategy"])
    if emb.get("model") is not None:
        out["embedding_model"] = str(emb["model"])
    if llm.get("model") is not None:
        out["mistral_model"] = str(llm["model"])
    return out
