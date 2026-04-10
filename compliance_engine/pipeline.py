"""
compliance_engine/pipeline.py
──────────────────────────────
End-to-end orchestration pipeline.

run_comparison() is the single entry point used by the API server.
Supports:
  • async-style batch processing (uses asyncio.to_thread for CPU-bound work)
  • progress callbacks so the caller can stream status updates
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable

from .embedder import PolicyIndex
from .comparator import compare_clause
from .scorer import compute_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Synchronous pipeline (blocked per-clause)
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison_sync(
    clauses_a: list[dict],
    clauses_b: list[dict],
    policy_a_name: str = "Policy A",
    policy_b_name: str = "Policy B",
    top_k: int = 5,
    llm_backend: str = "rules",
    llm_model: str | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Full synchronous pipeline.

    Returns the final report dict (including clause_details, critical_gaps …).
    """
    start = time.time()

    # ── Step 3: Embed Policy B into FAISS ───────────────────────────────────
    if progress_cb:
        progress_cb(0, len(clauses_a), "Building vector index for Policy B …")

    index_b = PolicyIndex().build(clauses_b)

    # ── Steps 4+5: Compare each clause in A against index of B ──────────────
    results: list[dict] = []
    total = len(clauses_a)

    for i, clause_a in enumerate(clauses_a, 1):
        result = compare_clause(
            clause_a,
            index_b,
            top_k=top_k,
            llm_backend=llm_backend,
            llm_model=llm_model,
        )
        results.append(result)
        if progress_cb:
            progress_cb(i, total, f"Compared clause {i}/{total}: {clause_a.get('clause_id', '')}")

    # ── Steps 6+7: Gap detection + scoring ──────────────────────────────────
    if progress_cb:
        progress_cb(total, total, "Computing compliance scores …")

    report = compute_scores(results)
    report["policy_a_name"] = policy_a_name
    report["policy_b_name"] = policy_b_name
    report["elapsed_seconds"] = round(time.time() - start, 2)
    return report


# ─────────────────────────────────────────────────────────────────────────────
#  Async wrapper (non-blocking for FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

async def run_comparison(
    clauses_a: list[dict],
    clauses_b: list[dict],
    policy_a_name: str = "Policy A",
    policy_b_name: str = "Policy B",
    top_k: int = 5,
    llm_backend: str = "rules",
    llm_model: str | None = None,
) -> dict:
    """Async wrapper – offloads CPU-intensive work to a thread pool."""
    return await asyncio.to_thread(
        run_comparison_sync,
        clauses_a, clauses_b,
        policy_a_name, policy_b_name,
        top_k, llm_backend, llm_model,
    )
