"""
compliance_engine/embedder.py
────────────────────────────
Step 3: Embeddings + Vector DB (FAISS)

Uses sentence-transformers (all-MiniLM-L6-v2) — fully offline, no API key needed.
Call build_index() to embed a list of clause dicts and store them in a FAISS index.
Call search() to retrieve the top-k nearest neighbours for a query clause.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np

# ── lazy imports so the rest of the server loads even if GPU libs are missing ──
_MODEL = None
_FAISS = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _get_faiss():
    global _FAISS
    if _FAISS is None:
        import faiss  # noqa: F401
        _FAISS = faiss
    return _FAISS


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clause_to_text(clause: dict) -> str:
    """Combine title + text for a richer embedding."""
    parts = []
    if clause.get("title"):
        parts.append(clause["title"])
    if clause.get("text"):
        parts.append(clause["text"])
    return " ".join(parts)[:2000]          # cap at 2 000 chars to stay fast


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

class PolicyIndex:
    """FAISS index + metadata for one policy document."""

    def __init__(self):
        self.clauses: list[dict] = []
        self.index = None            # faiss.IndexFlatIP

    # ------------------------------------------------------------------
    def build(self, clauses: list[dict]) -> "PolicyIndex":
        """Embed all clauses and build a FAISS inner-product index."""
        if not clauses:
            raise ValueError("No clauses provided to embed.")

        model = _get_model()
        faiss = _get_faiss()

        self.clauses = clauses
        texts = [_clause_to_text(c) for c in clauses]

        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False,
                                  normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim (vectors are L2-norm'd)
        self.index.add(embeddings)
        return self

    # ------------------------------------------------------------------
    def search(self, query_clause: dict, top_k: int = 5) -> list[dict]:
        """
        Return up to top_k nearest neighbours for query_clause.
        Each result is the matched clause dict plus 'similarity_score'.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        model = _get_model()
        faiss = _get_faiss()

        text = _clause_to_text(query_clause)
        qvec = model.encode([text], normalize_embeddings=True)
        qvec = np.array(qvec, dtype="float32")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qvec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.clauses[idx])
            entry["similarity_score"] = float(score)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Persist index to disk (two files: .faiss + .meta)."""
        faiss = _get_faiss()
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.clauses, f)

    @classmethod
    def load(cls, path: str) -> "PolicyIndex":
        """Restore a saved index from disk."""
        faiss = _get_faiss()
        obj = cls()
        obj.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            obj.clauses = pickle.load(f)
        return obj
