"""
compliance_engine/comparator.py
────────────────────────────────
Steps 4 + 5: Clause Matching & LLM Comparison

For each clause in Policy-A, finds top matches in Policy-B (via FAISS),
then uses an LLM (OpenAI GPT or a local Ollama model) to produce a
structured comparison result.

Hybrid logic overlay (Step 4 extra):
  • Detects "must" vs "should" strength  →  mandatory vs recommended
  • Tags policy category based on keyword patterns
"""

import json
import os
import re
from typing import Any
from pathlib import Path

# Load config
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
def _load_config():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

CONF = _load_config()
LLM_CONF = CONF.get("llm", {})
MATCH_CONF = CONF.get("matching", {})

from .clause_embedding import PolicyIndex

# ─────────────────────────────────────────────────────────────────────────────
#  Category detection
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "Security": [
        r"\bencrypt\w*\b", r"\baes\b", r"\bssl\b", r"\btls\b", r"\bfirewall\b",
        r"\bpassword\b", r"\bauthenticat\w*\b", r"\bvulnerabilit\w*\b", r"\bpentest\b",
        r"\bincident\b", r"\bbreach\b",
    ],
    "Access Control": [
        r"\baccess control\b", r"\brole\b", r"\bprivilege\b", r"\bpermission\b",
        r"\bleast privilege\b", r"\bmfa\b", r"\bmulti.factor\b", r"\bauthori[sz]ation\b",
        r"\bidentity\b", r"\biam\b",
    ],
    "Logging": [
        r"\baudit log\b", r"\blog\w*\b", r"\bmonitor\w*\b", r"\btrack\w*\b",
        r"\btrail\b", r"\bevent\b", r"\bsiem\b", r"\bretention\b",
    ],
    "Data Privacy": [
        r"\bgdpr\b", r"\bpersonal data\b", r"\bpii\b", r"\bprivacy\b",
        r"\bdata subject\b", r"\bconsent\b", r"\bdata protection\b",
        r"\banonymis\w*\b", r"\bpseudonym\w*\b",
    ],
}

_STRENGTH_RE = re.compile(r"\b(must|shall|required|mandatory)\b", re.I)
_RECOMMENDED_RE = re.compile(r"\b(should|recommended|encouraged|may)\b", re.I)


def detect_category(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for cat, patterns in _CATEGORY_PATTERNS.items():
        scores[cat] = sum(1 for p in patterns if re.search(p, text_lower, re.I))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "General"


def detect_strength(text: str) -> str:
    if _STRENGTH_RE.search(text):
        return "mandatory"
    if _RECOMMENDED_RE.search(text):
        return "recommended"
    return "informational"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM backends
# ─────────────────────────────────────────────────────────────────────────────

_COMPARISON_PROMPT = """\
You are a policy compliance expert. Compare the following two policy clauses and return ONLY valid JSON.

Policy A clause (source):
ID: {id_a}
Title: {title_a}
Text: {text_a}

Policy B clause (best match, similarity {sim:.2f}):
ID: {id_b}
Title: {title_b}
Text: {text_b}

Return this exact JSON structure:
{{
  "status": "Yes|No|Partial",
  "reason": "one-sentence explanation",
  "gap": "description of what is missing or different; empty string if status=Yes",
  "confidence": 0.0
}}

Rules:
- status=Yes  → Policy B fully covers the requirement in Policy A
- status=Partial → Policy B partially covers it (gap is notable)
- status=No → Policy B has no meaningful coverage of Policy A's requirement
- confidence is a float 0-1 (your certainty about the classification)
"""


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """Call OpenAI Chat API (requires OPENAI_API_KEY env var)."""
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        return {"error": str(exc)}


def _call_ollama(prompt: str, model: str = "llama3") -> dict:
    """Call local Ollama server (no API key needed)."""
    import requests
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        resp = requests.post("http://localhost:11434/api/generate",
                             json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        # Ollama sometimes wraps in markdown fences – strip them
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


def _rule_based_fallback(clause_a: dict, clause_b: dict, sim: float) -> dict:
    """
    Purely rule-based comparison when no LLM is available.
    Uses category match + strength + similarity score.
    """
    cat_match = detect_category(clause_a.get("text", "")) == detect_category(clause_b.get("text", ""))
    strength_a = detect_strength(clause_a.get("text", ""))

    if sim >= 0.80 and cat_match:
        status = "Yes"
        reason = f"High semantic similarity ({sim:.2f}) and same policy category."
        gap = ""
        confidence = min(0.95, sim)
    elif sim >= 0.55:
        status = "Partial"
        reason = f"Moderate similarity ({sim:.2f}); some aspects may differ."
        gap = "Review the specific obligations and thresholds between both policies."
        confidence = 0.65
    else:
        status = "No"
        reason = f"Low similarity ({sim:.2f}); likely a coverage gap."
        gap = f"Policy B has no equivalent clause for '{clause_a.get('title', clause_a.get('clause_id', ''))}'"
        if strength_a == "mandatory":
            gap += " (this is a MANDATORY requirement in Policy A)."
        confidence = 0.70

    return {
        "status": status,
        "reason": reason,
        "gap": gap,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main comparison function
# ─────────────────────────────────────────────────────────────────────────────

def compare_clause(
    clause_a: dict,
    index_b: PolicyIndex,
    top_k: int = 5,
    llm_backend: str = "rules",   # "openai" | "ollama" | "rules"
    llm_model: str | None = None,
) -> dict:
    """
    For a single clause from Policy A, find best matches in Policy B
    and return a comprehensive comparison result.
    """
    matches = index_b.search(clause_a, top_k=top_k)

    # Category + strength enrichment
    clause_text = clause_a.get("text", "")
    category = detect_category(clause_text)
    strength = detect_strength(clause_text)

    if not matches:
        return {
            "clause_a_id":   clause_a.get("clause_id", ""),
            "clause_a_title": clause_a.get("title", ""),
            "category":      category,
            "strength":      strength,
            "top_matches":   [],
            "status":        "No",
            "reason":        "No matching clauses found in Policy B.",
            "gap":           f"Policy B is missing any coverage for '{clause_a.get('title', '')}'",
            "confidence":    1.0,
        }

    best = matches[0]
    sim = best.get("similarity_score", 0.0)

    # ── LLM comparison ──────────────────────────────────────────────────────
    llm_result: dict = {}

    if llm_backend == "openai" and os.environ.get("OPENAI_API_KEY"):
        prompt = _COMPARISON_PROMPT.format(
            id_a=clause_a.get("clause_id", ""), title_a=clause_a.get("title", ""),
            text_a=clause_a.get("text", "")[:800],
            id_b=best.get("clause_id", ""),    title_b=best.get("title", ""),
            text_b=best.get("text", "")[:800],
            sim=sim,
        )
        llm_result = _call_openai(prompt, model=llm_model or "gpt-4o-mini")

    elif llm_backend == "ollama":
        prompt = _COMPARISON_PROMPT.format(
            id_a=clause_a.get("clause_id", ""), title_a=clause_a.get("title", ""),
            text_a=clause_a.get("text", "")[:800],
            id_b=best.get("clause_id", ""),    title_b=best.get("title", ""),
            text_b=best.get("text", "")[:800],
            sim=sim,
        )
        llm_result = _call_ollama(prompt, model=llm_model or "llama3")

    # Fall through to rules if LLM returned an error or wasn't selected
    if not llm_result or "error" in llm_result:
        llm_result = _rule_based_fallback(clause_a, best, sim)

    return {
        "clause_a_id":    clause_a.get("clause_id", ""),
        "clause_a_title": clause_a.get("title", ""),
        "category":       category,
        "strength":       strength,
        "top_matches": [
            {
                "clause_b_id":    m.get("clause_id", ""),
                "clause_b_title": m.get("title", ""),
                "similarity":     round(m.get("similarity_score", 0.0), 4),
            }
            for m in matches
        ],
        "best_match_id":    best.get("clause_id", ""),
        "best_match_title": best.get("title", ""),
        "best_similarity":  round(sim, 4),
        "status":           llm_result.get("status", "No"),
        "reason":           llm_result.get("reason", ""),
        "gap":              llm_result.get("gap", ""),
        "confidence":       round(float(llm_result.get("confidence", 0.5)), 3),
    }
