"""
compliance_engine/scorer.py
────────────────────────────
Steps 6 + 7: Gap Detection Engine + Scoring System

Given a list of per-clause comparison results (from comparator.py),
produces:
  • overall compliance %
  • per-category scores
  • critical gaps (mandatory + No/Partial)
  • partial matches list
  • missing clause list
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring weights
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_SCORE = {
    "Yes": 1.0,
    "Partial": 0.5,
    "No": 0.0,
}

_STRENGTH_WEIGHT = {
    "mandatory":    1.5,   # heavier penalty for missing mandatory requirements
    "recommended":  1.0,
    "informational": 0.6,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Gap detection
# ─────────────────────────────────────────────────────────────────────────────

def classify_gap(result: dict) -> str:
    """
    Returns one of:
      "missing"  – status=No (no meaningful match at all)
      "gap"      – status=No|Partial AND strength=mandatory
      "partial"  – status=Partial
      "compliant" – status=Yes
    """
    status = result.get("status", "No")
    strength = result.get("strength", "informational")

    if status == "Yes":
        return "compliant"
    if status == "Partial":
        if strength == "mandatory":
            return "gap"
        return "partial"
    # status == "No"
    if strength == "mandatory":
        return "gap"
    if result.get("best_similarity", 0) < 0.30:
        return "missing"
    return "gap"


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(comparison_results: list[dict]) -> dict:
    """
    Aggregate comparison results into a compliance report.

    Parameters
    ----------
    comparison_results : list of dicts from comparator.compare_clause()

    Returns
    -------
    dict with:
      overall_compliance_pct  : float  (0-100)
      total_clauses           : int
      compliant               : int
      partial_matches         : int
      gaps                    : int (mandatory + partial/no)
      missing                 : int (no match at all)
      critical_gaps           : list[dict]
      partial_matches_list    : list[dict]
      missing_clauses         : list[dict]
      category_scores         : dict[str, dict]
      clause_details          : list[dict]   (full result + gap_type)
    """
    if not comparison_results:
        return {
            "overall_compliance_pct": 0.0,
            "total_clauses": 0,
            "compliant": 0,
            "partial_matches": 0,
            "gaps": 0,
            "missing": 0,
            "critical_gaps": [],
            "partial_matches_list": [],
            "missing_clauses": [],
            "category_scores": {},
            "clause_details": [],
        }

    weighted_sum = 0.0
    weight_total = 0.0

    compliant_count = 0
    partial_count   = 0
    gap_count       = 0
    missing_count   = 0

    critical_gaps:     list[dict] = []
    partial_list:      list[dict] = []
    missing_list:      list[dict] = []

    # Per-category accumulators: {cat: {"score_sum": float, "weight_sum": float}}
    cat_acc: dict[str, dict] = {}

    clause_details: list[dict] = []

    for r in comparison_results:
        gap_type  = classify_gap(r)
        status    = r.get("status", "No")
        strength  = r.get("strength", "informational")
        category  = r.get("category", "General")

        score  = _STATUS_SCORE.get(status, 0.0)
        weight = _STRENGTH_WEIGHT.get(strength, 1.0)

        weighted_sum  += score * weight
        weight_total  += weight

        # Category accumulation
        if category not in cat_acc:
            cat_acc[category] = {"score_sum": 0.0, "weight_sum": 0.0}
        cat_acc[category]["score_sum"]  += score * weight
        cat_acc[category]["weight_sum"] += weight

        # Count types
        if gap_type == "compliant":
            compliant_count += 1
        elif gap_type == "partial":
            partial_count += 1
            partial_list.append(_summary(r, gap_type))
        elif gap_type == "gap":
            gap_count += 1
            critical_gaps.append(_summary(r, gap_type))
        elif gap_type == "missing":
            missing_count += 1
            missing_list.append(_summary(r, gap_type))

        clause_details.append({**r, "gap_type": gap_type})

    overall_pct = round((weighted_sum / weight_total) * 100, 2) if weight_total else 0.0

    category_scores = {
        cat: {
            "compliance_pct": round(
                (v["score_sum"] / v["weight_sum"]) * 100, 2
            ) if v["weight_sum"] else 0.0,
        }
        for cat, v in cat_acc.items()
    }

    return {
        "overall_compliance_pct": overall_pct,
        "total_clauses":          len(comparison_results),
        "compliant":              compliant_count,
        "partial_matches":        partial_count,
        "gaps":                   gap_count,
        "missing":                missing_count,
        "critical_gaps":          critical_gaps,
        "partial_matches_list":   partial_list,
        "missing_clauses":        missing_list,
        "category_scores":        category_scores,
        "clause_details":         clause_details,
    }


def _summary(r: dict, gap_type: str) -> dict:
    return {
        "clause_a_id":    r.get("clause_a_id", ""),
        "clause_a_title": r.get("clause_a_title", ""),
        "status":         r.get("status", ""),
        "strength":       r.get("strength", ""),
        "category":       r.get("category", ""),
        "gap_type":       gap_type,
        "gap":            r.get("gap", ""),
        "reason":         r.get("reason", ""),
        "confidence":     r.get("confidence", 0.0),
        "best_match_id":  r.get("best_match_id", ""),
        "best_similarity": r.get("best_similarity", 0.0),
    }
