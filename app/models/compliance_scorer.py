"""
ComplianceScorer — Aggregates ComplianceResult lists into summary scores.

Moved from src/compliance_scorer.py to app/models/ — it operates on
PolicyClause/ComplianceResult domain objects, making it a model-layer concern.
"""
from typing import List, Dict

from ..common.schemas import ComplianceResult, ComplianceStatus, ComplianceScore


class ComplianceScorer:
    """
    Converts a list of ComplianceResult objects into a ComplianceScore summary.

    Scoring weights:
        Compliant     → 1.0  (full credit)
        Partial       → 0.5  (half credit)
        Non-Compliant → 0.0
        Missing       → 0.0
    """

    WEIGHTS: Dict[ComplianceStatus, float] = {
        ComplianceStatus.COMPLIANT:     1.0,
        ComplianceStatus.PARTIAL:       0.5,
        ComplianceStatus.NON_COMPLIANT: 0.0,
        ComplianceStatus.MISSING:       0.0,
    }

    def score(
        self,
        results: List[ComplianceResult],
        direction: str = "A→B",
    ) -> ComplianceScore:
        if not results:
            return ComplianceScore(
                overall_pct=0.0,
                compliant_count=0,
                partial_count=0,
                non_compliant_count=0,
                missing_count=0,
                critical_gaps=0,
                scores_by_category={},
                total_clauses=0,
                direction=direction,
            )

        counts: Dict[ComplianceStatus, int] = {s: 0 for s in ComplianceStatus}
        for r in results:
            counts[r.status] += 1

        total_weight = sum(self.WEIGHTS[r.status] for r in results)
        overall_pct  = round((total_weight / len(results)) * 100, 1)

        critical_gaps = sum(
            1 for r in results
            if r.status in {ComplianceStatus.NON_COMPLIANT, ComplianceStatus.MISSING}
            and r.clause_a.is_requirement
        )

        return ComplianceScore(
            overall_pct=overall_pct,
            compliant_count=counts[ComplianceStatus.COMPLIANT],
            partial_count=counts[ComplianceStatus.PARTIAL],
            non_compliant_count=counts[ComplianceStatus.NON_COMPLIANT],
            missing_count=counts[ComplianceStatus.MISSING],
            critical_gaps=critical_gaps,
            scores_by_category=self._by_category(results),
            total_clauses=len(results),
            direction=direction,
        )

    def _by_category(self, results: List[ComplianceResult]) -> Dict[str, float]:
        groups: Dict[str, List[ComplianceResult]] = {}
        for r in results:
            cat = r.clause_a.category or "General"
            groups.setdefault(cat, []).append(r)

        return {
            cat: round(
                sum(self.WEIGHTS[r.status] for r in cat_results)
                / len(cat_results) * 100,
                1,
            )
            for cat, cat_results in groups.items()
        }

    @staticmethod
    def print_summary(score: ComplianceScore) -> None:
        print(f"\n{'='*55}")
        print(f"  Compliance Score ({score.direction})")
        print(f"{'='*55}")
        print(f"  Overall        : {score.overall_pct}%")
        print(f"  Total Clauses  : {score.total_clauses}")
        print(f"  ✅ Compliant   : {score.compliant_count}")
        print(f"  ⚠️  Partial     : {score.partial_count}")
        print(f"  ❌ Non-Compliant: {score.non_compliant_count}")
        print(f"  ➕ Missing      : {score.missing_count}")
        print(f"  🚨 Critical Gaps: {score.critical_gaps}")
        if score.scores_by_category:
            print(f"\n  By Category:")
            for cat, pct in sorted(score.scores_by_category.items()):
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"    {cat:<28} {bar} {pct}%")
        print(f"{'='*55}\n")
