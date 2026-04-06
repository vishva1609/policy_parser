"""Unit tests for GapDetector threshold behaviour (no LLM / embeddings)."""
import unittest

from app.models.clause_comparator import GapDetector
from app.common.schemas import ClauseMatch, ComplianceStatus, PolicyClause


def _clause(title: str = "Test") -> PolicyClause:
    return PolicyClause(title=title, text="Must encrypt data at rest.")


def _match(sim: float) -> ClauseMatch:
    return ClauseMatch(
        text="We use encryption.",
        similarity_score=sim,
        clause_id="B-001",
        filename="b.pdf",
        clause_title="Encryption",
        policy_label="B",
    )


class TestGapDetector(unittest.TestCase):
    def test_default_missing_rule(self):
        d = GapDetector()
        c = _clause()
        candidates = [_match(0.2)]
        llm = {"status": "Compliant", "reason": "", "gap": "", "best_candidate_index": 1}
        r = d.detect(c, candidates, llm)
        self.assertEqual(r.status, ComplianceStatus.MISSING)

    def test_custom_missing_threshold_more_lenient(self):
        d = GapDetector(missing_threshold=0.1)
        c = _clause()
        candidates = [_match(0.2)]
        llm = {"status": "Compliant", "reason": "", "gap": "", "best_candidate_index": 1}
        r = d.detect(c, candidates, llm)
        self.assertEqual(r.status, ComplianceStatus.COMPLIANT)

    def test_gap_threshold_non_compliant(self):
        d = GapDetector(gap_threshold=0.6)
        c = _clause()
        candidates = [_match(0.45)]
        llm = {
            "status": "Non-Compliant",
            "reason": "Weak",
            "gap": "x",
            "best_candidate_index": 1,
        }
        r = d.detect(c, candidates, llm)
        self.assertEqual(r.status, ComplianceStatus.NON_COMPLIANT)

    def test_gap_threshold_high_similarity_trusts_llm_elsewhere(self):
        d = GapDetector(gap_threshold=0.6)
        c = _clause()
        candidates = [_match(0.8)]
        llm = {
            "status": "Partial",
            "reason": "x",
            "gap": "",
            "best_candidate_index": 1,
        }
        r = d.detect(c, candidates, llm)
        self.assertEqual(r.status, ComplianceStatus.PARTIAL)


if __name__ == "__main__":
    unittest.main()
