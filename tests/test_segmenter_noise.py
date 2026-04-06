"""Regression tests for clause segmenter PDF-artifact cleanup."""
import unittest

from app.models.clause_segmenter import ClauseSegmenter
from app.common.schemas import DocumentElement, DocumentSection


class TestSegmenterNoise(unittest.TestCase):
    def test_skips_lone_numbering_and_toc(self):
        seg = ClauseSegmenter(use_llm_fallback=False)
        section = DocumentSection(
            title="POLICY",
            level=1,
            page_start=1,
            page_end=1,
            elements=[
                DocumentElement(
                    text="1.",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 12},
                ),
                DocumentElement(
                    text="Table of Contents",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 12},
                ),
                DocumentElement(
                    text="Scope",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 16},
                ),
                DocumentElement(
                    text="All users must protect data.",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 12},
                ),
                DocumentElement(
                    text="•",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 12},
                ),
                DocumentElement(
                    text="Access must be reviewed quarterly.",
                    element_type="paragraph",
                    page=1,
                    metadata={"font_size": 12},
                ),
            ],
        )
        clauses = seg._segment_section(section)
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].title, "Scope")
        self.assertIn("protect data", clauses[0].text)
        self.assertEqual(len(clauses[0].sub_clauses), 1)
        self.assertIn("quarterly", clauses[0].sub_clauses[0].text)


if __name__ == "__main__":
    unittest.main()
