"""
Excel Utilities — Consolidated Excel output helpers.

Merges what was previously split across:
    src/excel_generator.py     — ExcelGenerator (document index)
    src/comparison_report.py   — ComparisonReportGenerator (compliance report)

Both are output/presentation classes with no business logic, so they
belong together in common/ as shared utilities — matching the pattern of
common/openapi.py in archit1012/qa-bot-llm which centralised all output
helpers into one place.
"""
import re
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

from .schemas import (
    ParsedDocument, DocumentChunk,
    ComplianceResult, ComplianceStatus, PolicyComparisonReport,
)


# ── Shared text helpers ────────────────────────────────────────────────────────

def clean_text_for_excel(text: str) -> str:
    """Remove illegal characters from Excel cells — shared utility."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
    return cleaned


def _clean(text, max_len: int = 500) -> str:
    """Strip control chars and truncate for Excel cells."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text[:max_len]


# ── Compliance question export ─────────────────────────────────────────────────

def export_compliance_questions_to_excel(questions, statements, output_file):
    """
    Export generated compliance questions to Excel with multiple sheets.

    Args:
        questions:   List of ComplianceQuestion objects
        statements:  List of PolicyStatement objects
        output_file: Path to output Excel file
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        questions_data = [{
            'Sr No.': idx + 1,
            'Question': clean_text_for_excel(q.question),
            'Category': q.category,
            'Type': q.question_type,
            'Expected Evidence': clean_text_for_excel(q.expected_evidence),
            'Rationale': clean_text_for_excel(q.rationale),
            'Section': clean_text_for_excel(q.section),
            'Page': q.page,
            'Source': clean_text_for_excel(
                q.source_statement[:150] + '...'
                if len(q.source_statement) > 150 else q.source_statement
            )
        } for idx, q in enumerate(questions)]

        df_questions = pd.DataFrame(questions_data)
        df_questions.to_excel(writer, sheet_name='Questions', index=False)

        category_summary = df_questions.groupby('Category').size().reset_index(name='Count')
        category_summary.to_excel(writer, sheet_name='By Category', index=False)

        type_summary = df_questions.groupby('Type').size().reset_index(name='Count')
        type_summary.to_excel(writer, sheet_name='By Type', index=False)

        requirements_data = [{
            'Statement': clean_text_for_excel(s.text),
            'Category': s.category.value,
            'Section': clean_text_for_excel(s.section),
            'Page': s.page,
            'Confidence': s.confidence
        } for s in statements]

        df_requirements = pd.DataFrame(requirements_data)
        df_requirements.to_excel(writer, sheet_name='Requirements', index=False)

    # Auto-adjust column widths
    wb = load_workbook(output_file)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except Exception:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 100)
    wb.save(output_file)


# ── Document index Excel generator ────────────────────────────────────────────

class ExcelGenerator:
    """
    Generate Excel file with searchable key-value pairs and page numbers.

    Moved from src/excel_generator.py into common/ — it is a pure
    presentation/output class with no business logic, so it belongs here
    alongside other shared utilities.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_searchable_index(
        self,
        document: ParsedDocument,
        chunks: List[DocumentChunk],
        output_filename: str = None,
    ) -> str:
        """
        Generate Excel file with searchable content and page numbers.

        Returns:
            Path to generated Excel file
        """
        if output_filename is None:
            base_name = Path(document.filename).stem
            output_filename = f"{base_name}_searchable_index.xlsx"

        excel_path = self.output_dir / output_filename
        rows = []

        rows.append({
            'Type': 'Document',
            'Section': 'Metadata',
            'Content': f"Document: {document.filename}",
            'Page Numbers': f"1-{document.total_pages}",
            'Chunk ID': document.document_id,
            'Keywords': 'document, metadata'
        })

        for section in document.sections:
            rows.append({
                'Type': 'Section',
                'Section': section.title,
                'Content': f"Section: {section.title}",
                'Page Numbers': f"{section.page_start}-{section.page_end}",
                'Chunk ID': '',
                'Keywords': section.title.lower()
            })

        for i, chunk in enumerate(chunks, 1):
            clean_text = clean_text_for_excel(chunk.text)
            words = clean_text.split()
            keywords = ' '.join([w.lower() for w in words[:10] if len(w) > 3])

            if chunk.pages:
                page_range = (
                    f"{min(chunk.pages)}-{max(chunk.pages)}"
                    if len(chunk.pages) > 1
                    else str(chunk.pages[0])
                )
            else:
                page_range = "N/A"

            section_name = clean_text_for_excel(
                chunk.parent_context[0] if chunk.parent_context else 'N/A'
            )
            rows.append({
                'Type': 'Chunk',
                'Section': section_name,
                'Content': clean_text[:500] + '...' if len(clean_text) > 500 else clean_text,
                'Page Numbers': page_range,
                'Chunk ID': chunk.chunk_id,
                'Keywords': keywords
            })

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Searchable Index', index=False)

            summary_data = {
                'Metric': ['Total Pages', 'Total Sections', 'Total Chunks', 'Document Name'],
                'Value': [document.total_pages, len(document.sections), len(chunks), document.filename]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            page_mapping = []
            for chunk in chunks:
                if chunk.pages:
                    for page in chunk.pages:
                        sn = clean_text_for_excel(
                            chunk.parent_context[0] if chunk.parent_context else 'N/A'
                        )
                        page_mapping.append({
                            'Page Number': page,
                            'Section': sn,
                            'Chunk ID': chunk.chunk_id,
                            'Content Preview': clean_text_for_excel(chunk.text[:200]) + '...'
                        })

            if page_mapping:
                page_df = pd.DataFrame(page_mapping).sort_values('Page Number')
                page_df.to_excel(writer, sheet_name='Page Mapping', index=False)

        print(f"Excel index generated: {excel_path}")
        return str(excel_path)

    def search_excel(self, excel_path: str, search_term: str) -> pd.DataFrame:
        """Search the Excel file for a term and return matching rows."""
        df = pd.read_excel(excel_path, sheet_name='Searchable Index')
        mask = (
            df['Content'].str.contains(search_term, case=False, na=False) |
            df['Keywords'].str.contains(search_term, case=False, na=False)
        )
        return df[mask]


# ── Colour palette ─────────────────────────────────────────────────────────────

_C = {
    "compliant":     "C6EFCE",  # light green
    "partial":       "FFEB9C",  # light yellow
    "non_compliant": "FFC7CE",  # light red
    "missing":       "F4CCCC",  # pale red
    "header_bg":     "2E4057",  # dark blue-grey
    "header_fg":     "FFFFFF",  # white
}

STATUS_FILL = {
    ComplianceStatus.COMPLIANT:     _C["compliant"],
    ComplianceStatus.PARTIAL:       _C["partial"],
    ComplianceStatus.NON_COMPLIANT: _C["non_compliant"],
    ComplianceStatus.MISSING:       _C["missing"],
}

STATUS_LABEL = {
    ComplianceStatus.COMPLIANT:     "✅ Compliant",
    ComplianceStatus.PARTIAL:       "⚠️ Partial",
    ComplianceStatus.NON_COMPLIANT: "❌ Non-Compliant",
    ComplianceStatus.MISSING:       "➕ Missing",
}


# ── Compliance comparison report ───────────────────────────────────────────────

class ComparisonReportGenerator:
    """
    Generates a comprehensive color-coded Excel compliance report.

    Moved from src/comparison_report.py into common/ — pure output class
    with no business logic, it belongs with other shared Excel utilities.

    Sheets produced:
        📊 Dashboard          — score at a glance (both directions)
        A→B Comparison        — full clause-by-clause detail
        B→A Comparison        — reverse direction
        🔴 Gaps Only          — filtered non-compliant + missing rows
        📈 Category Breakdown — per-category compliance %
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, report: PolicyComparisonReport, filename: str = None) -> str:
        """
        Write the full Excel report and return its absolute path.

        Args:
            report:   PolicyComparisonReport containing both direction results.
            filename: Optional custom output filename.

        Returns:
            Absolute path to the generated .xlsx file.
        """
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compliance_report_{ts}.xlsx"

        out = self.output_dir / filename
        print(f"\n[Report] Generating: {filename}")

        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            self._sheet_dashboard(writer, report)
            self._sheet_comparison(writer, report.results_a_to_b, "A→B Comparison")
            self._sheet_comparison(writer, report.results_b_to_a, "B→A Comparison")
            self._sheet_gaps(writer, report)
            self._sheet_categories(writer, report)

        self._format(out)
        print(f"[Report] Saved → {out}")
        return str(out)

    def _sheet_dashboard(self, writer, report: PolicyComparisonReport):
        sa = report.score_a_to_b
        sb = report.score_b_to_a

        def val(score, field):
            return getattr(score, field, "N/A") if score else "N/A"

        rows = [
            {"Metric": "Generated",           "A→B": report.timestamp,            "B→A": ""},
            {"Metric": "Policy A",             "A→B": report.policy_a_name,        "B→A": ""},
            {"Metric": "Policy B",             "A→B": report.policy_b_name,        "B→A": ""},
            {"Metric": "",                     "A→B": "",                           "B→A": ""},
            {"Metric": "Total Clauses",        "A→B": val(sa, "total_clauses"),     "B→A": val(sb, "total_clauses")},
            {"Metric": "Overall Compliance %", "A→B": f"{val(sa,'overall_pct')}%", "B→A": f"{val(sb,'overall_pct')}%"},
            {"Metric": "✅ Compliant",          "A→B": val(sa, "compliant_count"),   "B→A": val(sb, "compliant_count")},
            {"Metric": "⚠️  Partial",           "A→B": val(sa, "partial_count"),     "B→A": val(sb, "partial_count")},
            {"Metric": "❌ Non-Compliant",      "A→B": val(sa, "non_compliant_count"), "B→A": val(sb, "non_compliant_count")},
            {"Metric": "➕ Missing",             "A→B": val(sa, "missing_count"),     "B→A": val(sb, "missing_count")},
            {"Metric": "🚨 Critical Gaps",      "A→B": val(sa, "critical_gaps"),     "B→A": val(sb, "critical_gaps")},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="📊 Dashboard", index=False)

    def _sheet_comparison(self, writer, results: List[ComplianceResult], sheet_name: str):
        if not results:
            pd.DataFrame([{"Note": f"No results for {sheet_name}"}]).to_excel(
                writer, sheet_name=sheet_name, index=False
            )
            return

        rows = []
        for i, r in enumerate(results, 1):
            bm_text = bm_section = bm_sim = bm_file = bm_title = ""
            if r.best_match:
                bm_text    = _clean(r.best_match.text, 400)
                bm_section = _clean(r.best_match.section)
                bm_sim     = f"{r.best_match.similarity_score:.2%}"
                bm_file    = _clean(r.best_match.filename)
                bm_title   = _clean(r.best_match.clause_title, 150)

            rows.append({
                "#":                   i,
                "Clause ID":           r.clause_a.clause_id,
                "Clause Number":       r.clause_a.clause_number,
                "Title":               _clean(r.clause_a.title, 150),
                "Source File":         _clean(r.clause_a.filename),
                "Source Text":         _clean(r.clause_a.text, 500),
                "Sub-clauses":         len(r.clause_a.sub_clauses),
                "Category":            r.clause_a.category,
                "Requirement?":        "Yes" if r.clause_a.is_requirement else "No",
                "Status":              STATUS_LABEL[r.status],
                "Confidence":          f"{r.confidence:.0%}",
                "Match Clause ID":     r.best_match.clause_id if r.best_match else "",
                "Match Clause Title":  bm_title,
                "Match File":          bm_file,
                "Best Match Text":     bm_text,
                "Best Match Section":  bm_section,
                "Similarity":          bm_sim,
                "Reason":              _clean(r.reason, 400),
                "Gap / Missing Req.":  _clean(r.gap_description, 300),
                "Evidence Quote":      _clean(r.evidence_quote, 300),
            })

        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    def _sheet_gaps(self, writer, report: PolicyComparisonReport):
        gap_statuses = {ComplianceStatus.NON_COMPLIANT, ComplianceStatus.MISSING}
        rows = []

        for direction, results in [("A→B", report.results_a_to_b), ("B→A", report.results_b_to_a)]:
            for r in results:
                if r.status in gap_statuses:
                    rows.append({
                        "Direction":       direction,
                        "Clause ID":       r.clause_a.clause_id,
                        "Clause Number":   r.clause_a.clause_number,
                        "Title":           _clean(r.clause_a.title, 150),
                        "Source File":     _clean(r.clause_a.filename),
                        "Status":          STATUS_LABEL[r.status],
                        "Critical?":       "🚨 YES" if r.clause_a.is_requirement else "No",
                        "Category":        r.clause_a.category,
                        "Sub-clauses":     len(r.clause_a.sub_clauses),
                        "Gap Description": _clean(r.gap_description, 400),
                        "Reason":          _clean(r.reason, 400),
                        "Source Text":     _clean(r.clause_a.text, 400),
                        "Match File":      _clean(r.best_match.filename) if r.best_match else "",
                    })

        if not rows:
            rows = [{"Note": "🎉 No gaps found! Full compliance achieved."}]

        pd.DataFrame(rows).to_excel(writer, sheet_name="🔴 Gaps Only", index=False)

    def _sheet_categories(self, writer, report: PolicyComparisonReport):
        sa = report.score_a_to_b
        sb = report.score_b_to_a

        all_cats = set()
        if sa:
            all_cats.update(sa.scores_by_category)
        if sb:
            all_cats.update(sb.scores_by_category)

        rows = []
        for cat in sorted(all_cats):
            a = sa.scores_by_category.get(cat) if sa else None
            b = sb.scores_by_category.get(cat) if sb else None
            avg = round(
                ((a or 0) + (b or 0)) / (2 if a is not None and b is not None else 1),
                1,
            )
            rows.append({
                "Category":         cat,
                "A→B Compliance %": f"{a}%" if a is not None else "N/A",
                "B→A Compliance %": f"{b}%" if b is not None else "N/A",
                "Average %":        f"{avg}%",
            })

        if not rows:
            rows = [{"Category": "No data", "A→B Compliance %": "N/A",
                     "B→A Compliance %": "N/A", "Average %": "N/A"}]

        pd.DataFrame(rows).to_excel(writer, sheet_name="📈 Category Breakdown", index=False)

    def _format(self, path: Path):
        wb = load_workbook(path)
        header_fill = PatternFill("solid", fgColor=_C["header_bg"])
        header_font = Font(bold=True, color=_C["header_fg"], size=11)

        for ws in wb.worksheets:
            for cell in ws[1]:
                cell.fill      = header_fill
                cell.font      = header_font
                cell.alignment = Alignment(
                    horizontal="center", vertical="center", wrap_text=True
                )

            if "Comparison" in ws.title or "Gaps" in ws.title:
                self._color_rows(ws)

            for col in ws.columns:
                letter = get_column_letter(col[0].column)
                max_w  = max((len(str(c.value or "")) for c in col), default=8)
                ws.column_dimensions[letter].width = min(max_w + 4, 60)

            ws.freeze_panes = "A2"

        wb.save(path)

    def _color_rows(self, ws):
        """Identify the 'Status' column and apply per-row background color."""
        status_col = None
        for i, cell in enumerate(ws[1], 1):
            if "Status" in str(cell.value or ""):
                status_col = i
                break
        if status_col is None:
            return

        status_to_fill = {
            label: PatternFill("solid", fgColor=color)
            for status, color in STATUS_FILL.items()
            for label in [STATUS_LABEL[status]]
        }

        for row in ws.iter_rows(min_row=2):
            cell_val = str(row[status_col - 1].value or "")
            fill = status_to_fill.get(cell_val)
            if fill:
                for cell in row:
                    cell.fill = fill
