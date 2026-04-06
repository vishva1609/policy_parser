"""
PolicyComparisonPipeline — Comparison service.

Mirrors qa_apis_service.py from archit1012/qa-bot-llm, applying the same
Dependency Injection pattern:

    # qa_apis_service.py creates the processor, calls its ABC methods:
    def process_request(request, embeddings, chat):  ← chat INJECTED
        ...
        response = get_response_from_query(db, query, chat)  ← injected

    # Our pipeline accepts injected MistralConfig:
    pipeline = PolicyComparisonPipeline(config=mistral_config)  ← INJECTED

All imports updated to app.* paths.
No algorithmic changes — purely structural injection.
"""
import os
import json
import time
import uuid
from datetime import datetime
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..common.schemas import (
    PolicyClause,
    ParsedDocument,
    ComplianceResult,
    ComplianceScore,
    PolicyComparisonReport,
    DocumentChunk,
    ClauseVectorMetadata,
)

# Stepwise session artifacts (under pipeline ``output_dir``)
FN_PARSED_A = "parsed_policy_a.json"
FN_PARSED_B = "parsed_policy_b.json"
FN_CLAUSES_A = "clauses_policy_a.json"
FN_CLAUSES_B = "clauses_policy_b.json"
FN_RESULTS_AB = "comparison_results_a_to_b.json"
FN_RESULTS_BA = "comparison_results_b_to_a.json"
FN_SCORES = "comparison_scores.json"
from ..models.document_processor import SimplePDFParser, DocumentChunker
from ..models.embedder import EmbeddingPipeline
from ..models.clause_segmenter import ClauseSegmenter
from ..models.clause_comparator import (
    ClauseMatcher, ComplianceReasoner, GapDetector,
    GAP_THRESHOLD, MISSING_THRESHOLD,
)
from ..models.compliance_scorer import ComplianceScorer
from ..common.excel_utils import ComparisonReportGenerator
from ..common.llm_client import MistralConfig, LANGCHAIN_AVAILABLE

# PolicyAnalyzer — now lives in app/models/
try:
    from app.models.policy_analyzer import PolicyAnalyzer
    _POLICY_ANALYZER_AVAILABLE = True
except ImportError:
    _POLICY_ANALYZER_AVAILABLE = False


class PolicyComparisonPipeline:
    """
    End-to-end bidirectional policy compliance comparison.

    Moved from src/comparison_pipeline.py to app/service/comparison_service.py.
    API is identical — no changes to constructor arguments or public methods.

    Usage::

        pipeline = PolicyComparisonPipeline(output_dir="./output")
        report_path = pipeline.compare(
            pdf_a="samples/Policy_A.pdf",
            pdf_b="samples/Policy_B.pdf",
        )
    """

    def __init__(
        self,
        output_dir: str = "./output",
        embedding_model: str = "all-MiniLM-L6-v2",
        mistral_model: str = "mistral-small-latest",
        api_key: str = None,
        top_k_candidates: int = 5,
        use_llm_segmentation: bool = True,
        chunk_by: str = "section",
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 200,
        missing_similarity_threshold: Optional[float] = None,
        gap_similarity_threshold: Optional[float] = None,
        config: Optional[MistralConfig] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialise the comparison pipeline.

        Args:
            ...
            config: Optional pre-built MistralConfig (dependency injection from views).
                    If provided, this is passed into ComplianceReasoner instead of
                    creating a new one — mirrors how qa_apis.py injects 'chat'.
        """
        self.output_dir        = Path(output_dir)
        self.embedding_model   = embedding_model
        self.mistral_model     = mistral_model
        self.api_key           = api_key or os.getenv("MISTRAL_API_KEY")
        self.top_k             = top_k_candidates
        self.use_llm_seg       = use_llm_segmentation
        self.missing_similarity_threshold = (
            MISSING_THRESHOLD if missing_similarity_threshold is None
            else missing_similarity_threshold
        )
        self.gap_similarity_threshold = (
            GAP_THRESHOLD if gap_similarity_threshold is None
            else gap_similarity_threshold
        )
        # Store injected config (may be None if not provided)
        self._config = config
        self.session_id = session_id or ""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._parser   = SimplePDFParser()
        self._chunker  = DocumentChunker(
            chunk_by=chunk_by,
            max_chars=max_chunk_chars,
            min_chars=min_chunk_chars,
        )
        self._scorer   = ComplianceScorer()
        self._reporter = ComparisonReportGenerator(str(self.output_dir))

        print("=" * 60)
        print("  Policy Comparison Pipeline initialised")
        print(f"  Embedding model : {embedding_model}")
        print(f"  LLM model       : {mistral_model}")
        print(f"  Top-k candidates: {top_k_candidates}")
        if config:
            print("  MistralConfig   : injected (dependency injection ✓)")
        print("=" * 60)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_manifest(self, pdf_a: str, pdf_b: str) -> str:
        """
        Parse + segment both PDFs and write output/segment_manifest.json.
        No LLM calls, no API cost.
        """
        pdf_a, pdf_b = Path(pdf_a), Path(pdf_b)
        print(f"\n{'='*60}\n  Building segment manifest (no API calls)…\n{'='*60}\n")

        doc_a = self._parser.parse_pdf(str(pdf_a))
        doc_b = self._parser.parse_pdf(str(pdf_b))

        seg_a = ClauseSegmenter(policy_label="A", policy_id="policy_a", use_llm_fallback=False)
        seg_b = ClauseSegmenter(policy_label="B", policy_id="policy_b", use_llm_fallback=False)
        clauses_a = self._enrich_categories(seg_a.segment(doc_a))
        clauses_b = self._enrich_categories(seg_b.segment(doc_b))

        self._save_segment_manifest(clauses_a, clauses_b, pdf_a.name, pdf_b.name)
        manifest_path = str(self.output_dir / "segment_manifest.json")
        print(f"\n✅  Done — open: {manifest_path}\n")
        return manifest_path

    def compare(
        self,
        pdf_a: str,
        pdf_b: str,
        report_filename: str = None,
    ) -> str:
        """
        Run full bidirectional comparison and produce an Excel report.

        Args:
            pdf_a:           Path to Policy A (source of truth).
            pdf_b:           Path to Policy B (to validate).
            report_filename: Optional output filename.

        Returns:
            Absolute path to the generated Excel report.
        """
        pdf_a, pdf_b = Path(pdf_a), Path(pdf_b)
        print(f"\n{'='*60}\n  Comparing:\n    A: {pdf_a.name}\n    B: {pdf_b.name}\n{'='*60}\n")

        # Step 1: Parse
        print("Step 1/7 — Parsing PDFs…")
        doc_a = self._parser.parse_pdf(str(pdf_a))
        doc_b = self._parser.parse_pdf(str(pdf_b))

        # Step 2: Segment
        print("\nStep 2/7 — Segmenting into clauses…")
        seg_a = ClauseSegmenter(
            policy_label="A",
            policy_id="policy_a",
            api_key=self.api_key,
            model=self.mistral_model,
            use_llm_fallback=self.use_llm_seg,
            config=self._config,
        )
        seg_b = ClauseSegmenter(
            policy_label="B",
            policy_id="policy_b",
            api_key=self.api_key,
            model=self.mistral_model,
            use_llm_fallback=self.use_llm_seg,
            config=self._config,
        )
        clauses_a = self._enrich_categories(seg_a.segment(doc_a))
        clauses_b = self._enrich_categories(seg_b.segment(doc_b))
        print(f"  Policy A clauses: {len(clauses_a)}")
        print(f"  Policy B clauses: {len(clauses_b)}")

        # Step 3: Embed
        print("\nStep 3/7 — Embedding clauses into vector DBs…")
        embedder_a = self._build_embedder("policy_a", clauses_a)
        embedder_b = self._build_embedder("policy_b", clauses_b)
        self._save_segment_manifest(clauses_a, clauses_b, pdf_a.name, pdf_b.name)

        # Step 4 & 5: Bidirectional comparison
        # ComplianceReasoner receives injected config (DI from views layer)
        # Same reasoner instance is reused for both directions — no re-init
        print("\nStep 4/7 — Initialising ComplianceReasoner (injected config)…")
        reasoner = ComplianceReasoner(
            api_key=self.api_key,
            model=self.mistral_model,
            config=self._config,        # ← INJECTED from views layer if available
        )
        detector = GapDetector(
            missing_threshold=self.missing_similarity_threshold,
            gap_threshold=self.gap_similarity_threshold,
        )

        print("\nStep 4/7 — A→B comparison…")
        results_a_to_b = self._run_comparison(clauses_a, embedder_b, reasoner, detector, "A→B")

        print("\nStep 5/7 — B→A comparison (reusing same reasoner — no re-init)…")
        results_b_to_a = self._run_comparison(clauses_b, embedder_a, reasoner, detector, "B→A")

        # Step 6: Score
        print("\nStep 6/7 — Scoring…")
        score_a_to_b = self._scorer.score(results_a_to_b, direction="A→B")
        score_b_to_a = self._scorer.score(results_b_to_a, direction="B→A")
        ComplianceScorer.print_summary(score_a_to_b)
        ComplianceScorer.print_summary(score_b_to_a)

        # Step 7: Export
        print("Step 7/7 — Generating Excel report…")
        report = PolicyComparisonReport(
            policy_a_name=pdf_a.name,
            policy_b_name=pdf_b.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results_a_to_b=results_a_to_b,
            results_b_to_a=results_b_to_a,
            score_a_to_b=score_a_to_b,
            score_b_to_a=score_b_to_a,
        )
        report_path = self._reporter.generate(report, filename=report_filename)
        self._save_json(report, pdf_a.stem, pdf_b.stem)

        print(f"\n{'='*60}\n  ✅ Comparison complete!\n  Excel report : {report_path}\n{'='*60}\n")
        return report_path

    # ── Stepwise API (sessions / debugging) ────────────────────────────────────

    def _is_stub_policy_b(self) -> bool:
        """True if session was created in single-PDF mode (no real policy B)."""
        p = self.output_dir / FN_PARSED_B
        if not p.is_file():
            return False
        doc_b = ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))
        return bool(doc_b.metadata.get("single_file_mode"))

    def _artifact_paths(self) -> Dict[str, Path]:
        return {
            "parsed_a": self.output_dir / FN_PARSED_A,
            "parsed_b": self.output_dir / FN_PARSED_B,
            "clauses_a": self.output_dir / FN_CLAUSES_A,
            "clauses_b": self.output_dir / FN_CLAUSES_B,
            "results_ab": self.output_dir / FN_RESULTS_AB,
            "results_ba": self.output_dir / FN_RESULTS_BA,
            "scores": self.output_dir / FN_SCORES,
        }

    @staticmethod
    def _parsed_summary(doc: ParsedDocument) -> Dict[str, Any]:
        return {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "total_pages": doc.total_pages,
            "sections_count": len(doc.sections),
        }

    def step_parse(self, pdf_a: str, pdf_b: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse PDF(s) and write ``parsed_policy_*.json``.

        If ``pdf_b`` is None, Policy B is a stub (no second PDF read) for
        single-file testing: segment/embed run only on A; compare/score/export need a full session.
        """
        pdf_a_p = Path(pdf_a)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_a = self._parser.parse_pdf(str(pdf_a_p))
        if pdf_b:
            doc_b = self._parser.parse_pdf(str(Path(pdf_b)))
        else:
            doc_b = ParsedDocument(
                document_id=str(uuid.uuid4()),
                filename="(single-file mode — no policy B)",
                total_pages=0,
                sections=[],
                metadata={"parser": "none", "single_file_mode": True},
            )
        ap = self._artifact_paths()
        ap["parsed_a"].write_text(doc_a.model_dump_json(indent=2), encoding="utf-8")
        ap["parsed_b"].write_text(doc_b.model_dump_json(indent=2), encoding="utf-8")
        out: Dict[str, Any] = {
            "step": "parse",
            "policy_a": self._parsed_summary(doc_a),
            "policy_b": self._parsed_summary(doc_b),
            "artifacts": {
                "parsed_policy_a": str(ap["parsed_a"]),
                "parsed_policy_b": str(ap["parsed_b"]),
            },
        }
        if pdf_b is None:
            out["single_file_mode"] = True
        return out

    def _load_clauses_json(self, filename: str) -> List[PolicyClause]:
        path = self.output_dir / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        return [PolicyClause.model_validate(x) for x in data]

    def step_segment(self) -> Dict[str, Any]:
        """Segment parsed documents → ``clauses_policy_*.json``."""
        ap = self._artifact_paths()
        if not ap["parsed_a"].is_file() or not ap["parsed_b"].is_file():
            raise FileNotFoundError("Missing parsed JSON; run step_parse first.")
        doc_a = ParsedDocument.model_validate_json(
            ap["parsed_a"].read_text(encoding="utf-8")
        )
        doc_b = ParsedDocument.model_validate_json(
            ap["parsed_b"].read_text(encoding="utf-8")
        )
        seg_a = ClauseSegmenter(
            policy_label="A",
            policy_id="policy_a",
            api_key=self.api_key,
            model=self.mistral_model,
            use_llm_fallback=self.use_llm_seg,
            config=self._config,
        )
        seg_b = ClauseSegmenter(
            policy_label="B",
            policy_id="policy_b",
            api_key=self.api_key,
            model=self.mistral_model,
            use_llm_fallback=self.use_llm_seg,
            config=self._config,
        )
        clauses_a = self._enrich_categories(seg_a.segment(doc_a))
        clauses_b = self._enrich_categories(seg_b.segment(doc_b))
        ap["clauses_a"].write_text(
            json.dumps(
                [c.model_dump(mode="json") for c in clauses_a],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        ap["clauses_b"].write_text(
            json.dumps(
                [c.model_dump(mode="json") for c in clauses_b],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "step": "segment",
            "clause_count_a": len(clauses_a),
            "clause_count_b": len(clauses_b),
            "artifacts": {
                "clauses_policy_a": str(ap["clauses_a"]),
                "clauses_policy_b": str(ap["clauses_b"]),
            },
        }

    def step_embed(self) -> Dict[str, Any]:
        """Embed clauses → Chroma under ``vectordb/policy_a|b`` + ``segment_manifest.json``."""
        ap = self._artifact_paths()
        if not ap["clauses_a"].is_file() or not ap["clauses_b"].is_file():
            raise FileNotFoundError("Missing clause JSON; run step_segment first.")
        clauses_a = self._load_clauses_json(FN_CLAUSES_A)
        clauses_b = self._load_clauses_json(FN_CLAUSES_B)
        doc_a = ParsedDocument.model_validate_json(
            ap["parsed_a"].read_text(encoding="utf-8")
        )
        doc_b = ParsedDocument.model_validate_json(
            ap["parsed_b"].read_text(encoding="utf-8")
        )
        embedder_a = self._build_embedder("policy_a", clauses_a)
        embedder_b = self._build_embedder("policy_b", clauses_b)
        self._save_segment_manifest(
            clauses_a, clauses_b, doc_a.filename, doc_b.filename
        )
        ca = sum(1 + len(c.sub_clauses) for c in clauses_a)
        cb = sum(1 + len(c.sub_clauses) for c in clauses_b)
        return {
            "step": "embed",
            "indexed_chunks_a": ca,
            "indexed_chunks_b": cb,
            "vectordb_policy_a": str(self.output_dir / "vectordb" / "policy_a"),
            "vectordb_policy_b": str(self.output_dir / "vectordb" / "policy_b"),
            "segment_manifest": str(self.output_dir / "segment_manifest.json"),
        }

    def _open_embedder(self, label: str) -> EmbeddingPipeline:
        """Re-open an existing Chroma collection (no re-embedding)."""
        return EmbeddingPipeline(
            model_name=self.embedding_model,
            db_path=str(self.output_dir / "vectordb" / label),
            collection_name=label,
        )

    def step_compare(self) -> Dict[str, Any]:
        """Run A↔B matcher + LLM + gap detection; write result JSON arrays."""
        if self._is_stub_policy_b():
            raise FileNotFoundError(
                "Single-file session: comparison needs a full session with policy A and B PDFs."
            )
        ap = self._artifact_paths()

        def _vdb_ok(sub: str) -> bool:
            p = self.output_dir / "vectordb" / sub
            return p.is_dir() and any(p.iterdir())

        if not ap["clauses_a"].is_file() or not ap["clauses_b"].is_file():
            raise FileNotFoundError("Missing clause JSON; run step_segment first.")
        if not _vdb_ok("policy_a") or not _vdb_ok("policy_b"):
            raise FileNotFoundError("Vector DB missing; run step_embed first.")

        clauses_a = self._load_clauses_json(FN_CLAUSES_A)
        clauses_b = self._load_clauses_json(FN_CLAUSES_B)
        embedder_a = self._open_embedder("policy_a")
        embedder_b = self._open_embedder("policy_b")
        reasoner = ComplianceReasoner(
            api_key=self.api_key,
            model=self.mistral_model,
            config=self._config,
        )
        detector = GapDetector(
            missing_threshold=self.missing_similarity_threshold,
            gap_threshold=self.gap_similarity_threshold,
        )
        print("\n[step_compare] A→B …")
        results_a_to_b = self._run_comparison(
            clauses_a, embedder_b, reasoner, detector, "A→B"
        )
        print("\n[step_compare] B→A …")
        results_b_to_a = self._run_comparison(
            clauses_b, embedder_a, reasoner, detector, "B→A"
        )
        ap["results_ab"].write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in results_a_to_b],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        ap["results_ba"].write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in results_b_to_a],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "step": "compare",
            "results_a_to_b_count": len(results_a_to_b),
            "results_b_to_a_count": len(results_b_to_a),
            "artifacts": {
                "comparison_results_a_to_b": str(ap["results_ab"]),
                "comparison_results_b_to_a": str(ap["results_ba"]),
            },
        }

    def step_score(self) -> Dict[str, Any]:
        """Aggregate scores from comparison result JSON files."""
        if self._is_stub_policy_b():
            raise FileNotFoundError(
                "Single-file session: scoring needs a full session with policy A and B PDFs."
            )
        ap = self._artifact_paths()
        if not ap["results_ab"].is_file() or not ap["results_ba"].is_file():
            raise FileNotFoundError("Missing comparison results; run step_compare first.")
        results_a_to_b = [
            ComplianceResult.model_validate(x)
            for x in json.loads(ap["results_ab"].read_text(encoding="utf-8"))
        ]
        results_b_to_a = [
            ComplianceResult.model_validate(x)
            for x in json.loads(ap["results_ba"].read_text(encoding="utf-8"))
        ]
        score_a_to_b = self._scorer.score(results_a_to_b, direction="A→B")
        score_b_to_a = self._scorer.score(results_b_to_a, direction="B→A")
        ComplianceScorer.print_summary(score_a_to_b)
        ComplianceScorer.print_summary(score_b_to_a)
        scores_payload = {
            "score_a_to_b": score_a_to_b.model_dump(mode="json"),
            "score_b_to_a": score_b_to_a.model_dump(mode="json"),
        }
        ap["scores"].write_text(
            json.dumps(scores_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "step": "score",
            "scores": scores_payload,
            "artifact": str(ap["scores"]),
        }

    def step_export_excel(self, report_filename: Optional[str] = None) -> str:
        """Build Excel + full comparison JSON from scored artifacts."""
        if self._is_stub_policy_b():
            raise FileNotFoundError(
                "Single-file session: export needs a full session with policy A and B PDFs."
            )
        ap = self._artifact_paths()
        if not ap["scores"].is_file():
            raise FileNotFoundError("Missing scores; run step_score first.")
        doc_a = ParsedDocument.model_validate_json(
            ap["parsed_a"].read_text(encoding="utf-8")
        )
        doc_b = ParsedDocument.model_validate_json(
            ap["parsed_b"].read_text(encoding="utf-8")
        )
        results_a_to_b = [
            ComplianceResult.model_validate(x)
            for x in json.loads(ap["results_ab"].read_text(encoding="utf-8"))
        ]
        results_b_to_a = [
            ComplianceResult.model_validate(x)
            for x in json.loads(ap["results_ba"].read_text(encoding="utf-8"))
        ]
        scores = json.loads(ap["scores"].read_text(encoding="utf-8"))
        score_a_to_b = ComplianceScore.model_validate(scores["score_a_to_b"])
        score_b_to_a = ComplianceScore.model_validate(scores["score_b_to_a"])

        report = PolicyComparisonReport(
            policy_a_name=doc_a.filename,
            policy_b_name=doc_b.filename,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results_a_to_b=results_a_to_b,
            results_b_to_a=results_b_to_a,
            score_a_to_b=score_a_to_b,
            score_b_to_a=score_b_to_a,
        )
        report_path = self._reporter.generate(report, filename=report_filename)
        self._save_json(report, Path(doc_a.filename).stem, Path(doc_b.filename).stem)
        return str(report_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_embedder(self, label: str, clauses: List[PolicyClause]) -> EmbeddingPipeline:
        db_path = str(self.output_dir / "vectordb" / label)
        embedder = EmbeddingPipeline(
            model_name=self.embedding_model,
            db_path=db_path,
            collection_name=label,
        )

        chunks: List[DocumentChunk] = []
        indexed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for c in clauses:
            sub_texts = [s.text for s in c.sub_clauses if s.text]
            full_text = c.text
            if sub_texts:
                full_text = c.text.rstrip() + "\n" + "\n".join(f"- {t}" for t in sub_texts)

            parent_text = (full_text or c.title or "").strip()
            parent_meta = ClauseVectorMetadata.from_parent_clause(c).to_flat_metadata()
            parent_meta.update(
                {
                    "session_id": self.session_id,
                    "indexed_at": indexed_at,
                    "source_policy": c.policy_label or "",
                    "source_pdf_name": c.filename or "",
                    "source_page_start": int(c.page or 0),
                    "source_page_end": int(c.page or 0),
                    "text_sha256": hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
                    if parent_text
                    else "",
                }
            )
            chunks.append(DocumentChunk(
                chunk_id=c.clause_id,
                document_id=c.policy_id,
                text=parent_text,
                chunk_type="clause",
                parent_context=[c.context],
                metadata=parent_meta,
                pages=[c.page],
            ))

            for sub in c.sub_clauses:
                if not sub.text:
                    continue
                sub_text = (sub.text or "").strip()
                sub_meta = ClauseVectorMetadata.from_sub_clause(c, sub).to_flat_metadata()
                sub_meta.update(
                    {
                        "session_id": self.session_id,
                        "indexed_at": indexed_at,
                        "source_policy": (sub.policy_label or c.policy_label or ""),
                        "source_pdf_name": (sub.filename or c.filename or ""),
                        "source_page_start": int((sub.page or c.page or 0)),
                        "source_page_end": int((sub.page or c.page or 0)),
                        "text_sha256": hashlib.sha256(sub_text.encode("utf-8")).hexdigest()
                        if sub_text
                        else "",
                    }
                )
                chunks.append(DocumentChunk(
                    chunk_id=sub.clause_id,
                    document_id=c.policy_id,
                    text=sub_text,
                    chunk_type="sub_clause",
                    parent_context=[c.context, c.title],
                    metadata=sub_meta,
                    pages=[sub.page],
                ))

        print(f"  [{label}] Embedding {len(chunks)} chunks "
              f"({len(clauses)} clauses + {len(chunks)-len(clauses)} sub-clauses)")
        embedder.process_chunks(chunks)
        return embedder

    def _run_comparison(self, source_clauses, target_embedder, reasoner, detector, direction):
        matcher = ClauseMatcher(target_embedder)
        results = []
        total = len(source_clauses)
        for i, clause in enumerate(source_clauses, 1):
            print(f"  [{i}/{total}] {clause.clause_id}: {clause.title[:60]}")
            candidates = matcher.find_matches(clause, top_k=self.top_k)
            llm_out    = reasoner.compare(clause, candidates)
            result     = detector.detect(clause, candidates, llm_out, direction)
            results.append(result)
            time.sleep(0.5)
        return results

    def _enrich_categories(self, clauses: List[PolicyClause]) -> List[PolicyClause]:
        if not _POLICY_ANALYZER_AVAILABLE:
            return clauses
        analyzer = PolicyAnalyzer()
        for clause in clauses:
            category, _ = analyzer._categorize_statement(clause.text)
            clause.category = category.value
        return clauses

    def _save_json(self, report: PolicyComparisonReport, stem_a: str, stem_b: str):
        json_path = self.output_dir / f"comparison_{stem_a}_vs_{stem_b}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"  JSON report  : {json_path}")

    def _save_segment_manifest(self, clauses_a, clauses_b, filename_a, filename_b):
        def _clause_to_dict(c, is_sub=False, parent_id=""):
            return {
                "chunk_id": c.clause_id, "clause_id": c.clause_id,
                "file_id": c.file_id,
                "sub_clause_id": c.clause_id if is_sub else "",
                "clause_number": c.clause_number, "clause_title": c.title,
                "filename": c.filename, "policy_id": c.policy_id,
                "policy_label": c.policy_label, "is_sub_clause": is_sub,
                "parent_clause_id": parent_id, "sub_clause_count": len(c.sub_clauses),
                "section_title": c.context, "chunk_type": "sub_clause" if is_sub else "clause",
                "category": c.category, "is_requirement": c.is_requirement,
                "keywords": c.keywords, "page": c.page,
                "text_preview": c.text[:300] if c.text else "",
                "text_length": len(c.text),
            }

        def _flatten(clauses):
            rows = []
            for c in clauses:
                rows.append(_clause_to_dict(c))
                for sub in c.sub_clauses:
                    if sub.text:
                        rows.append(_clause_to_dict(sub, is_sub=True, parent_id=c.clause_id))
            return rows

        flat_a = _flatten(clauses_a)
        flat_b = _flatten(clauses_b)
        manifest = {
            "_description": "Clause manifest — chunk_id == clause_id == ChromaDB document ID.",
            "policy_a": {"filename": filename_a, "policy_id": "policy_a",
                         "total_clauses": len(clauses_a), "chunks": flat_a},
            "policy_b": {"filename": filename_b, "policy_id": "policy_b",
                         "total_clauses": len(clauses_b), "chunks": flat_b},
        }
        manifest_path = self.output_dir / "segment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  Segment manifest saved: {manifest_path}")
