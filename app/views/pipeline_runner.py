"""
PipelineRunner — Views layer entry point.

This directly mirrors views/qa_apis.py from archit1012/qa-bot-llm:

    # qa_apis.py (reference):
    embeddings = OpenAIEmbeddings()                    ← created ONCE at module level
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", ...) ← created ONCE at module level

    def upload_files():
        response = process_request(request, embeddings, chat)  ← INJECTED into service

PipelineRunner does the exact same thing:
    - Creates MistralConfig ONCE in __init__  (= ChatOpenAI)
    - Creates EmbeddingPipeline ONCE in __init__ (= OpenAIEmbeddings)
    - INJECTS both into every service call

This is Dependency Injection — the views layer owns the shared resources,
and passes them down into the service, which passes them into the models.
No model creates its own LLM or embedder independently.

Usage::

    runner = PipelineRunner(api_key="...", model="mistral-small-latest")

    # Single document pipeline
    results = runner.run_document_pipeline("samples/policy.pdf")

    # Compliance comparison (injected config + embedder used automatically)
    report = runner.run_comparison_pipeline("samples/A.pdf", "samples/B.pdf")

    # Segment manifest only (no LLM)
    manifest = runner.run_manifest_only("samples/A.pdf", "samples/B.pdf")
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ..common.config_loader import load_pipeline_config
from ..common.llm_client import MistralConfig, load_mistral_config_optional
from ..models.embedder import EmbeddingPipeline
from ..service.document_service import DocumentService
from ..service.comparison_service import PolicyComparisonPipeline


class PipelineRunner:
    """
    CLI views layer — creates shared resources once and injects them.

    Mirrors views/qa_apis.py from archit1012/qa-bot-llm:
        - qa_apis.py creates 'embeddings' + 'chat' once at module level
        - PipelineRunner creates 'config' + 'embedder' once in __init__
        - Both inject these shared objects into all downstream calls

    KEY OOP PRINCIPLE: Dependency Injection
        Resources are created in the views layer and flow DOWNWARD:
            views (PipelineRunner)
                → service (DocumentService / PolicyComparisonPipeline)
                    → models (ClauseSegmenter / ComplianceReasoner)

    Attributes:
        config:   MistralConfig instance (shared LLM — like ChatOpenAI in reference)
        embedder: EmbeddingPipeline instance (shared embedder — like OpenAIEmbeddings)
    """

    def __init__(
        self,
        output_dir: str = "./output",
        embedding_model: str = "all-MiniLM-L6-v2",
        mistral_model: str = "mistral-small-latest",
        api_key: Optional[str] = None,
    ):
        """
        Initialise shared resources — created ONCE, injected everywhere.

        Mirrors qa_apis.py:
            embeddings = OpenAIEmbeddings()         ← our self.embedder
            chat = ChatOpenAI(model_name=..., ...)  ← our self.config

        Args:
            output_dir:      Base output directory for all pipeline outputs.
            embedding_model: sentence-transformers model name for self.embedder.
            mistral_model:   Mistral model name for self.config.
            api_key:         Mistral API key (falls back to MISTRAL_API_KEY env var).
        """
        self.output_dir     = output_dir
        self.mistral_model  = mistral_model
        self._api_key       = api_key or os.getenv("MISTRAL_API_KEY")
        self._embedding_model = embedding_model

        # ── Create shared resources ONCE (mirrors qa_apis.py global vars) ──────

        # self.embedder ← equivalent to: embeddings = OpenAIEmbeddings()
        self.embedder = EmbeddingPipeline(
            model_name=embedding_model,
            db_path=str(Path(output_dir) / "vectordb"),
        )

        # self.config ← equivalent to: chat = ChatOpenAI(...); hyperparameters from config.yaml
        self.config = load_mistral_config_optional(api_key=self._api_key)

        self._defaults_from_yaml = False
        self._chunking_defaults: Dict[str, Any] = {}
        self._comparison_defaults: Dict[str, Any] = {}
        self._segmentation_defaults: Dict[str, Any] = {}
        self._output_defaults: Dict[str, Any] = {}

    @classmethod
    def from_config_file(
        cls,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "PipelineRunner":
        """
        Build a runner from repo-root ``config.yaml`` (same idea as the reference repo).

        YAML-driven defaults apply to :meth:`run_document_pipeline` and
        :meth:`run_comparison_pipeline`. :meth:`run_manifest_only` always
        disables LLM segmentation regardless of YAML.
        """
        raw = load_pipeline_config(config_path)
        llm = raw.get("llm") or {}
        emb = raw.get("embeddings") or {}
        out = raw.get("output") or {}

        inst = cls(
            output_dir=out.get("dir", "./output"),
            embedding_model=emb.get("model", "all-MiniLM-L6-v2"),
            mistral_model=llm.get("model", "mistral-small-latest"),
            api_key=api_key,
        )
        inst._defaults_from_yaml = True
        inst._chunking_defaults = raw.get("chunking") or {}
        inst._comparison_defaults = raw.get("comparison") or {}
        inst._segmentation_defaults = raw.get("segmentation") or {}
        inst._output_defaults = out
        return inst

    def _apply_yaml_document_defaults(
        self,
        output_format: str,
        overwrite_files: bool,
        max_chunk_chars: int,
        min_chunk_chars: int,
        chunk_by: str,
    ):
        if not getattr(self, "_defaults_from_yaml", False):
            return output_format, overwrite_files, max_chunk_chars, min_chunk_chars, chunk_by
        chunk = getattr(self, "_chunking_defaults", None) or {}
        outd = getattr(self, "_output_defaults", None) or {}
        if chunk:
            max_chunk_chars = int(chunk.get("max_chars", max_chunk_chars))
            min_chunk_chars = int(chunk.get("min_chars", min_chunk_chars))
            chunk_by = str(chunk.get("strategy", chunk_by))
        if outd:
            output_format = str(outd.get("chunk_format", output_format))
            overwrite_files = bool(outd.get("overwrite_chunks", overwrite_files))
        return output_format, overwrite_files, max_chunk_chars, min_chunk_chars, chunk_by

    def _apply_yaml_comparison_defaults(
        self,
        top_k_candidates: int,
        use_llm_segmentation: bool,
        missing_threshold: Optional[float],
        gap_threshold: Optional[float],
    ):
        if not getattr(self, "_defaults_from_yaml", False):
            return top_k_candidates, use_llm_segmentation, missing_threshold, gap_threshold
        comp = getattr(self, "_comparison_defaults", None) or {}
        seg = getattr(self, "_segmentation_defaults", None) or {}
        if comp:
            if missing_threshold is None and "missing_threshold" in comp:
                missing_threshold = float(comp["missing_threshold"])
            if gap_threshold is None and "gap_threshold" in comp:
                gap_threshold = float(comp["gap_threshold"])
            if "top_k_candidates" in comp:
                top_k_candidates = int(comp["top_k_candidates"])
        if seg and "use_llm_fallback" in seg:
            use_llm_segmentation = bool(seg["use_llm_fallback"])
        return top_k_candidates, use_llm_segmentation, missing_threshold, gap_threshold

    # ── Single document pipeline ───────────────────────────────────────────────

    def run_document_pipeline(
        self,
        file_path: str,
        output_format: str = "txt",
        overwrite_files: bool = False,
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 500,
        chunk_by: str = "section",
    ) -> Dict[str, Any]:
        """
        Process a single document: parse → chunk → embed → export.

        Injects self.embedder into DocumentService (dependency injection).
        Mirrors process_request(request, embeddings, chat) from reference.

        Args:
            file_path:       Path to source document (PDF).
            output_format:   "txt", "json", or "md" for chunk files.
            overwrite_files: If False, skip existing chunk files.
            max_chunk_chars: Max characters per chunk.
            min_chunk_chars: Min characters per chunk.
            chunk_by:        Chunking strategy: "section" or "paragraph".

        Returns:
            Results dict from DocumentService.process()
        """
        self._print_header("DOCUMENT PROCESSING PIPELINE", file_path)

        output_format, overwrite_files, max_chunk_chars, min_chunk_chars, chunk_by = (
            self._apply_yaml_document_defaults(
                output_format, overwrite_files, max_chunk_chars, min_chunk_chars, chunk_by
            )
        )

        # Inject self.embedder into the service — DI, not internal creation
        service = DocumentService(
            output_dir=self.output_dir,
            embedding_model=self._embedding_model,
            chunk_by=chunk_by,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
            embedder=self.embedder,       # ← INJECTED (like embeddings param in reference)
        )

        results = service.process(file_path, output_format=output_format,
                                  overwrite_files=overwrite_files)
        self._print_document_stats(results)
        return results

    # ── Comparison pipeline ────────────────────────────────────────────────────

    def run_comparison_pipeline(
        self,
        pdf_a: str,
        pdf_b: str,
        report_filename: Optional[str] = None,
        top_k_candidates: int = 5,
        use_llm_segmentation: bool = True,
        missing_threshold: Optional[float] = None,
        gap_threshold: Optional[float] = None,
    ) -> str:
        """
        Run bidirectional compliance comparison between two policy documents.

        Injects self.config (MistralConfig) and self.embedder into the pipeline.
        This is the exact pattern from qa_apis.py:
            response = process_request(request, embeddings, chat)
                                                 ↑           ↑
                                           injected      injected

        Args:
            pdf_a:                Path to Policy A (source of truth).
            pdf_b:                Path to Policy B (to validate).
            report_filename:      Optional custom Excel filename.
            top_k_candidates:     Number of candidates to retrieve per clause.
            use_llm_segmentation: Enable LLM fallback for complex sections.
            missing_threshold:    Similarity below this → Missing status.
            gap_threshold:        Similarity below this + Non-Compliant → Non-Compliant.

        Returns:
            Absolute path to the generated Excel compliance report.
        """
        self._print_header(
            "POLICY COMPLIANCE COMPARISON",
            f"A={Path(pdf_a).name} vs B={Path(pdf_b).name}",
        )

        top_k_candidates, use_llm_segmentation, missing_threshold, gap_threshold = (
            self._apply_yaml_comparison_defaults(
                top_k_candidates,
                use_llm_segmentation,
                missing_threshold,
                gap_threshold,
            )
        )

        # Inject self.config + self.embedder into the service
        pipeline = PolicyComparisonPipeline(
            output_dir=self.output_dir,
            embedding_model=self._embedding_model,
            mistral_model=self.mistral_model,
            api_key=self._api_key,
            top_k_candidates=top_k_candidates,
            use_llm_segmentation=use_llm_segmentation,
            missing_similarity_threshold=missing_threshold,
            gap_similarity_threshold=gap_threshold,
            config=self.config,      # ← INJECTED (like chat param in reference)
        )

        return pipeline.compare(pdf_a, pdf_b, report_filename=report_filename)

    # ── Manifest only (no LLM cost) ────────────────────────────────────────────

    def run_manifest_only(self, pdf_a: str, pdf_b: str) -> str:
        """
        Parse + segment both PDFs, write segment_manifest.json (no LLM calls).
        """
        self._print_header(
            "SEGMENT MANIFEST (no API calls)",
            f"A={Path(pdf_a).name} vs B={Path(pdf_b).name}",
        )

        pipeline = PolicyComparisonPipeline(
            output_dir=self.output_dir,
            embedding_model=self._embedding_model,
            use_llm_segmentation=False,
        )
        return pipeline.build_manifest(pdf_a, pdf_b)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_header(title: str, subtitle: str = ""):
        print("\n" + "=" * 70)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 70)

    @staticmethod
    def _print_document_stats(results: Dict[str, Any]):
        stats = results.get("stats", {})
        print("\nPROCESSING STATISTICS:")
        print("-" * 70)
        print(f"  Total Pages     : {stats.get('total_pages', 'N/A')}")
        print(f"  Total Sections  : {stats.get('total_sections', 'N/A')}")
        print(f"  Total Chunks    : {stats.get('total_chunks', 'N/A')}")
        print(f"  Avg Chunk Size  : {stats.get('avg_chunk_size', 'N/A')} chars")
        print("-" * 70)
        if results.get("excel_path"):
            print(f"\n  Excel Index   : {results['excel_path']}")
        print("  Processing complete!\n")
