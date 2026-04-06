"""
DocumentService — Factory + orchestrator for single-document processing.

Directly mirrors qa_apis_service.py from archit1012/qa-bot-llm:

    # qa_apis_service.py (reference):
    def process_request(request, embeddings, chat):  ← embeddings INJECTED
        if doc_file_path.endswith('.pdf'):
            llm = PdfFileProcessor()                 ← factory picks class
        elif doc_file_path.endswith('.json'):
            llm = JsonFileProcessor()

        documents = llm.document_loader(file_path)   ← calls ABC method
        docs      = llm.text_splitter(documents)      ← calls ABC method
        db        = llm.prepare_vectordb(docs, embeddings)  ← injects embeddings

    # Our DocumentService.process():
    def process(self, file_path, embedder=None):     ← embedder INJECTED
        processor = self._SUPPORTED_FORMATS[ext]()   ← factory picks class
        documents = processor.document_loader(path)  ← calls ABC method
        docs      = processor.text_splitter(documents) ← calls ABC method
        db        = processor.prepare_vectordb(docs, embedder)  ← injects embedder
"""
from pathlib import Path
from typing import Dict, Any, Optional

import networkx as nx
import json

from ..models.document_processor import SimplePDFParser, DocumentChunker
from ..models.json_document_processor import SimpleJSONParser
from ..models.embedder import EmbeddingPipeline
from ..common.schemas import ParsedDocument, DocumentChunk
from ..common.excel_utils import ExcelGenerator


class DocumentService:
    """
    Orchestrates single-document processing pipeline.

    Mirrors process_request() in qa_apis_service.py:
        1. Factory: picks the right processor class by file extension
        2. Calls document_loader() → text_splitter() → prepare_vectordb()
           using the injected embedder (dependency injection from views layer)

    The embedder is injected from PipelineRunner (views layer), not created here.
    This mirrors how process_request receives 'embeddings' from qa_apis.py.

    Usage (direct)::
        service = DocumentService(output_dir="./output")
        results = service.process("samples/policy.pdf", embedder=my_embedder)

    Usage (via PipelineRunner — preferred)::
        runner = PipelineRunner()
        results = runner.run_document_pipeline("samples/policy.pdf")
    """

    # Factory registry: extension → processor class
    # Add new formats (JSON, DOCX) here without touching any other layer
    _SUPPORTED_FORMATS = {
        ".pdf": SimplePDFParser,
        ".json": SimpleJSONParser,
    }

    def __init__(
        self,
        output_dir: str = "./output",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_by: str = "section",
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 500,
        embedder: Optional[EmbeddingPipeline] = None,
    ):
        """
        Initialise the service.

        Args:
            output_dir:      Base output directory.
            embedding_model: Model name (used only if embedder is None).
            chunk_by:        Chunking strategy.
            max_chunk_chars: Max chars per chunk.
            min_chunk_chars: Min chars per chunk.
            embedder:        Pre-built EmbeddingPipeline (dependency injection).
                             If None, a new one is created internally.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._chunker = DocumentChunker(
            chunk_by=chunk_by,
            max_chars=max_chunk_chars,
            min_chars=min_chunk_chars,
        )

        # Use injected embedder if provided; otherwise create one (backward compat)
        self._embedder = embedder or EmbeddingPipeline(
            model_name=embedding_model,
            db_path=str(self.output_dir / "vectordb"),
        )
        self._excel = ExcelGenerator(str(self.output_dir))

    def process(
        self,
        file_path: str,
        output_format: str = "txt",
        overwrite_files: bool = False,
        embedder: Optional[EmbeddingPipeline] = None,
    ) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.

        Mirrors process_request() from qa_apis_service.py — the key pattern is:
            1. Factory picks the right processor class
            2. Calls document_loader() → text_splitter() → prepare_vectordb()
               using the injected embedder

        Args:
            file_path:       Path to document (PDF, future: JSON).
            output_format:   "txt", "json", or "md" for chunk files.
            overwrite_files: If False, skip existing chunk files.
            embedder:        Optional embedder to use instead of self._embedder.

        Returns:
            Dict with document, chunks, paths, and stats.

        Raises:
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # Factory: pick the right processor (mirrors service layer in reference repo)
        processor_cls = self._SUPPORTED_FORMATS.get(ext)
        if processor_cls is None:
            supported = ", ".join(self._SUPPORTED_FORMATS.keys())
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported: {supported}"
            )

        parser = processor_cls()
        active_embedder = embedder or self._embedder

        # Step 1: document_loader() — exact method name from reference ABC
        print(f"\nStep 1: Parsing {ext.upper()} document…")
        parsed_doc: ParsedDocument = parser.document_loader(str(path))

        # Step 2: text_splitter() — exact method name from reference ABC
        print("\nStep 2: Splitting into chunks…")
        chunks = self._chunker.text_splitter(parsed_doc)

        # Step 3: Excel index
        print("\nStep 3: Generating Excel searchable index…")
        excel_path = self._excel.generate_searchable_index(parsed_doc, chunks)

        # Step 4: Write chunk files
        print(f"\nStep 4: Writing chunk files ({output_format})…")
        chunk_files = self._write_chunks(chunks, parsed_doc.filename, output_format,
                                        overwrite_files)

        # Step 5: prepare_vectordb() — exact method name from reference ABC
        #          The injected embedder is passed in — mirrors:
        #            db = llm.prepare_vectordb(docs, embeddings)  # reference
        print("\nStep 5: Building vector DB (prepare_vectordb)…")
        db = parser.prepare_vectordb(chunks, active_embedder)

        # Step 6: Save parsed JSON
        self._save_outputs(parsed_doc, path.stem)

        avg_size = int(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0

        return {
            "document": parsed_doc,
            "chunks": chunks,
            "vector_db": db,          # explicit: db returned by prepare_vectordb
            "excel_path": excel_path,
            "chunk_files": chunk_files,
            "stats": {
                "total_pages": parsed_doc.total_pages,
                "total_sections": len(parsed_doc.sections),
                "total_chunks": len(chunks),
                "avg_chunk_size": avg_size,
                "output_format": output_format,
            },
        }

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Semantic search across indexed documents."""
        results = self._embedder.search(query, n_results=n_results)
        formatted = []
        if results.get('documents') and results['documents'][0]:
            for i, (doc, dist, meta) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0],
            )):
                formatted.append({
                    "rank": i + 1,
                    "text": doc,
                    "score": f"{(1 - dist):.3f}",
                    "document": meta.get('document', 'N/A'),
                    "section": meta.get('section_title', 'N/A'),
                    "pages": meta.get('pages', []),
                })
        return {"query": query, "results": formatted}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _write_chunks(
        self,
        chunks,
        filename: str,
        fmt: str,
        overwrite: bool,
    ):
        """Write individual chunk files to disk."""
        base = Path(filename).stem
        doc_dir = self.output_dir / "chunks" / base
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_paths = []
        created = skipped = 0
        for i, chunk in enumerate(chunks, 1):
            file_path = doc_dir / f"chunk_{i:03d}.{fmt}"
            if file_path.exists() and not overwrite:
                skipped += 1
                file_paths.append(str(file_path))
                continue
            if fmt == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== CHUNK {i} ===\nID: {chunk.chunk_id}\n"
                            f"Section: {chunk.parent_context[0] if chunk.parent_context else 'N/A'}\n"
                            f"Pages: {chunk.pages}\n\n{'='*50}\n\n{chunk.text}")
            elif fmt == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'chunk_index': i, 'chunk_id': chunk.chunk_id,
                        'text': chunk.text, 'pages': chunk.pages,
                        'parent_context': chunk.parent_context,
                    }, f, indent=2)
            created += 1
            file_paths.append(str(file_path))

        print(f"Created {created} chunks, skipped {skipped} existing in {doc_dir}")
        return file_paths

    def _save_outputs(self, parsed_doc: ParsedDocument, stem: str):
        """Save parsed document JSON."""
        parsed_path = self.output_dir / f"{stem}_parsed.json"
        with open(parsed_path, 'w', encoding='utf-8') as f:
            f.write(parsed_doc.model_dump_json(indent=2))
        print(f"Saved parsed JSON: {parsed_path}")
