"""
JSON document processor — mirrors JsonFileProcessor from archit1012/qa-bot-llm.

Loads JSON into ParsedDocument, then uses the same chunking / vectordb contract
as SimplePDFParser (BaseDocumentProcessor).
"""
import json
import uuid
from pathlib import Path
from typing import Any, List

from .base_document_processor import BaseDocumentProcessor
from .document_processor import DocumentChunker
from ..common.schemas import DocumentElement, DocumentSection, ParsedDocument, DocumentChunk


class SimpleJSONParser(BaseDocumentProcessor):
    """
    Parse JSON files into ParsedDocument for the shared DocumentService pipeline.

    Supports:
    - Exports produced by this project (ParsedDocument-shaped JSON).
    - Arbitrary JSON: flattened to readable text under one section.
    """

    def document_loader(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "sections" in data and "document_id" in data:
            try:
                return ParsedDocument.model_validate(data)
            except Exception:
                pass

        body = self._json_to_text(data)
        section = DocumentSection(
            title="JSON Content",
            level=1,
            elements=[
                DocumentElement(
                    text=body,
                    element_type="paragraph",
                    page=1,
                    metadata={"source": "json"},
                )
            ],
            page_start=1,
            page_end=1,
        )
        return ParsedDocument(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            total_pages=1,
            sections=[section],
            metadata={"parser": "json", "original_type": type(data).__name__},
        )

    def text_splitter(self, documents: ParsedDocument) -> List[DocumentChunk]:
        return DocumentChunker().chunk_document(documents)

    def prepare_vectordb(self, docs: List[DocumentChunk], embedder: Any) -> Any:
        embedder.process_chunks(docs)
        return embedder

    @staticmethod
    def _json_to_text(data: Any, _indent: int = 0) -> str:
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{k}:\n{SimpleJSONParser._json_to_text(v, _indent + 1)}")
                else:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines)
        if isinstance(data, list):
            parts = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    parts.append(f"[{i}]\n{SimpleJSONParser._json_to_text(item, _indent + 1)}")
                else:
                    parts.append(f"[{i}] {item}")
            return "\n".join(parts)
        return str(data)
