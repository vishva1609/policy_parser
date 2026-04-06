"""
Document Processor — PDF parsing and intelligent chunking.

Concrete implementations of BaseDocumentProcessor (mirrors PdfFileProcessor from
archit1012/qa-bot-llm), using the EXACT same method names as the reference ABC:

    SimplePDFParser   — implements document_loader()   (PDF → ParsedDocument)
    DocumentChunker   — implements text_splitter()     (ParsedDocument → List[DocumentChunk])

Both also implement prepare_vectordb(docs, embedder) which takes a pre-built
embedder (injected from the views layer) and returns it after indexing —
mirroring how PdfFileProcessor.prepare_vectordb(docs, embeddings) works.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import uuid
import re

from .base_document_processor import BaseDocumentProcessor
from ..common.schemas import (
    DocumentElement, DocumentSection, ParsedDocument, DocumentChunk,
)


# ============================================================
# PDF Parser  (implements load())
# ============================================================

class SimplePDFParser(BaseDocumentProcessor):
    """
    Parse PDFs using PyMuPDF — fast and reliable.

    Concrete implementation of BaseDocumentProcessor — mirrors PdfFileProcessor
    from archit1012/qa-bot-llm with identical method names:
        document_loader()   — parse PDF → ParsedDocument
        text_splitter()     — ParsedDocument → List[DocumentChunk]
        prepare_vectordb()  — embed chunks into vector store, return the store
    """

    def __init__(self):
        self.heading_patterns = [
            r'^#+\s+',
            r'^\d+\.\s+[A-Z]',
            r'^\d+\.\d+(\.\d+)*\s+[A-Z]',
            r'^[A-Z][A-Z\s]{3,}$',
        ]

    # ── BaseDocumentProcessor contract ─────────────────────────────────────────

    def document_loader(self, file_path: str) -> ParsedDocument:
        """
        Load and parse a PDF file.

        Exact equivalent of PdfFileProcessor.document_loader() in reference repo:
            def document_loader(self, file_path):
                loader = PyPDFLoader(file_path=file_path)
                return loader.load()

        Args:
            file_path: Path to PDF file.

        Returns:
            ParsedDocument with sections and elements.
        """
        return self.parse_pdf(file_path)

    def text_splitter(self, documents: ParsedDocument) -> List[DocumentChunk]:
        """
        Split a parsed document using default chunking settings.

        Exact equivalent of PdfFileProcessor.text_splitter() in reference repo:
            def text_splitter(self, documents):
                splitter = RecursiveCharacterTextSplitter(...)
                return splitter.split_documents(documents)

        Delegates to DocumentChunker for fine-grained control.
        """
        chunker = DocumentChunker()
        return chunker.chunk_document(documents)

    def prepare_vectordb(self, docs: List[DocumentChunk], embedder) -> object:
        """
        Store chunks in vector DB via the injected embedder, return the store.

        Exact equivalent of PdfFileProcessor.prepare_vectordb(docs, embeddings):
            def prepare_vectordb(self, docs, embeddings):
                vector_db = FAISS.from_documents(docs, embeddings)
                return vector_db

        The embedder is injected from the views layer (dependency injection),
        not created here. Returns the populated embedder for later querying.
        """
        embedder.process_chunks(docs)
        return embedder

    # ── Core parsing logic ─────────────────────────────────────────────────────

    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file into structured format.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            ParsedDocument with sections and elements.
        """
        pdf_path = Path(pdf_path)
        document_id = str(uuid.uuid4())

        print(f"\nParsing PDF: {pdf_path.name}")
        print("-" * 60)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        print(f"Pages: {total_pages}")

        all_sections = []
        current_section = None
        section_counter = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        text_parts = []
                        is_bold = False
                        font_size = 12

                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
                            font_size = max(font_size, span.get("size", 12))
                            if "bold" in span.get("font", "").lower():
                                is_bold = True

                        text = " ".join(text_parts).strip()
                        if not text:
                            continue

                        is_heading = (
                            is_bold and font_size > 13 or
                            self._looks_like_heading(text)
                        )

                        if is_heading:
                            if current_section and current_section.elements:
                                all_sections.append(current_section)
                            section_counter += 1
                            current_section = DocumentSection(
                                title=text,
                                level=1,
                                elements=[],
                                page_start=page_num + 1,
                                page_end=page_num + 1
                            )
                        else:
                            if current_section is None:
                                section_counter += 1
                                current_section = DocumentSection(
                                    title="Document Content",
                                    level=1,
                                    elements=[],
                                    page_start=page_num + 1,
                                    page_end=page_num + 1
                                )

                            element = DocumentElement(
                                text=text,
                                element_type="paragraph",
                                page=page_num + 1,
                                metadata={"font_size": font_size}
                            )
                            current_section.elements.append(element)
                            current_section.page_end = page_num + 1

        if current_section and current_section.elements:
            all_sections.append(current_section)

        doc.close()

        print(f"Extracted {len(all_sections)} sections")
        print(f"Parsing complete\n")

        return ParsedDocument(
            document_id=document_id,
            filename=pdf_path.name,
            total_pages=total_pages,
            sections=all_sections,
            metadata={"parser": "pymupdf"}
        )

    def _looks_like_heading(self, text: str) -> bool:
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        return False


# ============================================================
# Document Chunker  (implements split())
# ============================================================

class DocumentChunker(BaseDocumentProcessor):
    """
    Split documents into semantic chunks.

    Concrete implementation of BaseDocumentProcessor focused on text_splitter().
    Mirrors the RecursiveCharacterTextSplitter usage in the reference repo,
    but uses our custom section-aware splitting algorithm.
    """

    def __init__(
        self,
        chunk_by: str = "section",
        max_chars: int = 3000,
        min_chars: int = 500,
    ):
        self.chunk_by = chunk_by
        self.max_chars = max_chars
        self.min_chars = min_chars

    # ── BaseDocumentProcessor contract ─────────────────────────────────────────

    def document_loader(self, file_path: str):
        """Not applicable to chunker — use SimplePDFParser.document_loader() instead."""
        raise NotImplementedError(
            "DocumentChunker does not load files. Use SimplePDFParser.document_loader()."
        )

    def text_splitter(self, documents: ParsedDocument) -> List[DocumentChunk]:
        """
        Split a parsed document into semantic chunks.

        This is the primary method — implements BaseDocumentProcessor.text_splitter().
        Mirrors RecursiveCharacterTextSplitter.split_documents() from reference repo.
        """
        return self.chunk_document(documents)

    def prepare_vectordb(self, docs: List[DocumentChunk], embedder) -> object:
        """Embed and index chunks via the injected embedder; return it."""
        embedder.process_chunks(docs)
        return embedder

    # ── Core chunking logic ────────────────────────────────────────────────────

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        chunks = []

        for section in doc.sections:
            if self.chunk_by == "section":
                section_chunks = self._chunk_section_intelligently(section, doc)
                chunks.extend(section_chunks)

            elif self.chunk_by == "paragraph":
                for elem in section.elements:
                    if len(elem.text.strip()) < 20:
                        continue
                    if len(elem.text) > self.max_chars:
                        para_chunks = self._split_at_sentences(elem.text, self.max_chars)
                        for idx, text in enumerate(para_chunks):
                            chunk = DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                document_id=doc.document_id,
                                text=text,
                                chunk_type="paragraph",
                                parent_context=[section.title],
                                metadata={
                                    "section_title": section.title,
                                    "document": doc.filename,
                                    "chunk_index": idx,
                                    "split_paragraph": True
                                },
                                pages=[elem.page]
                            )
                            chunks.append(chunk)
                    else:
                        chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            text=elem.text,
                            chunk_type="paragraph",
                            parent_context=[section.title],
                            metadata={"section_title": section.title, "document": doc.filename},
                            pages=[elem.page]
                        )
                        chunks.append(chunk)

        print(f"Created {len(chunks)} chunks from {len(doc.sections)} sections")
        return chunks

    def _chunk_section_intelligently(
        self, section, doc: ParsedDocument
    ) -> List[DocumentChunk]:
        paragraphs = []
        for elem in section.elements:
            if elem.text.strip():
                paragraphs.append({'text': elem.text, 'page': elem.page})

        if not paragraphs:
            return []

        total_text = "\n\n".join(p['text'] for p in paragraphs)
        total_chars = len(total_text)

        if total_chars <= self.max_chars:
            pages = sorted(set(p['page'] for p in paragraphs))
            return [DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc.document_id,
                text=total_text,
                chunk_type="section",
                parent_context=[section.title],
                metadata={
                    "section_title": section.title,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "document": doc.filename,
                    "char_count": total_chars
                },
                pages=pages
            )]

        return self._split_at_paragraph_boundaries(paragraphs, section, doc)

    def _split_at_paragraph_boundaries(
        self, paragraphs: List[dict], section, doc: ParsedDocument
    ) -> List[DocumentChunk]:
        chunks = []
        current_chunk_text = []
        current_chunk_pages = set()
        current_size = 0

        for para in paragraphs:
            para_size = len(para['text'])

            if para_size > self.max_chars:
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        "\n\n".join(current_chunk_text),
                        sorted(current_chunk_pages), section, doc, len(chunks)
                    ))
                    current_chunk_text = []
                    current_chunk_pages = set()
                    current_size = 0
                for sent_chunk in self._split_at_sentences(para['text'], self.max_chars):
                    chunks.append(self._create_chunk(
                        sent_chunk, [para['page']], section, doc, len(chunks)
                    ))
                continue

            if current_size + para_size > self.max_chars and current_chunk_text:
                chunks.append(self._create_chunk(
                    "\n\n".join(current_chunk_text),
                    sorted(current_chunk_pages), section, doc, len(chunks)
                ))
                current_chunk_text = []
                current_chunk_pages = set()
                current_size = 0

            current_chunk_text.append(para['text'])
            current_chunk_pages.add(para['page'])
            current_size += para_size + 2

        if current_chunk_text:
            chunks.append(self._create_chunk(
                "\n\n".join(current_chunk_text),
                sorted(current_chunk_pages), section, doc, len(chunks)
            ))

        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks

    def _split_at_sentences(self, text: str, max_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if sentence_size > max_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(sentence)
                continue

            if current_size + sentence_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]

    def _create_chunk(
        self, text: str, pages: List[int], section, doc: ParsedDocument,
        chunk_index: int
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc.document_id,
            text=text,
            chunk_type="section",
            parent_context=[section.title],
            metadata={
                "section_title": section.title,
                "chunk_index": chunk_index,
                "document": doc.filename,
                "char_count": len(text),
                "section_level": section.level
            },
            pages=pages
        )
