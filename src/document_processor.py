"""
Document Processor - PDF parsing and intelligent chunking.

Merges former parser.py and chunker.py into a single module.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import uuid
import re

from .schemas import DocumentElement, DocumentSection, ParsedDocument, DocumentChunk


# ============================================================
# PDF Parser
# ============================================================

class SimplePDFParser:
    """Parse PDFs using PyMuPDF - fast and reliable."""
    
    def __init__(self):
        """Initialize parser."""
        self.heading_patterns = [
            r'^#+\s+',                      # Markdown-style headings
            r'^\d+\.\s+[A-Z]',             # Numbered sections (e.g., "5. Policy")
            r'^\d+\.\d+(\.\d+)*\s+[A-Z]',  # Sub-sections (e.g., "5.1 Governance", "5.2.1 Access")
            r'^[A-Z][A-Z\s]{3,}$',          # ALL CAPS headings
        ]

    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file into structured format.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument with sections and elements
        """
        pdf_path = Path(pdf_path)
        document_id = str(uuid.uuid4())
        
        print(f"\nParsing PDF: {pdf_path.name}")
        print("-" * 60)
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        print(f"Pages: {total_pages}")
        
        # Extract text from all pages
        all_sections = []
        current_section = None
        section_counter = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Get text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        # Extract text from spans
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
                        
                        # Determine if this is a heading
                        is_heading = (
                            is_bold and font_size > 13 or
                            self._looks_like_heading(text)
                        )
                        
                        if is_heading:
                            # Start new section
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
                            # Add to current section
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
        
        # Add last section
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
        """Check if text looks like a heading."""
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        return False


# ============================================================
# Document Chunker
# ============================================================

class DocumentChunker:
    """Split documents into chunks with context."""
    
    def __init__(self, chunk_by: str = "section", max_chars: int = 3000, min_chars: int = 500):
        """
        Initialize chunker with intelligent splitting.
        
        Args:
            chunk_by: "section" or "paragraph"
            max_chars: Maximum characters per chunk (2000-5000 recommended)
            min_chars: Minimum characters to consider splitting
        """
        self.chunk_by = chunk_by
        self.max_chars = max_chars
        self.min_chars = min_chars
    
    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Chunk document with intelligent semantic boundaries.
        
        Rules:
        - Keep small sections together
        - Split large sections at paragraph boundaries
        - Never split mid-paragraph or mid-sentence
        - No content mixing between sections
        
        Args:
            doc: Parsed document
            
        Returns:
            List of enriched chunks
        """
        chunks = []
        
        for section in doc.sections:
            if self.chunk_by == "section":
                # Process section with intelligent splitting
                section_chunks = self._chunk_section_intelligently(section, doc)
                chunks.extend(section_chunks)
            
            elif self.chunk_by == "paragraph":
                # Each element is a chunk (but respect size limits)
                for elem in section.elements:
                    if len(elem.text.strip()) < 20:  # Skip very short
                        continue
                    
                    # If paragraph is too large, split at sentence boundaries
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
                            metadata={
                                "section_title": section.title,
                                "document": doc.filename
                            },
                            pages=[elem.page]
                        )
                        chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from {len(doc.sections)} sections")
        return chunks
    
    def _chunk_section_intelligently(self, section, doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Intelligently chunk a section respecting natural boundaries.
        
        Strategy:
        1. If section fits in max_chars -> single chunk
        2. If section is large -> split at paragraph boundaries
        3. If paragraph is large -> split at sentence boundaries
        """
        # Collect all paragraphs in section
        paragraphs = []
        for elem in section.elements:
            if elem.text.strip():
                paragraphs.append({
                    'text': elem.text,
                    'page': elem.page
                })
        
        if not paragraphs:
            return []
        
        # Calculate total section size
        total_text = "\n\n".join(p['text'] for p in paragraphs)
        total_chars = len(total_text)
        
        # If section fits in one chunk, keep it together
        if total_chars <= self.max_chars:
            pages = sorted(set(p['page'] for p in paragraphs))
            chunk = DocumentChunk(
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
            )
            return [chunk]
        
        # Section is too large - split at paragraph boundaries
        return self._split_at_paragraph_boundaries(paragraphs, section, doc)
    
    def _split_at_paragraph_boundaries(self, paragraphs: List[dict], section, doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Split paragraphs into chunks at natural boundaries.
        Never splits mid-paragraph.
        """
        chunks = []
        current_chunk_text = []
        current_chunk_pages = set()
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para['text'])
            
            # If single paragraph exceeds max_chars, split it at sentences
            if para_size > self.max_chars:
                # Save current chunk if exists
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        "\n\n".join(current_chunk_text),
                        sorted(current_chunk_pages),
                        section,
                        doc,
                        len(chunks)
                    ))
                    current_chunk_text = []
                    current_chunk_pages = set()
                    current_size = 0
                
                # Split large paragraph at sentences
                sentence_chunks = self._split_at_sentences(para['text'], self.max_chars)
                for sent_chunk in sentence_chunks:
                    chunks.append(self._create_chunk(
                        sent_chunk,
                        [para['page']],
                        section,
                        doc,
                        len(chunks)
                    ))
                continue
            
            # Check if adding this paragraph exceeds limit
            if current_size + para_size > self.max_chars and current_chunk_text:
                # Save current chunk
                chunks.append(self._create_chunk(
                    "\n\n".join(current_chunk_text),
                    sorted(current_chunk_pages),
                    section,
                    doc,
                    len(chunks)
                ))
                current_chunk_text = []
                current_chunk_pages = set()
                current_size = 0
            
            # Add paragraph to current chunk
            current_chunk_text.append(para['text'])
            current_chunk_pages.add(para['page'])
            current_size += para_size + 2  # +2 for \n\n
        
        # Save final chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(
                "\n\n".join(current_chunk_text),
                sorted(current_chunk_pages),
                section,
                doc,
                len(chunks)
            ))
        
        # Update total_chunks metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _split_at_sentences(self, text: str, max_size: int) -> List[str]:
        """
        Split text at sentence boundaries.
        Never splits mid-sentence.
        """
        # Split into sentences (basic regex)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If single sentence exceeds max_size, keep it anyway (don't split mid-sentence)
            if sentence_size > max_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(sentence)
                continue
            
            # Check if adding sentence exceeds limit
            if current_size + sentence_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _create_chunk(self, text: str, pages: List[int], section, doc: ParsedDocument, chunk_index: int) -> DocumentChunk:
        """Helper to create a chunk with metadata."""
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
