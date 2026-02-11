"""
Document chunking with parent context preservation and intelligent boundary detection.
"""
from typing import List
import uuid
import re

from .schemas import ParsedDocument, DocumentChunk


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
        1. If section fits in max_chars → single chunk
        2. If section is large → split at paragraph boundaries
        3. If paragraph is large → split at sentence boundaries
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
