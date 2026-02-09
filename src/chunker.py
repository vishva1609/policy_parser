"""
Document chunking with parent context preservation.
"""
from typing import List
import uuid

from .schemas import ParsedDocument, DocumentChunk


class DocumentChunker:
    """Split documents into chunks with context."""
    
    def __init__(self, chunk_by: str = "section", max_words: int = 500):
        """
        Initialize chunker.
        
        Args:
            chunk_by: "section" or "paragraph"
            max_words: Maximum words per chunk
        """
        self.chunk_by = chunk_by
        self.max_words = max_words
    
    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Chunk document into pieces with parent context.
        
        Args:
            doc: Parsed document
            
        Returns:
            List of enriched chunks
        """
        chunks = []
        
        for section in doc.sections:
            if self.chunk_by == "section":
                # Combine all elements in section
                section_text = "\n\n".join(
                    elem.text for elem in section.elements
                )
                
                if section_text.strip():
                    # Split if too long
                    sub_chunks = self._split_text(section_text)
                    
                    for idx, text in enumerate(sub_chunks):
                        chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            text=text,
                            chunk_type="section",
                            parent_context=[section.title],
                            metadata={
                                "section_title": section.title,
                                "chunk_index": idx,
                                "total_chunks": len(sub_chunks),
                                "document": doc.filename
                            },
                            pages=list(range(section.page_start, section.page_end + 1))
                        )
                        chunks.append(chunk)
            
            elif self.chunk_by == "paragraph":
                # Each element is a chunk
                for elem in section.elements:
                    if len(elem.text.split()) > 10:  # Skip very short paragraphs
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
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into chunks."""
        words = text.split()
        
        if len(words) <= self.max_words:
            return [text]
        
        chunks = []
        for i in range(0, len(words), self.max_words - 50):  # 50 word overlap
            chunk_words = words[i:i + self.max_words]
            chunks.append(" ".join(chunk_words))
        
        return chunks
