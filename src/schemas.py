"""
Simple document schemas using Pydantic.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentElement(BaseModel):
    """A document element (paragraph, heading, etc.)."""
    text: str
    element_type: str  # "heading", "paragraph", "list", "table"
    page: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A document section with elements."""
    title: str
    level: int
    elements: List[DocumentElement]
    page_start: int
    page_end: int


class ParsedDocument(BaseModel):
    """Complete parsed document."""
    document_id: str
    filename: str
    total_pages: int
    sections: List[DocumentSection]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    """Enriched document chunk."""
    chunk_id: str
    document_id: str
    text: str
    chunk_type: str
    parent_context: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[int] = Field(default_factory=list)


class EmbeddedChunk(BaseModel):
    """Chunk with embedding vector."""
    chunk_id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
