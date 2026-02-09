"""
Knowledge graph for document relationships.
"""
import networkx as nx
from typing import List, Dict, Any
import json
from pathlib import Path

from .schemas import ParsedDocument, DocumentChunk


class KnowledgeGraph:
    """Manage document structure and relationships."""
    
    def __init__(self):
        """Initialize graph."""
        self.graph = nx.DiGraph()
    
    def add_document(self, doc: ParsedDocument):
        """Add document to graph."""
        # Add document node
        self.graph.add_node(
            doc.document_id,
            type="document",
            filename=doc.filename,
            total_pages=doc.total_pages
        )
        
        # Add sections
        for section in doc.sections:
            section_id = f"{doc.document_id}_section_{section.title[:30]}"
            self.graph.add_node(
                section_id,
                type="section",
                title=section.title,
                pages=f"{section.page_start}-{section.page_end}"
            )
            self.graph.add_edge(doc.document_id, section_id, relation="contains")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to graph."""
        for chunk in chunks:
            self.graph.add_node(
                chunk.chunk_id,
                type="chunk",
                preview=chunk.text[:100] + "...",
                pages=str(chunk.pages)
            )
            self.graph.add_edge(chunk.document_id, chunk.chunk_id, relation="has_chunk")
    
    def save(self, output_path: str):
        """Save graph to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }
