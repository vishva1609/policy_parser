"""
Main pipeline orchestrator.
"""
from pathlib import Path
from typing import Dict, Any
import json

from .parser import SimplePDFParser
from .chunker import DocumentChunker
from .embedder import EmbeddingPipeline
from .knowledge_graph import KnowledgeGraph


class DocumentPipeline:
    """End-to-end document processing pipeline."""
    
    def __init__(
        self,
        chunk_by: str = "section",
        embedding_model: str = "all-MiniLM-L6-v2",
        output_dir: str = "./output"
    ):
        """
        Initialize pipeline.
        
        Args:
            chunk_by: "section" or "paragraph"
            embedding_model: Sentence transformer model name
            output_dir: Output directory
        """
        print("=" * 60)
        print("Document Processing Pipeline")
        print("=" * 60)
        
        self.parser = SimplePDFParser()
        self.chunker = DocumentChunker(chunk_by=chunk_by)
        self.embedder = EmbeddingPipeline(model_name=embedding_model)
        self.graph = KnowledgeGraph()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Configuration:")
        print(f"  - Chunking: {chunk_by}")
        print(f"  - Embedding: {embedding_model}")
        print(f"  - Output: {output_dir}")
        print("=" * 60)
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing results
        """
        pdf_path = Path(pdf_path)
        
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}\n")
        
        # Stage 1: Parse
        print("STAGE 1: Parse PDF")
        print("-" * 60)
        parsed_doc = self.parser.parse_pdf(str(pdf_path))
        
        # Save parsed document
        parsed_path = self.output_dir / f"{pdf_path.stem}_parsed.json"
        with open(parsed_path, 'w') as f:
            json.dump(parsed_doc.model_dump(), f, indent=2, default=str)
        print(f"Saved: {parsed_path}\n")
        
        # Stage 2: Chunk
        print("STAGE 2: Chunk & Enrich")
        print("-" * 60)
        chunks = self.chunker.chunk_document(parsed_doc)
        
        # Save chunks
        chunks_path = self.output_dir / f"{pdf_path.stem}_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2, default=str)
        print(f"Saved: {chunks_path}\n")
        
        # Stage 3: Embed & Index
        print("STAGE 3: Embed & Index")
        print("-" * 60)
        embedded_chunks = self.embedder.process_chunks(chunks)
        
        # Stage 4: Knowledge Graph
        print("STAGE 4: Knowledge Graph")
        print("-" * 60)
        self.graph.add_document(parsed_doc)
        self.graph.add_chunks(chunks)
        
        # Save graph
        graph_path = self.output_dir / f"{pdf_path.stem}_graph.json"
        self.graph.save(str(graph_path))
        print(f"Graph saved: {graph_path}")
        print(f"✓ Knowledge graph built ({self.graph.get_stats()['nodes']} nodes)\n")
        
        # Summary
        print(f"{'='*60}")
        print("✓ Processing Complete!")
        print(f"{'='*60}")
        print(f"Document: {pdf_path.name}")
        print(f"  - Sections: {len(parsed_doc.sections)}")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Embeddings: {len(embedded_chunks)}")
        print(f"  - Graph nodes: {self.graph.get_stats()['nodes']}")
        print(f"{'='*60}\n")
        
        return {
            "document": parsed_doc,
            "chunks": chunks,
            "embedded_chunks": embedded_chunks
        }
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search documents.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Search results
        """
        print(f"\nSearching for: '{query}'")
        print("-" * 60)
        
        results = self.embedder.search(query, n_results=n_results)
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, dist, meta) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "text": doc,
                    "score": f"{(1 - dist):.3f}",
                    "document": meta.get('document', 'N/A'),
                    "section": meta.get('section_title', 'N/A'),
                    "pages": meta.get('pages', [])
                })
        
        print(f"Found {len(formatted_results)} results\n")
        return {"query": query, "results": formatted_results}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "vector_db": self.embedder.get_stats(),
            "knowledge_graph": self.graph.get_stats()
        }
