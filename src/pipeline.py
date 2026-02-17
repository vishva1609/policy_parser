"""
Main pipeline orchestrator with enhanced features.
"""
from pathlib import Path
from typing import Dict, Any
import json

from .parser import SimplePDFParser
from .chunker import DocumentChunker
from .embedder import EmbeddingPipeline
from .knowledge_graph import KnowledgeGraph
from .excel_generator import ExcelGenerator
from .chunk_file_writer import ChunkFileWriter


class DocumentPipeline:
    """End-to-end document processing pipeline with Excel and file-per-chunk features."""
    
    def __init__(
        self,
        chunk_by: str = "section",
        embedding_model: str = "all-MiniLM-L6-v2",
        output_dir: str = "./output",
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 500
    ):
        """
        Initialize pipeline.
        
        Args:
            chunk_by: "section" or "paragraph"
            embedding_model: Sentence transformer model name
            output_dir: Output directory
            max_chunk_chars: Maximum characters per chunk
            min_chunk_chars: Minimum characters per chunk
        """
        print("=" * 60)
        print("Document Processing Pipeline")
        print("=" * 60)
        
        self.parser = SimplePDFParser()
        self.chunker = DocumentChunker(
            chunk_by=chunk_by,
            max_chars=max_chunk_chars,
            min_chars=min_chunk_chars
        )
        self.embedder = EmbeddingPipeline(model_name=embedding_model)
        self.graph = KnowledgeGraph()
        self.excel_generator = ExcelGenerator(output_dir)
        self.file_writer = ChunkFileWriter(f"{output_dir}/chunks")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Configuration:")
        print(f"  - Chunking: {chunk_by} (max {max_chunk_chars} chars)")
        print(f"  - Embedding: {embedding_model}")
        print(f"  - Output: {output_dir}")
        print("=" * 60)
    
    def process_document(
        self, 
        pdf_path: str,
        output_format: str = "txt",
        overwrite_files: bool = False
    ) -> Dict[str, Any]:
        """
        Process a PDF through the complete pipeline with enhanced features.
        
        Args:
            pdf_path: Path to PDF file
            output_format: "txt", "json", or "md" for chunk files
            overwrite_files: If False, skip existing chunk files
            
        Returns:
            Processing results with all outputs
        """
        pdf_path = Path(pdf_path)
        
        print(f"\n{'='*70}")
        print(f"ENHANCED DOCUMENT PROCESSING")
        print(f"{'='*70}\n")
        
        # Stage 1: Parse
        print("Step 1: Parsing PDF...")
        parsed_doc = self.parser.parse_pdf(str(pdf_path))
        print(f"Parsed {parsed_doc.total_pages} pages, {len(parsed_doc.sections)} sections")
        
        # Stage 2: Intelligent Chunking
        print(f"\nStep 2: Intelligent chunking (max {self.chunker.max_chars} chars per chunk)...")
        chunks = self.chunker.chunk_document(parsed_doc)
        
        # Display chunk statistics
        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        print(f"  - Average chunk size: {avg_size:.0f} characters")
        print(f"  - Largest chunk: {max(len(c.text) for c in chunks)} characters")
        print(f"  - Smallest chunk: {min(len(c.text) for c in chunks)} characters")
        
        # Stage 3: Generate Excel Index
        print("\nStep 3: Generating Excel searchable index...")
        excel_path = self.excel_generator.generate_searchable_index(parsed_doc, chunks)
        
        # Stage 4: Write Individual Chunk Files
        print(f"\nStep 4: Writing chunks to individual {output_format} files...")
        chunk_files = self.file_writer.write_chunks_to_files(
            chunks, 
            parsed_doc.filename, 
            format=output_format,
            overwrite=overwrite_files
        )
        
        # Stage 5: Create Chunk Index
        print("\nStep 5: Creating chunk index...")
        index_path = self.file_writer.create_index_file(chunks, parsed_doc.filename, overwrite=overwrite_files)
        
        # Stage 6: Embed & Index
        print("\nStep 6: Generating embeddings and indexing...")
        embedded_chunks = self.embedder.process_chunks(chunks)
        
        # Stage 7: Knowledge Graph
        print("\nStep 7: Building knowledge graph...")
        self.graph.add_document(parsed_doc)
        self.graph.add_chunks(chunks)
        
        # Save outputs (parsed doc and graph only - chunks already saved as individual files)
        parsed_path = self.output_dir / f"{pdf_path.stem}_parsed.json"
        with open(parsed_path, 'w', encoding='utf-8') as f:
            f.write(parsed_doc.model_dump_json(indent=2))
        
        graph_path = self.output_dir / f"{pdf_path.stem}_graph.json"
        self.graph.save(str(graph_path))
        print(f"Saved JSON outputs to {self.output_dir}")
        
        # Summary
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"Excel Index: {excel_path}")
        print(f"Chunk Files: {len(chunk_files)} files in {Path(chunk_files[0]).parent}")
        print(f"Chunk Index: {index_path}")
        print(f"{'='*70}\n")
        
        return {
            "document": parsed_doc,
            "chunks": chunks,
            "embedded_chunks": embedded_chunks,
            "excel_path": excel_path,
            "chunk_files": chunk_files,
            "index_path": index_path,
            "stats": {
                "total_pages": parsed_doc.total_pages,
                "total_sections": len(parsed_doc.sections),
                "total_chunks": len(chunks),
                "avg_chunk_size": int(avg_size),
                "total_files_created": len(chunk_files) + 1,
                "output_format": output_format
            }
        }
    
    def search_in_excel(self, excel_path: str, search_term: str):
        """Search the Excel file and display results."""
        results = self.excel_generator.search_excel(excel_path, search_term)
        
        if results.empty:
            print(f"No results found for '{search_term}'")
        else:
            print(f"\nFound {len(results)} results for '{search_term}':\n")
            for idx, row in results.iterrows():
                print(f"[{idx+1}] Type: {row['Type']} | Section: {row['Section']}")
                print(f"    Pages: {row['Page Numbers']}")
                print(f"    Content: {row['Content'][:150]}...")
                print("-" * 70)
        
        return results
    
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
