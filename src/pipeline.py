"""
Main pipeline orchestrator with schemas, knowledge graph, and chunk writer.

Merges former schemas.py, knowledge_graph.py, chunk_file_writer.py, and pipeline.py.
"""
from pathlib import Path
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import networkx as nx

from .schemas import DocumentElement, DocumentSection, ParsedDocument, DocumentChunk, EmbeddedChunk
from .document_processor import SimplePDFParser, DocumentChunker
from .embedder import EmbeddingPipeline
from .excel_generator import ExcelGenerator


# ============================================================
# Knowledge Graph (formerly knowledge_graph.py)
# ============================================================

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


# ============================================================
# Chunk File Writer (formerly chunk_file_writer.py)
# ============================================================

class ChunkFileWriter:
    """Write each chunk to a separate file (with overwrite protection)"""
    
    def __init__(self, output_dir: str = "output/chunks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_chunks_to_files(
        self, 
        chunks: List[DocumentChunk],
        document_name: str,
        format: str = "txt",
        overwrite: bool = False
    ) -> List[str]:
        """
        Write each chunk to a separate file.
        
        Args:
            chunks: List of document chunks
            document_name: Base name for files
            format: File format ('txt', 'json', or 'md')
            overwrite: If False, skip existing files
        
        Returns:
            List of file paths created/skipped
        """
        base_name = Path(document_name).stem
        file_paths = []
        skipped = 0
        created = 0
        
        # Create document-specific directory
        doc_dir = self.output_dir / base_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        for i, chunk in enumerate(chunks, 1):
            if format == "txt":
                file_path = doc_dir / f"chunk_{i:03d}.txt"
            elif format == "json":
                file_path = doc_dir / f"chunk_{i:03d}.json"
            elif format == "md":
                file_path = doc_dir / f"chunk_{i:03d}.md"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Check if file exists
            if file_path.exists() and not overwrite:
                skipped += 1
                file_paths.append(str(file_path))
                continue
            
            # Write file
            if format == "txt":
                self._write_txt(file_path, chunk, i)
            elif format == "json":
                self._write_json(file_path, chunk, i)
            elif format == "md":
                self._write_markdown(file_path, chunk, i)
            
            created += 1
            file_paths.append(str(file_path))
        
        if skipped > 0:
            print(f"Created {created} chunks, skipped {skipped} existing files in {doc_dir}")
        else:
            print(f"Written {created} chunks to {doc_dir}")
        
        return file_paths
    
    def _write_txt(self, file_path: Path, chunk: DocumentChunk, index: int):
        """Write chunk as plain text file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== CHUNK {index} ===\n")
            f.write(f"ID: {chunk.chunk_id}\n")
            f.write(f"Section: {chunk.parent_context[0] if chunk.parent_context else 'N/A'}\n")
            f.write(f"Pages: {chunk.pages}\n")
            f.write(f"Type: {chunk.chunk_type}\n")
            f.write(f"Characters: {len(chunk.text)}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(chunk.text)
    
    def _write_json(self, file_path: Path, chunk: DocumentChunk, index: int):
        """Write chunk as JSON file"""
        data = {
            'chunk_index': index,
            'chunk_id': chunk.chunk_id,
            'document_id': chunk.document_id,
            'text': chunk.text,
            'pages': chunk.pages,
            'chunk_type': chunk.chunk_type,
            'parent_context': chunk.parent_context,
            'metadata': chunk.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _write_markdown(self, file_path: Path, chunk: DocumentChunk, index: int):
        """Write chunk as Markdown file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {index}\n\n")
            f.write(f"**ID:** `{chunk.chunk_id}`  \n")
            f.write(f"**Section:** {chunk.parent_context[0] if chunk.parent_context else 'N/A'}  \n")
            f.write(f"**Pages:** {chunk.pages}  \n")
            f.write(f"**Type:** {chunk.chunk_type}  \n")
            f.write(f"**Length:** {len(chunk.text)} characters  \n\n")
            f.write("---\n\n")
            f.write(chunk.text)
    
    def create_index_file(
        self, 
        chunks: List[DocumentChunk],
        document_name: str,
        overwrite: bool = False
    ) -> str:
        """Create an index file listing all chunks"""
        base_name = Path(document_name).stem
        doc_dir = self.output_dir / base_name
        index_path = doc_dir / "index.json"
        
        # Check if exists
        if index_path.exists() and not overwrite:
            print(f"Index file already exists: {index_path}")
            return str(index_path)
        
        index_data = {
            'document': document_name,
            'total_chunks': len(chunks),
            'chunks': []
        }
        
        for i, chunk in enumerate(chunks, 1):
            index_data['chunks'].append({
                'chunk_number': i,
                'chunk_id': chunk.chunk_id,
                'section': chunk.parent_context[0] if chunk.parent_context else 'N/A',
                'pages': chunk.pages,
                'chunk_type': chunk.chunk_type,
                'file': f"chunk_{i:03d}.txt",
                'char_count': len(chunk.text),
                'preview': chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text
            })
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created index file: {index_path}")
        return str(index_path)


# ============================================================
# Document Pipeline (orchestrator)
# ============================================================

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
