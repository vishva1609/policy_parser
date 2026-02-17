"""
Chunk File Writer - Writes each chunk to a separate file
"""
from pathlib import Path
from typing import List
import json
from .schemas import DocumentChunk


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
