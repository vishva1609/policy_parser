"""
Embedding Pipeline for Document Chunks
"""
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .schemas import DocumentChunk, EmbeddedChunk


class EmbeddingPipeline:
    """
    Generates embeddings for document chunks and manages vector database.
    
    Uses sentence-transformers for semantic embeddings and ChromaDB
    for persistent vector storage and similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "./output/vectordb",
        collection_name: str = "documents"
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            model_name: Name of sentence-transformers model
            db_path: Path to ChromaDB persistence directory
            collection_name: Name of the collection in ChromaDB
        """
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Initialize model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def process_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for chunks and store in vector database.
        
        Args:
            chunks: List of document chunks to embed
            batch_size: Number of chunks to process at once
            
        Returns:
            List of embedded chunks with vectors
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding=embedding.tolist(),
                metadata={
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type,
                    "parent_context": chunk.parent_context,
                    "pages": chunk.pages
                }
            )
            embedded_chunks.append(embedded_chunk)
        
        # Index in ChromaDB
        self._index_chunks(embedded_chunks)
        
        return embedded_chunks
    
    def _index_chunks(self, embedded_chunks: List[EmbeddedChunk]):
        """
        Index embedded chunks in ChromaDB.
        
        Args:
            embedded_chunks: List of chunks with embeddings
        """
        print(f"Indexing {len(embedded_chunks)} chunks in vector database...")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in embedded_chunks]
        embeddings = [chunk.embedding for chunk in embedded_chunks]
        documents = [chunk.text for chunk in embedded_chunks]
        metadatas = [chunk.metadata for chunk in embedded_chunks]
        
        # Add to collection in batches
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Indexing"):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"✓ Indexed {len(embedded_chunks)} chunks")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary with search results and metadata
        """
        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        count = self.collection.count()
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "model": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }
