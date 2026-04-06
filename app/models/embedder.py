"""
EmbeddingPipeline — Concrete implementation of BaseEmbedder.

Generates embeddings for document chunks using sentence-transformers
and stores them in ChromaDB for persistent semantic search.

Inherits BaseEmbedder and implements:
    process_chunks() — embed + index chunks in ChromaDB.
    search()         — cosine similarity retrieval.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base_embedder import BaseEmbedder
from ..common.schemas import DocumentChunk, EmbeddedChunk


class EmbeddingPipeline(BaseEmbedder):
    """
    Generates embeddings for document chunks and manages ChromaDB vector store.

    Inherits BaseEmbedder — any alternative embedding backend (OpenAI, Cohere)
    can replace this class while keeping the service layer unchanged (open/closed).

    Attributes:
        model_name:       sentence-transformers model identifier.
        db_path:          Path to ChromaDB persistence directory.
        collection_name:  Name of the ChromaDB collection.
        model:            Loaded SentenceTransformer instance.
        collection:       ChromaDB collection handle.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "./output/vectordb",
        collection_name: str = "documents",
    ):
        """
        Initialise embedding pipeline.

        Args:
            model_name:      sentence-transformers model name.
            db_path:         Path to ChromaDB persistence directory.
            collection_name: Name of the collection in ChromaDB.
        """
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ── BaseEmbedder contract ──────────────────────────────────────────────────

    def process_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32,
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for chunks and store in ChromaDB.

        Implements BaseEmbedder.process_chunks() — equivalent to
        prepare_vectordb() in the reference FileProcessor ABC.

        Args:
            chunks:     List of DocumentChunk objects to embed.
            batch_size: Number of chunks to process at once.

        Returns:
            List of EmbeddedChunk objects with embedding vectors.
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")

        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

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
                    "parent_context": (
                        json.dumps(chunk.parent_context)
                        if isinstance(chunk.parent_context, list)
                        else str(chunk.parent_context or "")
                    ),
                    "chunk_pages": (
                        json.dumps(chunk.pages)
                        if isinstance(chunk.pages, list)
                        else str(chunk.pages or "")
                    ),
                },
            )
            embedded_chunks.append(embedded_chunk)

        self._index_chunks(embedded_chunks)
        return embedded_chunks

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar chunks using semantic similarity.

        Implements BaseEmbedder.search().

        Args:
            query:           Search query text.
            n_results:       Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            Dict with keys: documents, distances, metadatas, ids.
        """
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
        )
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _index_chunks(self, embedded_chunks: List[EmbeddedChunk]):
        """Index embedded chunks in ChromaDB using upsert (idempotent)."""
        print(f"Indexing {len(embedded_chunks)} chunks in vector database...")

        ids        = [chunk.chunk_id for chunk in embedded_chunks]
        embeddings = [chunk.embedding for chunk in embedded_chunks]
        documents  = [chunk.text for chunk in embedded_chunks]
        metadatas  = [self._sanitize_metadata(chunk.metadata) for chunk in embedded_chunks]

        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Indexing"):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"Indexed {len(embedded_chunks)} chunks")

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """
        ChromaDB only accepts str, int, float, bool as metadata values.
        Convert lists/dicts to JSON strings; convert None to empty string.
        """
        result = {}
        for k, v in meta.items():
            if isinstance(v, (bool, int, float, str)):
                result[k] = v
            elif isinstance(v, (list, dict)):
                result[k] = json.dumps(v)
            elif v is None:
                result[k] = ""
            else:
                result[k] = str(v)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "model": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
        }
