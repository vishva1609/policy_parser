"""
BaseEmbedder — Abstract base class for vector embedding pipelines.

Defines the interface that all concrete embedder implementations must satisfy.
Any alternative embedding backend (OpenAI, Cohere, local models) can plug in
by inheriting this ABC and implementing the two abstract methods.

Inheritance hierarchy:
    BaseEmbedder (ABC)
        └── EmbeddingPipeline  — ChromaDB + sentence-transformers implementation

Usage::

    class MyOpenAIEmbedder(BaseEmbedder):
        def process_chunks(self, chunks, batch_size=32):
            ...
        def search(self, query, n_results=5, filter_metadata=None):
            ...
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding pipelines.

    Defines the two-method contract:
        process_chunks() — embed a list of chunks and store them.
        search()         — retrieve similar chunks by semantic query.

    Concrete subclasses swap in different embedding models or vector stores
    without any changes to the service or views layers.
    """

    @abstractmethod
    def process_chunks(self, chunks: List[Any], batch_size: int = 32) -> List[Any]:
        """
        Generate embeddings for chunks and store in the vector database.

        Args:
            chunks:     List of DocumentChunk objects to embed.
            batch_size: Processing batch size.

        Returns:
            List of EmbeddedChunk objects with embedding vectors.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks most semantically similar to the query.

        Args:
            query:           Natural language search query.
            n_results:       Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            Dict with keys: documents, distances, metadatas, ids.
        """
        pass
