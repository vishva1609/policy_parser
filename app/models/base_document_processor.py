"""
BaseDocumentProcessor — Abstract base class for document I/O.

This is the EXACT equivalent of FileProcessor(ABC) from archit1012/qa-bot-llm:

    class FileProcessor(ABC):
        @abstractmethod
        def document_loader(self, file_path): pass

        @abstractmethod
        def text_splitter(self, documents): pass

        @abstractmethod
        def prepare_vectordb(self, docs, embeddings): pass

We use the SAME method names so the service layer can call them identically:

    # service/qa_apis_service.py (reference):
    llm = PdfFileProcessor()
    documents = llm.document_loader(file_path)
    docs      = llm.text_splitter(documents)
    db        = llm.prepare_vectordb(docs, embeddings)

    # Our service/document_service.py:
    processor = SimplePDFParser()
    documents = processor.document_loader(file_path)
    docs      = processor.text_splitter(documents)
    db        = processor.prepare_vectordb(docs, embedder)   ← returns db

Inheritance hierarchy:
    BaseDocumentProcessor (ABC)
        ├── SimplePDFParser   — implements document_loader() + text_splitter()
        └── DocumentChunker   — implements text_splitter() (splits already-parsed docs)
"""
from abc import ABC, abstractmethod
from typing import List, Any


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processing.

    Exact mirror of FileProcessor(ABC) from archit1012/qa-bot-llm —
    uses identical method names so the service layer's polymorphic calls
    work without knowing the concrete type.
    """

    @abstractmethod
    def document_loader(self, file_path: str) -> Any:
        """
        Load and parse a document from disk.

        Exact equivalent of FileProcessor.document_loader() in reference repo.

        Args:
            file_path: Absolute or relative path to the source document.

        Returns:
            A parsed document representation (ParsedDocument or similar).
        """
        pass

    @abstractmethod
    def text_splitter(self, documents: Any) -> List[Any]:
        """
        Split a parsed document into chunks suitable for embedding.

        Exact equivalent of FileProcessor.text_splitter() in reference repo.

        Args:
            documents: Parsed document (output of document_loader()).

        Returns:
            List of document chunks.
        """
        pass

    @abstractmethod
    def prepare_vectordb(self, docs: List[Any], embedder: Any) -> Any:
        """
        Store document chunks in a vector database via an embedder.

        Exact equivalent of FileProcessor.prepare_vectordb(docs, embeddings)
        in reference repo — takes the built embedder/db and returns it
        so the service layer can later query it.

        Args:
            docs:    List of document chunks (output of text_splitter()).
            embedder: An embedder instance (EmbeddingPipeline or BaseEmbedder).
                      Equivalent to the OpenAIEmbeddings passed in reference repo.

        Returns:
            The populated embedder/vector store (so caller can search it).
        """
        pass
