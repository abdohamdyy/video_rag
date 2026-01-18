"""Vector store module using Chroma DB."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# Default collection name
DEFAULT_COLLECTION_NAME = "pdf_documents"


class VectorStore:
    """Vector store wrapper for Chroma DB."""

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        Initialize Chroma DB vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Chroma DB client and collection."""
        logger.info(f"Initializing Chroma DB at: {self.persist_directory}")

        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        if not texts or not embeddings or not metadatas:
            raise ValueError("texts, embeddings, and metadatas must be non-empty lists")

        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("texts, embeddings, and metadatas must have the same length")

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        logger.info(f"Adding {len(texts)} documents to vector store")

        # Clean metadata: remove None values (ChromaDB doesn't accept None)
        cleaned_metadatas = [
            {k: v for k, v in metadata.items() if v is not None}
            for metadata in metadatas
        ]

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=cleaned_metadatas,
                ids=ids,
            )
            logger.info(f"Successfully added {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the vector store.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dictionary containing:
            - ids: List of document IDs
            - distances: List of distances (lower is better)
            - documents: List of document texts
            - metadatas: List of metadata dictionaries
        """
        logger.debug(f"Searching for {n_results} similar documents")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            # Chroma returns results in a nested format
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            }
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
        }

    def delete_collection(self) -> None:
        """Delete the collection (use with caution)."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self._initialize()  # Reinitialize to create a new collection

