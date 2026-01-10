"""Script to process PDF files and create embeddings using BGE-M3."""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings import create_embeddings
from app.pdf_processor import process_pdf_to_chunks
from app.settings import get_settings
from app.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def embed_pdf_file(pdf_path: str | Path, vector_store: VectorStore, settings) -> None:
    """
    Process a single PDF file and add its embeddings to the vector store.

    Args:
        pdf_path: Path to the PDF file
        vector_store: VectorStore instance
        settings: Settings instance
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    logger.info(f"Processing PDF: {pdf_path.name}")

    # Process PDF to chunks
    chunks = process_pdf_to_chunks(
        pdf_path,
        chunk_size=512,  # Default chunk size
        chunk_overlap=50,  # Default overlap
    )

    if not chunks:
        logger.warning(f"No chunks extracted from {pdf_path.name}")
        return

    # Extract texts and metadatas
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    # Generate IDs based on file name and chunk index
    ids = [f"{pdf_path.stem}_page{chunk['page_number']}_chunk{chunk['chunk_index']}" for chunk in chunks]

    # Create embeddings
    logger.info(f"Creating embeddings for {len(texts)} chunks using {settings.embedding_model_name}")
    batch_size = settings.embedding_batch_size if settings.embedding_batch_size > 0 else None
    embeddings = create_embeddings(
        texts,
        model_name=settings.embedding_model_name,
        batch_size=batch_size,
    )

    # Add to vector store
    logger.info(f"Adding {len(texts)} documents to vector store")
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"Successfully processed {pdf_path.name}: {len(chunks)} chunks added")


def main() -> None:
    """Main function to process PDF files."""
    settings = get_settings()

    # Get project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent

    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        persist_directory=settings.chroma_db_path,
        collection_name=settings.chroma_collection_name,
    )

    # Get collection info
    info = vector_store.get_collection_info()
    logger.info(f"Vector store initialized: {info}")

    # PDF files to process (relative to project root)
    pdf_files = [
        "2024-SCA_SalesBrochure_digital_USA-324-1-min.pdf",
        "Productcatalog2025.pdf",
    ]

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = project_root / pdf_file
        if pdf_path.exists():
            try:
                embed_pdf_file(pdf_path, vector_store, settings)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}", exc_info=True)
        else:
            logger.warning(f"PDF file not found: {pdf_path}")

    # Final collection info
    final_info = vector_store.get_collection_info()
    logger.info(f"Processing complete. Final collection info: {final_info}")


if __name__ == "__main__":
    main()

