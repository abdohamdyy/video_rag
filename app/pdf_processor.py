"""PDF processing module for extracting text and chunking."""

import logging
from pathlib import Path
from typing import List, Tuple

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Initialize tokenizer for chunking
ENCODING = tiktoken.get_encoding("cl100k_base")  # Used by GPT models, good for general text


def extract_text_from_pdf(pdf_path: str | Path) -> List[Tuple[str, int]]:
    """
    Extract text from PDF file, returning list of (text, page_number) tuples.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of tuples containing (text, page_number) for each page
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            pages.append((text, page_num))
            logger.debug(f"Extracted {len(text)} characters from page {page_num}")

    logger.info(f"Extracted text from {len(pages)} pages")
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding: tiktoken.Encoding = ENCODING,
) -> List[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        encoding: Tokenizer encoding to use

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Tokenize the text
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move start position with overlap
        start = end - chunk_overlap

        # Prevent infinite loop
        if start >= len(tokens):
            break

    return chunks


def process_pdf_to_chunks(
    pdf_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[dict]:
    """
    Process PDF file and return list of chunks with metadata.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks

    Returns:
        List of dictionaries containing:
        - text: The chunk text
        - page_number: Page number where chunk came from
        - chunk_index: Index of chunk within the page
        - source_file: Name of the PDF file
    """
    pdf_path = Path(pdf_path)
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []

    for page_text, page_num in pages:
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx, chunk_content in enumerate(page_chunks):
            all_chunks.append(
                {
                    "text": chunk_content,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "source_file": pdf_path.name,
                }
            )

    logger.info(f"Created {len(all_chunks)} chunks from {pdf_path.name}")
    return all_chunks

