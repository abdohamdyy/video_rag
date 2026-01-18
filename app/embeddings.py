"""Embedding module using BGE-M3 model."""

import logging
from functools import lru_cache
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE-M3 model name from Hugging Face
BGE_M3_MODEL_NAME = "BAAI/bge-m3"


def get_device() -> str:
    """
    Detect available device (GPU or CPU).
    Tests GPU compatibility before using it.

    Returns:
        Device string: 'cuda' if GPU available and compatible, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name} (CUDA {torch.version.cuda})")
        
        # Test GPU compatibility by trying a simple operation
        try:
            test_tensor = torch.randn(1, 1).cuda()
            _ = test_tensor * 2  # Simple operation
            del test_tensor
            torch.cuda.empty_cache()
            device = "cuda"
            logger.info(f"GPU is compatible and will be used")
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(
                    f"GPU detected but not compatible with current PyTorch version. "
                    f"Falling back to CPU. Error: {e}"
                )
                device = "cpu"
            else:
                # Re-raise if it's a different error
                raise
    else:
        device = "cpu"
        logger.info("No GPU detected, using CPU")
    return device


# Cache models per device
_model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def get_embedding_model(
    model_name: str = BGE_M3_MODEL_NAME, device: Optional[str] = None
) -> SentenceTransformer:
    """
    Load and cache the BGE-M3 embedding model.

    Args:
        model_name: Name of the model to load (default: BAAI/bge-m3)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        Loaded SentenceTransformer model
    """
    if device is None:
        device = get_device()

    # Check cache
    cache_key = (model_name, device)
    if cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name} on {device}")
        return _model_cache[cache_key]

    logger.info(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    _model_cache[cache_key] = model
    logger.info(f"Embedding model loaded successfully on {device}")
    return model


def create_embeddings(
    texts: List[str],
    model_name: str = BGE_M3_MODEL_NAME,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> List[List[float]]:
    """
    Create embeddings for a list of texts using BGE-M3.

    Args:
        texts: List of text strings to embed
        model_name: Name of the model to use (default: BAAI/bge-m3)
        batch_size: Batch size for processing. If None, auto-select based on device.
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []

    if device is None:
        device = get_device()

    # Auto-select batch size based on device
    if batch_size is None:
        if device == "cuda":
            # Larger batch size for GPU (8GB VRAM can handle ~32-64)
            batch_size = 32
        else:
            # Smaller batch size for CPU
            batch_size = 8

    model = get_embedding_model(model_name, device=device)
    logger.info(f"Creating embeddings for {len(texts)} texts (batch_size={batch_size}, device={device})")

    # BGE-M3 supports batch processing
    # The model returns numpy arrays, convert to list
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except RuntimeError as e:
        # If GPU operation fails, fall back to CPU
        if device == "cuda" and ("cuda" in str(e).lower() or "kernel" in str(e).lower()):
            logger.warning(
                f"GPU operation failed, falling back to CPU. Error: {e}"
            )
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Switch to CPU and retry
            device = "cpu"
            batch_size = 8  # Smaller batch for CPU
            model = get_embedding_model(model_name, device=device)
            logger.info(f"Retrying with CPU (batch_size={batch_size})")
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            # Re-raise if it's a different error
            raise

    # Convert numpy array to list of lists
    if embeddings.ndim == 1:
        embeddings = [embeddings.tolist()]
    else:
        embeddings = embeddings.tolist()

    logger.debug(f"Created embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
    return embeddings


def create_embedding(text: str, model_name: str = BGE_M3_MODEL_NAME) -> List[float]:
    """
    Create embedding for a single text using BGE-M3.

    Args:
        text: Text string to embed
        model_name: Name of the model to use (default: BAAI/bge-m3)

    Returns:
        Embedding vector as a list of floats
    """
    embeddings = create_embeddings([text], model_name=model_name)
    return embeddings[0] if embeddings else []

