import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()  # loads local .env if present (do not commit your real key)


class Settings(BaseModel):
    gemini_api_key: str = Field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    max_video_bytes: int = Field(default_factory=lambda: int(os.environ.get("MAX_VIDEO_BYTES", "26214400")))
    
    # Chroma DB settings
    chroma_db_path: str = Field(default_factory=lambda: os.environ.get("CHROMA_DB_PATH", "./chroma_db"))
    chroma_collection_name: str = Field(default_factory=lambda: os.environ.get("CHROMA_COLLECTION_NAME", "pdf_documents"))

    # BGE-M3 embedding model settings
    embedding_model_name: str = Field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
    embedding_batch_size: int = Field(default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "0")))  # 0 = auto

    # Retrieval and answering settings
    retrieval_top_k: int = Field(default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K", "6")))
    retrieval_per_query_k: int = Field(default_factory=lambda: int(os.environ.get("RETRIEVAL_PER_QUERY_K", "4")))
    gemini_answer_model: str = Field(default_factory=lambda: os.environ.get("GEMINI_ANSWER_MODEL", "gemini-2.5-pro"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


