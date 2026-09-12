from dataclasses import dataclass
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = ROOT / "sample_documents"
    db_dir: Path = ROOT / ".chroma"
    chunks_file: Path = ROOT / ".chroma" / "chunks.json"
    collection_name: str = "advanced_rag"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_repo: str = os.getenv(
        "HF_LLM_REPO", "meta-llama/Llama-3.1-8B-Instruct"
    )
    rerank_model: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_enabled: bool = _as_bool(os.getenv("RERANK_ENABLED", "true"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.38"))
    chunk_size: int = 850
    chunk_overlap: int = 140
    initial_candidates: int = 10
    final_results: int = 4

