from dataclasses import dataclass
import os


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_dim: int = 768
    data_dir: str = os.getenv("IDP_DATA_DIR", "data")
    disable_di: bool = _get_bool_env("DISABLE_DI", False)
    enable_hybrid_retrieval: bool = _get_bool_env("ENABLE_HYBRID_RETRIEVAL", False)
    enable_verifier: bool = _get_bool_env("ENABLE_VERIFIER", False)
    enable_reranker: bool = _get_bool_env("ENABLE_RERANKER", False)
    coverage_mode: str = os.getenv("COVERAGE_MODE", "llm_fallback")
    enable_document_facts: bool = _get_bool_env("ENABLE_DOCUMENT_FACTS", False)
    front_matter_pages: int = int(os.getenv("FRONT_MATTER_PAGES", "10"))
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


settings = Settings()
