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


settings = Settings()
