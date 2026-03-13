from dataclasses import dataclass
import os


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class KafkaSettings:
    """Kafka A2A configuration (§5, §8)."""
    enabled: bool = _get_bool_env("ENABLE_KAFKA_BUS", False)
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    request_timeout_ms: int = int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000"))
    # Producer tuning
    compression_type: str = os.getenv("KAFKA_COMPRESSION_TYPE", "lz4")
    linger_ms: int = int(os.getenv("KAFKA_LINGER_MS", "5"))
    batch_size: int = int(os.getenv("KAFKA_BATCH_SIZE", "16384"))
    acks: str = os.getenv("KAFKA_ACKS", "all")
    producer_retries: int = int(os.getenv("KAFKA_PRODUCER_RETRIES", "5"))
    # Consumer tuning
    max_poll_records: int = int(os.getenv("KAFKA_MAX_POLL_RECORDS", "10"))
    session_timeout_ms: int = int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000"))
    heartbeat_interval_ms: int = int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000"))
    fetch_min_bytes: int = int(os.getenv("KAFKA_FETCH_MIN_BYTES", "1"))
    # Resilience
    circuit_breaker_threshold: int = int(os.getenv("KAFKA_CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_cooldown_s: float = float(os.getenv("KAFKA_CIRCUIT_BREAKER_COOLDOWN_S", "60"))
    retry_max_attempts: int = int(os.getenv("KAFKA_RETRY_MAX_ATTEMPTS", "3"))
    retry_base_delay_s: float = float(os.getenv("KAFKA_RETRY_BASE_DELAY_S", "1.0"))
    retry_max_delay_s: float = float(os.getenv("KAFKA_RETRY_MAX_DELAY_S", "30.0"))
    enable_dlq: bool = _get_bool_env("KAFKA_ENABLE_DLQ", True)
    enable_idempotency: bool = _get_bool_env("KAFKA_ENABLE_IDEMPOTENCY", True)
    idempotency_ttl_s: float = float(os.getenv("KAFKA_IDEMPOTENCY_TTL_S", "300"))
    idempotency_max_entries: int = int(os.getenv("KAFKA_IDEMPOTENCY_MAX_ENTRIES", "50000"))
    # Security (SASL/TLS)
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    sasl_mechanism: str = os.getenv("KAFKA_SASL_MECHANISM", "")
    sasl_username: str = os.getenv("KAFKA_SASL_USERNAME", "")
    sasl_password: str = os.getenv("KAFKA_SASL_PASSWORD", "")
    ssl_cafile: str = os.getenv("KAFKA_SSL_CAFILE", "")
    ssl_certfile: str = os.getenv("KAFKA_SSL_CERTFILE", "")
    ssl_keyfile: str = os.getenv("KAFKA_SSL_KEYFILE", "")


@dataclass
class OtelSettings:
    """OpenTelemetry configuration (§5, §8)."""
    enabled: bool = _get_bool_env("ENABLE_OTEL", False)
    exporter_endpoint: str = os.getenv("OTEL_EXPORTER_ENDPOINT", "localhost:4317")
    service_name: str = os.getenv("OTEL_SERVICE_NAME", "idp-agent")
    sample_rate: float = float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))
    export_console: bool = _get_bool_env("OTEL_EXPORT_CONSOLE", False)


@dataclass
class Neo4jSettings:
    """Neo4j Knowledge Graph configuration (§6.4)."""
    enabled: bool = _get_bool_env("ENABLE_NEO4J", False)
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_dim: int = 768
    data_dir: str = os.getenv("IDP_DATA_DIR", "data")
    disable_di: bool = _get_bool_env("DISABLE_DI", False)
    enable_hybrid_retrieval: bool = _get_bool_env("ENABLE_HYBRID_RETRIEVAL", False)
    enable_verifier: bool = _get_bool_env("ENABLE_VERIFIER", False)
    enable_reranker: bool = _get_bool_env("ENABLE_RERANKER", False)
    coverage_mode: str = os.getenv("COVERAGE_MODE", "llm_fallback")
    enable_document_facts: bool = _get_bool_env("ENABLE_DOCUMENT_FACTS", False)
    enable_classifier: bool = _get_bool_env("ENABLE_CLASSIFIER", False)
    front_matter_pages: int = int(os.getenv("FRONT_MATTER_PAGES", "10"))
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    enable_redis_working_memory: bool = _get_bool_env("ENABLE_REDIS_WORKING_MEMORY", True)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_working_memory_ttl: int = int(os.getenv("REDIS_WORKING_MEMORY_TTL", "900"))
    # ── Sub-configs ──────────────────────────────────────────────────────
    kafka: KafkaSettings = None  # type: ignore[assignment]
    otel: OtelSettings = None  # type: ignore[assignment]
    neo4j: Neo4jSettings = None  # type: ignore[assignment]
    # ── Agent Evaluation ───────────────────────────────────────────────
    enable_agent_eval: bool = _get_bool_env("ENABLE_AGENT_EVAL", False)
    agent_eval_log_dir: str = os.getenv("AGENT_EVAL_LOG_DIR", "eval_logs")

    def __post_init__(self):
        if self.kafka is None:
            self.kafka = KafkaSettings()
        if self.otel is None:
            self.otel = OtelSettings()
        if self.neo4j is None:
            self.neo4j = Neo4jSettings()


settings = Settings()
