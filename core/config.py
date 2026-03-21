"""Application settings via Pydantic BaseSettings (auto env-var binding)."""

from pydantic import Field
from pydantic_settings import BaseSettings


class KafkaSettings(BaseSettings):
    """Kafka A2A configuration (§5, §8)."""
    model_config = {"env_prefix": "KAFKA_"}

    enabled: bool = Field(default=False, alias="ENABLE_KAFKA_BUS")
    bootstrap_servers: str = "localhost:9092"
    request_timeout_ms: int = 30000
    # Producer tuning
    compression_type: str = "lz4"
    linger_ms: int = 5
    batch_size: int = 16384
    acks: str = "all"
    producer_retries: int = 5
    # Consumer tuning
    max_poll_records: int = 10
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    fetch_min_bytes: int = 1
    # Resilience
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_s: float = 60.0
    retry_max_attempts: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    enable_dlq: bool = True
    enable_idempotency: bool = True
    idempotency_ttl_s: float = 300.0
    idempotency_max_entries: int = 50000
    # Security (SASL/TLS)
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str = ""
    sasl_username: str = ""
    sasl_password: str = ""
    ssl_cafile: str = ""
    ssl_certfile: str = ""
    ssl_keyfile: str = ""


class OtelSettings(BaseSettings):
    """OpenTelemetry configuration (§5, §8)."""
    model_config = {"env_prefix": "OTEL_"}

    enabled: bool = Field(default=False, alias="ENABLE_OTEL")
    exporter_endpoint: str = Field(default="localhost:4317", alias="OTEL_EXPORTER_ENDPOINT")
    service_name: str = Field(default="idp-agent", alias="OTEL_SERVICE_NAME")
    sample_rate: float = Field(default=1.0, alias="OTEL_SAMPLE_RATE")
    export_console: bool = Field(default=False, alias="OTEL_EXPORT_CONSOLE")


class Neo4jSettings(BaseSettings):
    """Neo4j Knowledge Graph configuration (§6.4)."""
    model_config = {"env_prefix": "NEO4J_"}

    enabled: bool = Field(default=False, alias="ENABLE_NEO4J")
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4jpassword"
    database: str = "neo4j"


class Settings(BaseSettings):
    """Main application settings — auto-populated from environment variables."""
    model_config = {"env_prefix": "", "populate_by_name": True}

    database_url: str = Field(default="", alias="DATABASE_URL")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_pool_min: int = Field(default=1, alias="DB_POOL_MIN")
    db_pool_overflow: int = Field(default=10, alias="DB_POOL_OVERFLOW")
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_dim: int = 768
    embedding_batch_size: int = Field(default=8, alias="EMBEDDING_BATCH_SIZE")
    data_dir: str = Field(default="data", alias="IDP_DATA_DIR")
    disable_di: bool = Field(default=False, alias="DISABLE_DI")
    enable_hybrid_retrieval: bool = Field(default=False, alias="ENABLE_HYBRID_RETRIEVAL")
    enable_verifier: bool = Field(default=False, alias="ENABLE_VERIFIER")
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")
    coverage_mode: str = Field(default="llm_fallback", alias="COVERAGE_MODE")
    enable_document_facts: bool = Field(default=False, alias="ENABLE_DOCUMENT_FACTS")
    enable_classifier: bool = Field(default=False, alias="ENABLE_CLASSIFIER")
    enable_preprocessor: bool = Field(default=False, alias="ENABLE_PREPROCESSOR")
    front_matter_pages: int = Field(default=10, alias="FRONT_MATTER_PAGES")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    enable_redis_working_memory: bool = Field(default=True, alias="ENABLE_REDIS_WORKING_MEMORY")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_working_memory_ttl: int = Field(default=900, alias="REDIS_WORKING_MEMORY_TTL")
    # ── Scaling & Concurrency ───────────────────────────────────────────
    bulk_insert_batch_size: int = Field(default=500, alias="BULK_INSERT_BATCH_SIZE")
    circuit_breaker_window_s: float = Field(default=60.0, alias="CIRCUIT_BREAKER_WINDOW_S")
    circuit_breaker_threshold: int = Field(default=5, alias="CIRCUIT_BREAKER_THRESHOLD")
    ingestion_worker_threads: int = Field(default=2, alias="INGESTION_WORKER_THREADS")
    kafka_fanout_workers: int = Field(default=16, alias="KAFKA_FANOUT_WORKERS")
    # ── Sub-configs ──────────────────────────────────────────────────────
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    otel: OtelSettings = Field(default_factory=OtelSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    # ── Agent Evaluation ───────────────────────────────────────────────
    enable_agent_eval: bool = Field(default=False, alias="ENABLE_AGENT_EVAL")
    agent_eval_log_dir: str = Field(default="eval_logs", alias="AGENT_EVAL_LOG_DIR")
    # ── Schema-Driven Extraction (§10) ────────────────────────────────
    enable_extractor: bool = Field(default=False, alias="ENABLE_EXTRACTOR")
    enable_transformer: bool = Field(default=False, alias="ENABLE_TRANSFORMER")
    # ── MCP Reference Data ────────────────────────────────────────────
    mcp_reference_server_url: str = Field(default="http://localhost:8100", alias="MCP_REFERENCE_SERVER_URL")
    mcp_request_timeout_s: int = Field(default=10, alias="MCP_REQUEST_TIMEOUT_S")
    # ── Azure OpenAI ──────────────────────────────────────────────────
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-06-01", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_id: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_ID")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")  # "openai" | "azure_openai"
    # ── Multi-Format Ingestion ────────────────────────────────────────
    enable_multi_format: bool = Field(default=False, alias="ENABLE_MULTI_FORMAT")
    # ── Parser Backend ────────────────────────────────────────────────
    parser_backend: str = Field(default="pymupdf", alias="PARSER_BACKEND")  # "pymupdf" | "docling"


settings = Settings()
