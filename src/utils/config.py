"""Configuration management for KnightGPT."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMSettings(BaseSettings):
    """vLLM server configuration."""

    # Embedding server
    embedding_url: str = Field(
        default="http://localhost:8001/v1",
        description="vLLM embedding server URL",
    )
    embedding_model: str = Field(
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        description="Embedding model name",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # Inference server
    inference_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM inference server URL",
    )
    inference_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct",
        description="Inference model name",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for generation",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
    )

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        extra="ignore",
    )


class Neo4jSettings(BaseSettings):
    """Neo4j database configuration."""

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    password: str = Field(
        default="password",
        description="Neo4j password",
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        extra="ignore",
    )


class PostgresSettings(BaseSettings):
    """Postgres (pgGraph + pgContext) database configuration."""

    dsn: str = Field(
        default="postgresql://postgres:password@localhost:5432/knightgpt",
        description="Postgres connection string (asyncpg format)",
    )
    pool_min_size: int = Field(
        default=2,
        description="Minimum connections in the asyncpg pool",
    )
    pool_max_size: int = Field(
        default=10,
        description="Maximum connections in the asyncpg pool",
    )

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        env_file=".env",
        extra="ignore",
    )


class GraphSettings(BaseSettings):
    """Knowledge graph configuration."""

    similarity_threshold: float = Field(
        default=0.7,
        description="Cosine similarity threshold for edge creation",
    )
    max_neighbors: int = Field(
        default=10,
        description="Maximum neighbors per node",
    )
    graph_path: Path = Field(
        default=Path("data/processed/graph.graphml"),
        description="Path to save/load graph",
    )

    model_config = SettingsConfigDict(
        env_prefix="GRAPH_",
        env_file=".env",
        extra="ignore",
    )


class ChunkingSettings(BaseSettings):
    """Text chunking configuration."""

    max_tokens: int = Field(
        default=500,
        description="Maximum tokens per chunk",
    )
    overlap_tokens: int = Field(
        default=50,
        description="Token overlap between chunks",
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in characters",
    )

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_",
        env_file=".env",
        extra="ignore",
    )


class IngestionSettings(BaseSettings):
    """Data ingestion configuration."""

    raw_pdf_dir: Path = Field(
        default=Path("data/raw_pdfs"),
        description="Directory for raw PDF files",
    )
    markdown_dir: Path = Field(
        default=Path("data/markdown"),
        description="Directory for converted markdown files",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed data",
    )
    force_ocr: bool = Field(
        default=False,
        description="Force OCR for all PDFs",
    )
    use_llm_enhancement: bool = Field(
        default=False,
        description="Use LLM for enhanced PDF parsing",
    )

    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=".env",
        extra="ignore",
    )


class RSSSettings(BaseSettings):
    """RSS feed ingestion configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable RSS feed discovery",
    )
    check_interval_hours: int = Field(
        default=168,
        description="Hours between RSS checks (default: weekly)",
    )
    kl_tools_api_url: Optional[str] = Field(
        default=None,
        description="kl-tools API URL for paper discovery",
    )
    max_papers_per_run: int = Field(
        default=50,
        description="Maximum papers to ingest per run",
    )
    filter_keywords: bool = Field(
        default=True,
        description="Filter papers by microbiome keywords",
    )

    model_config = SettingsConfigDict(
        env_prefix="RSS_",
        env_file=".env",
        extra="ignore",
    )


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    port: int = Field(
        default=8080,
        description="API server port",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication",
    )

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
    )


class Settings(BaseSettings):
    """Main application settings."""

    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    rss: RSSSettings = Field(default_factory=RSSSettings)
    api: APISettings = Field(default_factory=APISettings)

    # General settings
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def slurm_node(self) -> Optional[str]:
        """Get current SLURM node hostname from environment."""
        import os
        return os.environ.get("SLURM_NODELIST") or os.environ.get("SLURMD_NODENAME")

    def get_vllm_url(self, service: str = "embedding") -> str:
        """
        Get vLLM URL, auto-detecting SLURM node if running in a job.

        If SLURM_NODELIST is set and the URL is localhost, replace
        with the actual compute node hostname.
        """
        url = self.vllm.embedding_url if service == "embedding" else self.vllm.inference_url
        node = self.slurm_node
        if node and "localhost" in url:
            url = url.replace("localhost", node)
        return url


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
