"""Tests for Postgres settings."""

import pytest


@pytest.mark.unit
def test_postgres_settings_defaults(monkeypatch):
    """PostgresSettings should read POSTGRES_DSN and expose a default."""
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    from src.utils.config import PostgresSettings

    settings = PostgresSettings()
    assert settings.dsn == "postgresql://postgres:password@localhost:5432/knightgpt"


@pytest.mark.unit
def test_postgres_settings_from_env(monkeypatch):
    """PostgresSettings should pick up POSTGRES_DSN from the environment."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:p@dbhost:5432/knightgpt")
    from src.utils.config import PostgresSettings

    settings = PostgresSettings()
    assert settings.dsn == "postgresql://u:p@dbhost:5432/knightgpt"


@pytest.mark.unit
def test_settings_has_postgres_group():
    """Main Settings object should expose a postgres settings group."""
    from src.utils.config import Settings

    settings = Settings()
    assert settings.postgres.dsn


@pytest.mark.unit
def test_vllm_settings_api_key_defaults_to_empty(monkeypatch):
    """VLLMSettings.api_key should default to 'EMPTY' (self-hosted vLLM convention)."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    from src.utils.config import VLLMSettings

    settings = VLLMSettings(_env_file=None)
    assert settings.api_key == "EMPTY"


@pytest.mark.unit
def test_vllm_settings_api_key_from_env(monkeypatch):
    """VLLMSettings.api_key should pick up VLLM_API_KEY from the environment."""
    monkeypatch.setenv("VLLM_API_KEY", "sk-test-123")
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.api_key == "sk-test-123"


@pytest.mark.unit
def test_vllm_settings_embedding_dim_defaults_to_3584(monkeypatch):
    """VLLMSettings.embedding_dim should default to the current gte-Qwen2-7B dimension."""
    monkeypatch.delenv("VLLM_EMBEDDING_DIM", raising=False)
    from src.utils.config import VLLMSettings

    settings = VLLMSettings(_env_file=None)
    assert settings.embedding_dim == 3584


@pytest.mark.unit
def test_vllm_settings_embedding_dim_from_env(monkeypatch):
    """VLLMSettings.embedding_dim should pick up VLLM_EMBEDDING_DIM from the environment."""
    monkeypatch.setenv("VLLM_EMBEDDING_DIM", "4096")
    from src.utils.config import VLLMSettings

    settings = VLLMSettings()
    assert settings.embedding_dim == 4096
