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
