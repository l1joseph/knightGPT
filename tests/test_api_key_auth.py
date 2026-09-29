"""Unit tests for verify_api_key(), the shared-secret bearer-token
dependency gating /api/v1/chat, /api/v1/agent/chat, /api/v1/search, and
/v1/chat/completions. Tested directly as a pure function of (authorization
header, settings.api.api_key) rather than through a live TestClient(app) --
app startup requires real Postgres/DuckDB/vLLM connections (lifespan hard-
fails without them), which this unit test has no business standing up."""

from fastapi import HTTPException
import pytest

import src.api.main as main_module


@pytest.fixture
def restore_api_key():
    """verify_api_key() reads the module-level settings singleton, so tests
    mutate it directly -- restore the original value after each test rather
    than leaking state into whichever test runs next."""
    original = main_module.settings.api.api_key
    yield
    main_module.settings.api.api_key = original


@pytest.mark.unit
def test_verify_api_key_noop_when_unset(restore_api_key):
    """No API_KEY configured -- matches the webhook's existing behavior of
    staying open by default for local/dev/proof-test deploys."""
    main_module.settings.api.api_key = None
    main_module.verify_api_key(authorization=None)
    main_module.verify_api_key(authorization="Bearer anything")


@pytest.mark.unit
def test_verify_api_key_accepts_correct_bearer_token(restore_api_key):
    main_module.settings.api.api_key = "secret123"
    main_module.verify_api_key(authorization="Bearer secret123")


@pytest.mark.unit
def test_verify_api_key_rejects_missing_header(restore_api_key):
    main_module.settings.api.api_key = "secret123"
    with pytest.raises(HTTPException) as exc_info:
        main_module.verify_api_key(authorization=None)
    assert exc_info.value.status_code == 401


@pytest.mark.unit
def test_verify_api_key_rejects_wrong_token(restore_api_key):
    main_module.settings.api.api_key = "secret123"
    with pytest.raises(HTTPException) as exc_info:
        main_module.verify_api_key(authorization="Bearer wrong")
    assert exc_info.value.status_code == 401


@pytest.mark.unit
def test_verify_api_key_rejects_missing_bearer_prefix(restore_api_key):
    """A raw token without the "Bearer " prefix must not match -- guards
    against a client sending the key as a bare header value."""
    main_module.settings.api.api_key = "secret123"
    with pytest.raises(HTTPException) as exc_info:
        main_module.verify_api_key(authorization="secret123")
    assert exc_info.value.status_code == 401
