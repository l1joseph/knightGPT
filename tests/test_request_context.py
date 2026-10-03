"""Unit tests for RequestContext construction from trusted proxy headers
and the OpenAI-style `files` array -- see
docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
Pure-function tests, no FastAPI Request object needed: headers is any
Mapping[str, str], body is a plain dict."""

import pytest

from src.api.request_context import RequestContext, build_request_context


@pytest.mark.unit
def test_defaults_are_no_identity_no_collection():
    ctx = RequestContext()
    assert ctx.email is None
    assert ctx.is_admin is False
    assert ctx.collection_id is None


@pytest.mark.unit
def test_prefers_openwebui_header_over_auth_request_header():
    headers = {
        "X-OpenWebUI-User-Email": "alice@example.com",
        "X-Auth-Request-Email": "bob@example.com",
    }
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "alice@example.com"


@pytest.mark.unit
def test_falls_back_to_auth_request_header_when_openwebui_header_absent():
    headers = {"X-Auth-Request-Email": "bob@example.com"}
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "bob@example.com"


@pytest.mark.unit
def test_no_headers_present_gives_none_email_and_not_admin():
    ctx = build_request_context({}, {}, admin_emails={"alice@example.com"})
    assert ctx.email is None
    assert ctx.is_admin is False


@pytest.mark.unit
def test_header_lookup_is_case_insensitive():
    """HTTP headers are case-insensitive; a plain dict test double must not
    assume a specific case."""
    headers = {"x-openwebui-user-email": "alice@example.com"}
    ctx = build_request_context(headers, {}, admin_emails=set())
    assert ctx.email == "alice@example.com"


@pytest.mark.unit
def test_admin_allowlist_match_is_case_insensitive():
    headers = {"X-OpenWebUI-User-Email": "Alice@Example.com"}
    ctx = build_request_context(headers, {}, admin_emails={"alice@example.com"})
    assert ctx.is_admin is True


@pytest.mark.unit
def test_non_admin_email_not_in_allowlist():
    headers = {"X-OpenWebUI-User-Email": "mallory@example.com"}
    ctx = build_request_context(headers, {}, admin_emails={"alice@example.com"})
    assert ctx.is_admin is False


@pytest.mark.unit
def test_collection_id_from_first_collection_entry_in_files():
    body = {
        "files": [
            {"type": "collection", "id": "know-123"},
            {"type": "collection", "id": "know-456"},
        ]
    }
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_collection_id_skips_non_collection_file_entries():
    body = {
        "files": [
            {"type": "file", "id": "upload-1"},
            {"type": "collection", "id": "know-123"},
        ]
    }
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_collection_id_none_when_no_files_array():
    ctx = build_request_context({}, {}, admin_emails=set())
    assert ctx.collection_id is None


@pytest.mark.unit
def test_collection_id_none_when_files_array_has_no_collection_entry():
    body = {"files": [{"type": "file", "id": "upload-1"}]}
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id is None


@pytest.mark.unit
def test_collection_id_falls_back_to_top_level_body_key():
    """OWUI's own `files`/`metadata.files` payload gets dropped by Open
    WebUI's flattening step before it reaches us (see the Filter function
    in deploy/openwebui-functions/), so a plain top-level `collection_id`
    key -- injected by that Filter -- is the last-resort fallback source."""
    body = {"collection_id": "test-a"}
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "test-a"


@pytest.mark.unit
def test_collection_id_files_array_takes_priority_over_top_level_key():
    body = {
        "files": [{"type": "collection", "id": "know-123"}],
        "collection_id": "test-a",
    }
    ctx = build_request_context({}, body, admin_emails=set())
    assert ctx.collection_id == "know-123"


@pytest.mark.unit
def test_collection_id_none_when_neither_files_nor_top_level_key_present():
    ctx = build_request_context({}, {}, admin_emails=set())
    assert ctx.collection_id is None
