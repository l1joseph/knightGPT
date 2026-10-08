"""Unit tests for WebFetchTool (src/tools/webfetch.py).

The HTTP call is mocked via the tool's own requests.Session (patched
with unittest.mock, matching this project's established tool-test
convention -- see tests/test_tools_websearch.py and
tests/test_tools_search_corpus.py) -- no real network call is ever made.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_execute_extracts_plain_text_from_html():
    from src.tools.webfetch import WebFetchTool

    mock_response = MagicMock()
    mock_response.content = b"<html><body><p>Some   content   here.</p></body></html>"
    mock_response.raise_for_status.return_value = None

    tool = WebFetchTool()
    with patch.object(tool.session, "get", return_value=mock_response) as mock_get:
        result = tool.execute("https://example.com/page")

    assert result.success is True
    assert result.data == "Some content here."
    mock_get.assert_called_once_with("https://example.com/page", timeout=15)


@pytest.mark.unit
def test_execute_truncates_to_max_chars():
    from src.tools.webfetch import WebFetchTool

    long_text = "word " * 5000
    mock_response = MagicMock()
    mock_response.content = f"<html><body><p>{long_text}</p></body></html>".encode()
    mock_response.raise_for_status.return_value = None

    tool = WebFetchTool()
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("https://example.com/page", max_chars=100)

    assert result.success is True
    assert len(result.data) <= 100
    assert result.metadata["truncated"] is True


@pytest.mark.unit
def test_execute_default_max_chars_is_8000():
    from src.tools.webfetch import WebFetchTool

    long_text = "word " * 5000
    mock_response = MagicMock()
    mock_response.content = f"<html><body><p>{long_text}</p></body></html>".encode()
    mock_response.raise_for_status.return_value = None

    tool = WebFetchTool()
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("https://example.com/page")

    assert len(result.data) <= 8000


@pytest.mark.unit
def test_execute_handles_request_exception():
    from src.tools.webfetch import WebFetchTool

    tool = WebFetchTool()
    with patch.object(tool.session, "get", side_effect=RuntimeError("timed out")):
        result = tool.execute("https://example.com/page")

    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.unit
def test_execute_handles_malformed_html_gracefully():
    from src.tools.webfetch import WebFetchTool

    mock_response = MagicMock()
    mock_response.content = b"not even close to html <<<>>>"
    mock_response.raise_for_status.return_value = None

    tool = WebFetchTool()
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("https://example.com/page")

    assert result.success is True
    assert isinstance(result.data, str)


@pytest.mark.unit
def test_schema_declares_url_and_max_chars():
    from src.tools.webfetch import WebFetchTool

    schema = WebFetchTool().schema
    assert schema["parameters"]["required"] == ["url"]
    assert "url" in schema["parameters"]["properties"]
    assert "max_chars" in schema["parameters"]["properties"]


@pytest.mark.unit
def test_tool_name_is_web_fetch():
    from src.tools.webfetch import WebFetchTool

    assert WebFetchTool().name == "web_fetch"
