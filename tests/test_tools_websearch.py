"""Unit tests for WebSearchTool (src/tools/websearch.py).

The SearXNG HTTP call is mocked via the tool's own requests.Session
(patched with unittest.mock, matching this project's established
tool-test convention of mocking the dependency the tool talks to --
see tests/test_tools_search_corpus.py) -- no real network call is ever
made.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_execute_returns_mapped_results():
    from src.tools.websearch import WebSearchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "QIIME2 docs",
                "url": "https://docs.qiime2.org",
                "content": "QIIME2 is a microbiome analysis platform.",
            },
            {
                "title": "UniFrac paper",
                "url": "https://example.com/unifrac",
                "content": "UniFrac is a phylogenetic beta diversity metric.",
            },
        ]
    }
    mock_response.raise_for_status.return_value = None

    tool = WebSearchTool(base_url="http://searxng:8080")
    with patch.object(tool.session, "get", return_value=mock_response) as mock_get:
        result = tool.execute("qiime2")

    assert result.success is True
    assert result.data == [
        {
            "title": "QIIME2 docs",
            "url": "https://docs.qiime2.org",
            "snippet": "QIIME2 is a microbiome analysis platform.",
        },
        {
            "title": "UniFrac paper",
            "url": "https://example.com/unifrac",
            "snippet": "UniFrac is a phylogenetic beta diversity metric.",
        },
    ]
    assert result.metadata == {"query": "qiime2", "total_results": 2}
    mock_get.assert_called_once_with(
        "http://searxng:8080/search",
        params={"q": "qiime2", "format": "json"},
        timeout=15,
    )


@pytest.mark.unit
def test_execute_caps_results_to_count():
    from src.tools.websearch import WebSearchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": f"r{i}", "url": f"https://x/{i}", "content": f"c{i}"}
            for i in range(15)
        ]
    }
    mock_response.raise_for_status.return_value = None

    tool = WebSearchTool(base_url="http://searxng:8080")
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("query")

    assert len(result.data) == 10


@pytest.mark.unit
def test_execute_respects_custom_count():
    from src.tools.websearch import WebSearchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": f"r{i}", "url": f"https://x/{i}", "content": f"c{i}"}
            for i in range(15)
        ]
    }
    mock_response.raise_for_status.return_value = None

    tool = WebSearchTool(base_url="http://searxng:8080")
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("query", count=3)

    assert len(result.data) == 3


@pytest.mark.unit
def test_execute_handles_request_exception():
    from src.tools.websearch import WebSearchTool

    tool = WebSearchTool(base_url="http://searxng:8080")
    with patch.object(
        tool.session, "get", side_effect=RuntimeError("connection refused")
    ):
        result = tool.execute("query")

    assert result.success is False
    assert "connection refused" in result.error


@pytest.mark.unit
def test_execute_handles_empty_results():
    from src.tools.websearch import WebSearchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status.return_value = None

    tool = WebSearchTool(base_url="http://searxng:8080")
    with patch.object(tool.session, "get", return_value=mock_response):
        result = tool.execute("query")

    assert result.success is True
    assert result.data == []


@pytest.mark.unit
def test_init_defaults_base_url_from_settings():
    """With no explicit base_url, WebSearchTool should fall back to the
    module-level settings.searxng.url (see tests/test_config.py for
    SearXNGSettings' own env var coverage)."""
    from src.tools.websearch import WebSearchTool, settings

    tool = WebSearchTool()
    assert tool.base_url == settings.searxng.url


@pytest.mark.unit
def test_init_explicit_base_url_overrides_settings():
    from src.tools.websearch import WebSearchTool

    tool = WebSearchTool(base_url="http://override:1234")
    assert tool.base_url == "http://override:1234"


@pytest.mark.unit
def test_schema_declares_query_and_count():
    from src.tools.websearch import WebSearchTool

    schema = WebSearchTool(base_url="http://searxng:8080").schema
    assert schema["parameters"]["required"] == ["query"]
    assert "query" in schema["parameters"]["properties"]
    assert "count" in schema["parameters"]["properties"]


@pytest.mark.unit
def test_tool_name_is_web_search():
    from src.tools.websearch import WebSearchTool

    assert WebSearchTool(base_url="http://searxng:8080").name == "web_search"
