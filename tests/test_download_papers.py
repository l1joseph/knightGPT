"""Unit tests for scripts/download_papers.py's PMC full-text XML path.

NCBI retired the old PMC Open Access Web Service (oa.fcgi) in August
2026, and PMC's article "/pdf/" URLs now redirect through a JS-rendered
viewer that serves HTML instead of a raw PDF stream -- these tests cover
the E-utilities-based replacement (resolve_doi_pmc_id + fetch_pmc_fulltext
+ extract_pmc_body_text), which fetches and parses PMC's JATS full-text
XML directly instead of scraping a PDF URL.
"""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest

JATS_WITH_BODY = """<?xml version="1.0"?>
<pmc-articleset>
<article>
<front>
<article-meta>
<title-group><article-title>Gut Microbiome Diversity</article-title></title-group>
</article-meta>
</front>
<body>
<sec>
<title>Introduction</title>
<p>The gut microbiome is complex.</p>
</sec>
<sec>
<title>Methods</title>
<p>We used 16S rRNA sequencing.</p>
<p>Samples were collected weekly.</p>
</sec>
</body>
</article>
</pmc-articleset>
"""

JATS_NO_BODY = """<?xml version="1.0"?>
<pmc-articleset>
<article>
<!--The publisher of this article does not allow downloading of the full text in XML form.-->
<front>
<article-meta>
<title-group><article-title>Restricted Article</article-title></title-group>
</article-meta>
</front>
</article>
</pmc-articleset>
"""


@pytest.mark.unit
def test_extract_pmc_body_text_renders_sections_and_paragraphs():
    from scripts.download_papers import extract_pmc_body_text

    root = ET.fromstring(JATS_WITH_BODY)
    text = extract_pmc_body_text(root)

    assert "## Introduction" in text
    assert "The gut microbiome is complex." in text
    assert "## Methods" in text
    assert "We used 16S rRNA sequencing." in text
    assert "Samples were collected weekly." in text
    # Section order preserved.
    assert text.index("Introduction") < text.index("Methods")


@pytest.mark.unit
def test_extract_pmc_body_text_returns_empty_string_when_no_body():
    from scripts.download_papers import extract_pmc_body_text

    root = ET.fromstring(JATS_NO_BODY)
    assert extract_pmc_body_text(root) == ""


@pytest.mark.unit
def test_fetch_pmc_fulltext_success():
    from scripts.download_papers import fetch_pmc_fulltext

    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200, content=JATS_WITH_BODY.encode("utf-8")
    )

    result = fetch_pmc_fulltext("PMC1317376", session)

    assert result is not None
    assert result["title"] == "Gut Microbiome Diversity"
    assert "Introduction" in result["text"]
    # Numeric id (no "PMC" prefix) is what efetch actually accepts.
    call_kwargs = session.get.call_args.kwargs
    assert call_kwargs["params"]["id"] == "1317376"
    assert call_kwargs["params"]["db"] == "pmc"


@pytest.mark.unit
def test_fetch_pmc_fulltext_returns_none_when_publisher_restricts_body():
    from scripts.download_papers import fetch_pmc_fulltext

    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200, content=JATS_NO_BODY.encode("utf-8")
    )

    assert fetch_pmc_fulltext("PMC9999999", session) is None


@pytest.mark.unit
def test_fetch_pmc_fulltext_returns_none_on_non_200():
    from scripts.download_papers import fetch_pmc_fulltext

    session = MagicMock()
    session.get.return_value = MagicMock(status_code=429)

    assert fetch_pmc_fulltext("PMC1317376", session) is None


@pytest.mark.unit
def test_fetch_pmc_fulltext_returns_none_on_request_exception():
    from scripts.download_papers import fetch_pmc_fulltext

    session = MagicMock()
    session.get.side_effect = ConnectionError("boom")

    assert fetch_pmc_fulltext("PMC1317376", session) is None


@pytest.mark.unit
def test_resolve_doi_pmc_id_success():
    from scripts.download_papers import resolve_doi_pmc_id

    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"records": [{"pmcid": "PMC1317376"}]},
    )

    assert resolve_doi_pmc_id("10.1128/AEM.71.12.8228-8235.2005", session) == "PMC1317376"


@pytest.mark.unit
def test_resolve_doi_pmc_id_returns_none_when_not_in_pmc():
    from scripts.download_papers import resolve_doi_pmc_id

    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, json=lambda: {"records": [{}]})

    assert resolve_doi_pmc_id("10.1234/not-in-pmc", session) is None
