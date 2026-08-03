"""Unit tests for ETL resolvers turning paper-source rows into DOI lists."""

import csv
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_resolve_longread_table_extracts_direct_doi_urls(tmp_path):
    """Rows with a bare doi.org Link should yield the DOI directly, no HTTP call."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJEB58634\t154\tONT PromethION\tSereika et al., Nat Microbiol 2025\tdoi.org/10.1038/s41564-025-02062-z\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == ["10.1038/s41564-025-02062-z"]


@pytest.mark.unit
def test_resolve_longread_table_extracts_doi_from_biorxiv_content_url(tmp_path):
    """Rows whose Link is a biorxiv.org/content/... URL (not doi.org) should still resolve."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJNA1404836\t42\tPacBio Revio\tShi et al., bioRxiv 2026 (preprint)\t"
        "biorxiv.org/content/10.64898/2026.01.21.700959v1\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == ["10.64898/2026.01.21.700959"]


@pytest.mark.unit
def test_resolve_longread_table_skips_no_publication_rows(tmp_path):
    """Rows flagged 'no publication' have no paper and must be skipped, not resolved."""
    from scripts.etl_sheet_to_dois import resolve_longread_table

    tsv = tmp_path / "longread.tsv"
    tsv.write_text(
        "accession\tsamples\tplatform\tcitation\tlink\n"
        "PRJNA689363\t24\tPacBio SMRT\tno publication\tncbi.nlm.nih.gov/bioproject/PRJNA689363\n"
    )
    dois = resolve_longread_table(tsv, session=MagicMock())
    assert dois == []


@pytest.mark.unit
def test_resolve_mmc_sheet_reads_doi_column(tmp_path):
    """MMC sheet has a direct DOI column; rows should dedupe by DOI."""
    from scripts.etl_sheet_to_dois import resolve_mmc_sheet

    csv_path = tmp_path / "mmc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyTitle", "DOI", "DOI Url"])
        writer.writeheader()
        writer.writerow(
            {"StudyTitle": "Study A", "DOI": "10.1/aaa", "DOI Url": "10.1/aaa"}
        )
        writer.writerow(
            {"StudyTitle": "Study A dup", "DOI": "10.1/aaa", "DOI Url": "10.1/aaa"}
        )
        writer.writerow(
            {"StudyTitle": "Study B", "DOI": "10.1/bbb", "DOI Url": "10.1/bbb"}
        )

    dois = resolve_mmc_sheet(csv_path)
    assert dois == ["10.1/aaa", "10.1/bbb"]


@pytest.mark.unit
def test_resolve_qiita_tracker_resolves_pmid_via_openalex(tmp_path):
    """Cancer Qiita tracker rows have no DOI column; PMID in article_link resolves via OpenAlex."""
    from scripts.etl_sheet_to_dois import resolve_qiita_tracker

    csv_path = tmp_path / "qiita.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qiita_id", "article_link"])
        writer.writeheader()
        writer.writerow(
            {
                "qiita_id": "909",
                "article_link": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            }
        )
        writer.writerow(
            {
                "qiita_id": "909",
                "article_link": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            }
        )  # dup

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"doi": "https://doi.org/10.1/ccc"}]}
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response

    dois = resolve_qiita_tracker(csv_path, session=mock_session)
    assert dois == ["10.1/ccc"]
    assert mock_session.get.call_count == 1  # deduped by qiita_id before resolving


@pytest.mark.unit
def test_resolve_global_gut_sheet_extracts_doi_from_publisher_url(tmp_path):
    """Global Human Gut Microbiome sheet's Link to paper URLs encode a DOI in the path."""
    from scripts.etl_sheet_to_dois import resolve_global_gut_sheet

    csv_path = tmp_path / "gut.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Study", "Title", "Link to paper"])
        writer.writeheader()
        writer.writerow(
            {
                "Study": "S1",
                "Title": "T1",
                "Link to paper": "https://journals.asm.org/doi/full/10.1128/mbio.00519-19",
            }
        )

    dois = resolve_global_gut_sheet(csv_path, session=MagicMock())
    assert dois == ["10.1128/mbio.00519-19"]
