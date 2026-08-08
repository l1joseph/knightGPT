"""Unit tests for the Qiita study registry ingestion script."""

import pytest


@pytest.mark.unit
def test_parse_study_counts_multiple_studies():
    from scripts.qiita_registry_ingest import parse_study_counts

    output = "10317\t31556\n12949\t17277\n\nTotal samples\t48833\n"
    result = parse_study_counts(output)

    assert result == {10317: 31556, 12949: 17277}


@pytest.mark.unit
def test_parse_study_counts_single_study():
    from scripts.qiita_registry_ingest import parse_study_counts

    output = "10333\t1\n\nTotal samples\t1\n"
    result = parse_study_counts(output)

    assert result == {10333: 1}


@pytest.mark.unit
def test_parse_study_counts_empty_context():
    from scripts.qiita_registry_ingest import parse_study_counts

    assert parse_study_counts("") == {}


@pytest.mark.unit
def test_parse_study_counts_ignores_total_line_only():
    from scripts.qiita_registry_ingest import parse_study_counts

    # "Total samples" itself is never a valid numeric study_id, so a
    # naive int() cast would raise -- confirms the line is skipped, not
    # silently mis-parsed as a study.
    output = "10317\t5\n\nTotal samples\t5\n"
    result = parse_study_counts(output)

    assert "Total samples" not in result
    assert result == {10317: 5}


@pytest.mark.unit
def test_merge_context_results_sums_across_contexts():
    from scripts.qiita_registry_ingest import merge_context_results

    context_results = {
        "ctxA": {10317: 100, 12949: 50},
        "ctxB": {10317: 20, 99999: 5},
    }

    result = merge_context_results(context_results)

    assert result == {
        10317: {"sample_count": 120, "contexts": ["ctxA", "ctxB"]},
        12949: {"sample_count": 50, "contexts": ["ctxA"]},
        99999: {"sample_count": 5, "contexts": ["ctxB"]},
    }


@pytest.mark.unit
def test_merge_context_results_empty_input():
    from scripts.qiita_registry_ingest import merge_context_results

    assert merge_context_results({}) == {}
