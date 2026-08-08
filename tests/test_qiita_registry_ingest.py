"""Unit tests for the Qiita study registry ingestion script."""

import pytest


@pytest.mark.unit
def test_parse_study_counts_from_sample_ids_multiple_studies():
    from scripts.qiita_registry_ingest import parse_study_counts_from_sample_ids

    fetch_output = "10317.sample1\n10317.sample2\n12949.sample1\n"
    result = parse_study_counts_from_sample_ids(fetch_output)

    assert result == {10317: 2, 12949: 1}


@pytest.mark.unit
def test_parse_study_counts_from_sample_ids_empty_input():
    from scripts.qiita_registry_ingest import parse_study_counts_from_sample_ids

    assert parse_study_counts_from_sample_ids("") == {}


@pytest.mark.unit
def test_parse_study_counts_from_sample_ids_skips_malformed_prefix():
    from scripts.qiita_registry_ingest import parse_study_counts_from_sample_ids

    fetch_output = "10317.sample1\nnot_a_number.weird\n10317.sample2\n"
    result = parse_study_counts_from_sample_ids(fetch_output)

    assert result == {10317: 2}


@pytest.mark.unit
def test_parse_study_counts_from_sample_ids_sample_name_with_dots():
    from scripts.qiita_registry_ingest import parse_study_counts_from_sample_ids

    # sample names can contain further dots (e.g. diabimmune.sample.id.X) --
    # split(".", 1) must take only the FIRST dot as the boundary.
    fetch_output = "11884.diabimmune.sample.id.3105832\n"
    result = parse_study_counts_from_sample_ids(fetch_output)

    assert result == {11884: 1}


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
