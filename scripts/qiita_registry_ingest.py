#!/usr/bin/env python3
"""Qiita study registry ingestion (Stage 1).

Seeds the qiita_studies Postgres table with every study ID redbiom's
public index exposes, plus per-study sample counts. redbiom's backend
(http://qiita.ucsd.edu:7329) is public and requires no credentials.

This is Stage 1 only: study_id, sample_count, and which redbiom contexts
each study appears in. title/abstract/principal_investigator/funding/
metadata stay NULL here -- a separate, future Stage 2 backfills them
once direct Postgres access to Qiita's own database is available.
"""


def parse_study_counts(summarize_output: str) -> dict[int, int]:
    """Parse `redbiom summarize samples --category qiita_study_id`'s TSV
    stdout into {study_id: sample_count}.

    The real output is one "<study_id>\\t<count>" row per study, then a
    blank line, then a trailing "Total samples\\t<N>" summary row -- both
    the blank line and the summary row are skipped, not treated as data.
    """
    counts = {}
    for line in summarize_output.splitlines():
        line = line.strip()
        if not line or line.startswith("Total samples"):
            continue
        study_id_str, count_str = line.split("\t")
        counts[int(study_id_str)] = int(count_str)
    return counts


def merge_context_results(
    context_results: dict[str, dict[int, int]],
) -> dict[int, dict]:
    """Merge per-context {study_id: count} maps into a single registry.

    A study can appear in multiple redbiom contexts (different processing
    pipelines run on the same study's samples) -- this sums sample_count
    across all contexts a study appears in and records which contexts it
    appeared in, in first-seen order.
    """
    merged: dict[int, dict] = {}
    for context_name, study_counts in context_results.items():
        for study_id, count in study_counts.items():
            if study_id not in merged:
                merged[study_id] = {"sample_count": 0, "contexts": []}
            merged[study_id]["sample_count"] += count
            merged[study_id]["contexts"].append(context_name)
    return merged
