"""Unit tests for the NRP batch ingestion orchestrator."""

import pytest


@pytest.mark.unit
def test_partition_into_batches_even_split():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = list(range(10))
    batches = partition_into_batches(items, batch_size=5)

    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


@pytest.mark.unit
def test_partition_into_batches_uneven_remainder():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = list(range(7))
    batches = partition_into_batches(items, batch_size=3)

    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


@pytest.mark.unit
def test_partition_into_batches_empty_input():
    from scripts.nrp_batch_ingest import partition_into_batches

    assert partition_into_batches([], batch_size=5) == []


@pytest.mark.unit
def test_partition_into_batches_batch_size_larger_than_input():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = [1, 2, 3]
    assert partition_into_batches(items, batch_size=50) == [[1, 2, 3]]
