"""Unit tests for the NRP batch ingestion orchestrator."""

import pytest
from unittest.mock import MagicMock, patch


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


@pytest.mark.unit
def test_build_papers_dict_resolves_doi_per_chunk():
    from scripts.nrp_batch_ingest import build_papers_dict
    from src.chunking import Chunk

    chunks = [
        Chunk(
            id="c1",
            text="a",
            source_file="/data/markdown/10-1234_x.md",
            metadata={"title": "Paper X"},
        ),
        Chunk(
            id="c2",
            text="b",
            source_file="/data/markdown/10-1234_x.md",
            metadata={"title": "Paper X"},
        ),
    ]
    doi_lookup = {"10-1234_x": "10.1234/x"}

    papers = build_papers_dict(chunks, doi_lookup)

    assert papers["/data/markdown/10-1234_x.md"]["doi"] == "10.1234/x"
    assert papers["/data/markdown/10-1234_x.md"]["title"] == "Paper X"
    # Same source_file across multiple chunks must produce exactly one entry.
    assert len(papers) == 1


@pytest.mark.unit
def test_build_papers_dict_handles_missing_source_file():
    from scripts.nrp_batch_ingest import build_papers_dict
    from src.chunking import Chunk

    chunks = [Chunk(id="c1", text="a", source_file="", metadata={})]
    papers = build_papers_dict(chunks, doi_lookup={})

    assert papers == {}


@pytest.mark.unit
def test_all_embeddings_missing_true_when_every_chunk_lacks_embedding():
    from scripts.nrp_batch_ingest import all_embeddings_missing
    from src.chunking import Chunk

    chunks = [
        Chunk(id="c1", text="a", source_file="f.md", embedding=None),
        Chunk(id="c2", text="b", source_file="f.md", embedding=None),
    ]

    assert all_embeddings_missing(chunks) is True


@pytest.mark.unit
def test_all_embeddings_missing_false_when_some_chunks_embedded():
    from scripts.nrp_batch_ingest import all_embeddings_missing
    from src.chunking import Chunk

    chunks = [
        Chunk(id="c1", text="a", source_file="f.md", embedding=[0.1, 0.2]),
        Chunk(id="c2", text="b", source_file="f.md", embedding=None),
    ]

    assert all_embeddings_missing(chunks) is False


@pytest.mark.unit
def test_all_embeddings_missing_false_when_all_chunks_embedded():
    from scripts.nrp_batch_ingest import all_embeddings_missing
    from src.chunking import Chunk

    chunks = [
        Chunk(id="c1", text="a", source_file="f.md", embedding=[0.1, 0.2]),
        Chunk(id="c2", text="b", source_file="f.md", embedding=[0.3, 0.4]),
    ]

    assert all_embeddings_missing(chunks) is False


@pytest.mark.unit
def test_all_embeddings_missing_false_for_empty_list():
    from scripts.nrp_batch_ingest import all_embeddings_missing

    assert all_embeddings_missing([]) is False


@pytest.mark.unit
def test_wait_for_embedder_ready_returns_immediately_when_healthy():
    from scripts.nrp_batch_ingest import wait_for_embedder_ready

    embedder = MagicMock()
    embedder.check_health.return_value = True

    with patch("scripts.nrp_batch_ingest.time.sleep") as mock_sleep:
        wait_for_embedder_ready(embedder, timeout_s=60, poll_interval_s=10)

    embedder.check_health.assert_called_once()
    mock_sleep.assert_not_called()


@pytest.mark.unit
def test_wait_for_embedder_ready_polls_until_healthy():
    from scripts.nrp_batch_ingest import wait_for_embedder_ready

    embedder = MagicMock()
    embedder.check_health.side_effect = [False, False, True]

    with patch("scripts.nrp_batch_ingest.time.sleep") as mock_sleep:
        wait_for_embedder_ready(embedder, timeout_s=60, poll_interval_s=10)

    assert embedder.check_health.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.unit
def test_wait_for_embedder_ready_raises_on_timeout():
    from scripts.nrp_batch_ingest import wait_for_embedder_ready

    embedder = MagicMock()
    embedder.check_health.return_value = False

    # Simulate time passing without a real sleep: monotonic() is called once
    # up front for the deadline, then once per loop iteration.
    fake_times = iter([0, 1, 61, 121])
    with (
        patch("scripts.nrp_batch_ingest.time.sleep"),
        patch(
            "scripts.nrp_batch_ingest.time.monotonic",
            side_effect=lambda: next(fake_times),
        ),
        pytest.raises(RuntimeError, match="did not become ready"),
    ):
        wait_for_embedder_ready(embedder, timeout_s=60, poll_interval_s=10)
