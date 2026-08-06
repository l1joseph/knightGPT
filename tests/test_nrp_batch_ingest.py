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
