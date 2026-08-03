"""Test the FastAPI lifespan() wiring invariant the whole DuckDB
integration depends on: lifespan() must construct exactly one DuckDBStore
and pass that SAME instance to HybridRetriever(duckdb_store=...) -- it is
also the instance every insert_chunks(..., _duckdb_store) call in
/api/v1/ingest uses via the module-level _duckdb_store global. If lifespan
ever constructed two different DuckDBStore instances (or passed a
different one to the retriever than the one stored in the global), a chunk
inserted via /api/v1/ingest would silently be invisible to search --
HybridRetriever would be querying a different DuckDB connection/file
handle than the one ingestion just wrote to. Nothing before this test
exercised lifespan() directly."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_passes_same_duckdb_store_instance_to_hybrid_retriever():
    import src.api.main as main_module

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()

    mock_store_instance = MagicMock()

    mock_retriever_instance = MagicMock()

    with patch(
        "src.api.main.get_pg_pool", new=AsyncMock(return_value=mock_pool)
    ), patch(
        "src.api.main.DuckDBStore", return_value=mock_store_instance
    ) as MockDuckDBStore, patch(
        "src.api.main.HybridRetriever", return_value=mock_retriever_instance
    ) as MockHybridRetriever:
        async with main_module.lifespan(main_module.app):
            # Exactly one DuckDBStore constructed ...
            MockDuckDBStore.assert_called_once()
            # ... and that exact instance is what HybridRetriever received.
            MockHybridRetriever.assert_called_once_with(
                duckdb_store=mock_store_instance
            )
            # The module-level global (read by /api/v1/ingest's
            # insert_chunks(_pool, ..., _duckdb_store) call) must be the
            # same instance too.
            assert main_module._duckdb_store is mock_store_instance
            assert main_module._retriever is mock_retriever_instance

        # Shutdown must close both the retriever and the duckdb store.
        mock_retriever_instance.close.assert_called_once()
        mock_store_instance.close.assert_called_once()
        mock_pool.close.assert_awaited_once()
