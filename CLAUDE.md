# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnightGPT is a Retrieval-Augmented Generation (RAG) system for microbiome research, deployed on the SDSC Cosmos cluster (AMD MI300A APUs). It ingests PDFs, builds a knowledge graph, and serves RAG-enhanced chat via FastAPI + Open WebUI, backed by vLLM inference and embedding servers.

## Common Commands

```bash
# Tests
pytest tests/                                    # all tests
pytest tests/ -v -m unit                         # unit tests only
pytest tests/ -v -m integration                  # integration tests
pytest tests/ -v -m api                          # API tests
pytest tests/ --cov=src --cov-report=html        # coverage report
pytest tests/test_chunking.py::test_simple_text  # single test

# Linting & formatting
black .
ruff check .
mypy --ignore-missing-imports src/

# Run services
python -m src.api.main --host 0.0.0.0 --port 8080   # API server
python -m src.cli --chunks data/processed/chunks_with_emb.json  # CLI chat

# Ingestion pipeline
python scripts/ingest_pipeline.py

# Health & monitoring
python scripts/healthcheck.py
python scripts/bug_detection.py

# Environment setup
conda env create -f environment.mi300a.yml   # creates 'knightgpt' env (Python 3.10)
conda activate knightgpt
pip install -r requirements.txt              # if not using conda
```

## Architecture

### Data Pipeline (5 stages)

```
PDFs → pdf_to_markdown (marker-pdf) → SemanticChunker (tiktoken, 500 token max)
    → VLLMEmbedder (gte-Qwen2-7B, port 8001) → KnowledgeGraphBuilder (cosine sim ≥ 0.7)
    → GraphML + JSON artifacts → loaded at API startup
```

### Key Modules

- **`src/api/main.py`** — FastAPI app with lifespan handler. Exposes `/api/v1/chat`, `/api/v1/search`, `/api/v1/ingest`, plus OpenAI-compatible `/v1/chat/completions` for Open WebUI.
- **`src/retrieval/retriever.py`** — `GraphRAGRetriever` (semantic similarity + graph traversal) and `RAGEngine` (high-level query interface with streaming).
- **`src/embedding/embedder.py`** — `VLLMEmbedder` wraps OpenAI-compatible API with batch processing and tenacity retry (3 attempts, exponential backoff).
- **`src/chunking/chunker.py`** — `SemanticChunker` does paragraph-level chunking with section awareness.
- **`src/graph/builder.py`** — `KnowledgeGraphBuilder` creates NetworkX graphs with cosine similarity edges.
- **`src/storage/storage.py`** — Optional `Neo4jStorage` layer with full-text indexing.
- **`src/utils/config.py`** — Pydantic `BaseSettings` aggregating all config: `VLLMSettings`, `GraphSettings`, `ChunkingSettings`, `APISettings`, etc. Reads from `.env`.
- **`src/utils/logging.py`** — Loguru-based centralized logging (100MB rotation, 1-week retention).
- **`scripts/ingest_pipeline.py`** — Orchestrates the full 5-stage pipeline.

### vLLM Servers (SLURM)

| Server | Model | GPUs | Port | SLURM Script |
|--------|-------|------|------|-------------|
| Embedding | Alibaba-NLP/gte-Qwen2-7B-instruct | 1 | 8001 | `slurm/vllm_embedding.slurm` |
| Inference | meta-llama/Llama-3.3-70B-Instruct | 4 (tensor parallel) | 8000 | `slurm/vllm_inference.slurm` |

ROCm flags: `PYTORCH_ROCM_ARCH="gfx942"`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MHA=1`

### Deployment Stack (Docker Compose)

API (8080) + Open WebUI (3000) + Neo4j (7474/7687) + Watchtower + Cloudflare Tunnel

## Test Markers

Defined in `pytest.ini`: `unit`, `integration`, `api`, `slow`. Fixtures in `tests/conftest.py` provide `sample_chunks` (768-dim embeddings), `sample_markdown_file`, and `temp_dir`.

## Configuration

Settings are managed via Pydantic `BaseSettings` in `src/utils/config.py`, loaded from environment variables or `.env` file. See `.env.example` for all 51 configurable variables. Key defaults:

- Embedding URL: `http://localhost:8001/v1`
- Inference URL: `http://localhost:8000/v1`
- Similarity threshold: 0.7
- Max chunk tokens: 500
- API port: 8080
