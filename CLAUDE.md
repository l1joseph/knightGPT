# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnightGPT is a Retrieval-Augmented Generation (RAG) system for microbiome research, deployed on the SDSC Cosmos cluster (AMD MI300A APUs). It ingests PDFs, builds a knowledge graph, and serves RAG-enhanced chat via FastAPI, backed by vLLM inference and embedding servers running in Singularity containers via SLURM.

**Current state (as of 2026-03-14):** 117 papers ingested, 6,179 chunks embedded (3584-dim), knowledge graph with 6,173 nodes and 37,286 edges. Embedding and inference servers confirmed working on Cosmos MI300A.

## Common Commands

```bash
# Environment
conda activate knightGPT                        # Python 3.10 env

# Tests
pytest tests/                                    # all tests (30 pass, 10 known failures)
pytest tests/ -v -m unit                         # unit tests only
pytest tests/ -v -m integration                  # integration tests

# Linting & formatting
black .
ruff check .

# Start vLLM servers (SLURM)
sbatch slurm/vllm_embedding_v1.slurm            # gte-Qwen2-7B embedding (port 8001)
sbatch slurm/vllm_inference_v1.slurm            # Qwen2.5-72B inference (port 8000)
squeue -u $USER                                  # check node assignments

# IMPORTANT: after sbatch, update .env with actual node hostnames:
#   VLLM_EMBEDDING_URL=http://<node>:8001/v1
#   VLLM_INFERENCE_URL=http://<node>:8000/v1

# Run API server (on login node, after vLLM servers are up)
python -m src.api.main --host 0.0.0.0 --port 8080

# SSH tunnel from local machine (match the login node hostname)
# ssh -L 8080:localhost:8080 l1joseph@cosmos02.cosmos.sdsc.edu

# Ingestion pipeline
python scripts/download_papers.py --input data/paper_lists/initial_papers.txt --run-pipeline
python scripts/populate_zotero.py --run-pipeline  # Zotero → OpenAlex → download → embed

# CLI chat
python -m src.cli --chunks /cosmos/vast/scratch/l1joseph/knightgpt/data/processed/chunks_with_emb.json
```

## Architecture

### Data Pipeline (5 stages)

```
PDFs → pdf_to_markdown (marker-pdf/PyMuPDF4LLM fallback)
    → SemanticChunker (tiktoken, 500 token max)
    → VLLMEmbedder (gte-Qwen2-7B, 3584-dim, port 8001)
    → KnowledgeGraphBuilder (cosine sim ≥ 0.7, max 10 neighbors)
    → GraphML + JSON artifacts → loaded at API startup
```

### Key Modules

- **`src/api/main.py`** — FastAPI app with lifespan handler. Exposes `/api/v1/chat` (streaming SSE), `/api/v1/search`, `/api/v1/ingest`, `/api/v1/agent/chat` (multi-agent), plus OpenAI-compatible `/v1/chat/completions` for Open WebUI.
- **`src/retrieval/retriever.py`** — `GraphRAGRetriever` (semantic similarity + graph traversal) and `RAGEngine` (query interface with streaming).
- **`src/embedding/embedder.py`** — `VLLMEmbedder` wraps OpenAI-compatible API with batch processing and tenacity retry.
- **`src/chunking/chunker.py`** — `SemanticChunker` does paragraph-level chunking with section awareness.
- **`src/graph/builder.py`** — `KnowledgeGraphBuilder` creates NetworkX graphs. `save_graph()` sanitizes None/dict/control chars for GraphML compatibility.
- **`src/agents/orchestrator.py`** — `AgentOrchestrator` runs 4-stage pipeline: Plan → Execute (tools + RAG) → Verify → Generate. Uses vLLM inference for planning/generation.
- **`src/tools/`** — Domain tools: `PubMedTool`, `OpenAlexTool`, `KEGGTool`, `QIIME2Tool`, `ZoteroTool`.
- **`src/ingestion/`** — PDF conversion, web scraping (Unpaywall/PMC), RSS feeds, briefing parser, Google Form webhook.
- **`src/utils/config.py`** — Pydantic `BaseSettings` with env_prefix for each subsystem. Reads `.env`.

### vLLM Servers (SLURM + Singularity)

| Server | Model | GPUs | Port | SLURM Script |
|--------|-------|------|------|-------------|
| Embedding | Alibaba-NLP/gte-Qwen2-7B-instruct | 1 | 8001 | `slurm/vllm_embedding_v1.slurm` |
| Inference | Qwen/Qwen2.5-72B-Instruct | 4 (TP) | 8000 | `slurm/vllm_inference_v1.slurm` |

- Container: `~/vllm_rocm_0.6.6.sif` (vLLM 0.6.6, ROCm 6.3.1, Python 3.12)
- ROCm arch: `gfx942` (MI300A)
- The embedding SLURM script patches `rocm_flash_attn.py` at runtime to handle `ENCODER_ONLY` attention type (vLLM bug — missing case for embedding models). It extracts the file from the container, adds the `ENCODER_ONLY` branch, and bind-mounts it over the original.

### Data Sources

Papers are ingested from three sources:
1. **`data/paper_lists/initial_papers.txt`** — 28 DOIs, Knight Lab core papers (QIIME2, UniFrac, EMP, etc.)
2. **`data/paper_lists/zotero_papers.txt`** — 70 DOIs from Zotero via OpenAlex discovery (5 collections)
3. **`data/paper_lists/mmc_papers.txt`** — 194 DOIs from `mmc_datasheet.tsv` (broad microbiome studies)

Processed data lives on scratch: `/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/`
- `chunks_with_emb.json` (578MB, 6179 chunks × 3584-dim)
- `graph.graphml` (17MB, 6173 nodes, 37286 edges)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, chunk/graph counts |
| `/api/v1/chat` | POST | RAG chat (supports `stream: true` for SSE) |
| `/api/v1/search` | POST | Semantic search with similarity scores |
| `/api/v1/agent/chat` | POST | Multi-agent: Plan → Tools → Verify → Generate |
| `/api/v1/ingest` | POST | PDF ingestion (background task) |
| `/api/v1/ingest/rss` | POST | RSS feed discovery + ingestion |
| `/api/v1/ingest/briefing` | POST | Parse briefing text for paper refs |
| `/v1/chat/completions` | POST | OpenAI-compatible (for Open WebUI) |
| `/v1/models` | GET | List models (OpenAI-compatible) |

### Deployment Stack (Docker Compose)

API (8080) + Open WebUI (3000) + Neo4j (7474/7687) + Watchtower + Cloudflare Tunnel

## Configuration

Settings via Pydantic `BaseSettings` in `src/utils/config.py`, loaded from `.env`. Key vars:

| Variable | Current Value | Purpose |
|----------|---------------|---------|
| `VLLM_EMBEDDING_URL` | `http://<node>:8001/v1` | Embedding server (update after sbatch) |
| `VLLM_INFERENCE_URL` | `http://<node>:8000/v1` | Inference server (update after sbatch) |
| `VLLM_INFERENCE_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Chat model |
| `GRAPH_SIMILARITY_THRESHOLD` | `0.7` | Edge creation threshold |
| `INGEST_PROCESSED_DIR` | `/cosmos/vast/scratch/.../processed` | Artifact storage |
| `ZOTERO_LIBRARY_ID` | `19943541` | Zotero user library |
| `HF_TOKEN` | set in .env | HuggingFace gated model access |

## Test Markers

Defined in `pytest.ini`: `unit`, `integration`, `api`, `slow`. Fixtures in `tests/conftest.py`.

**Known test failures (10):** Mock targets wrong in `test_embedding.py` (`src.embedding.OpenAI` doesn't exist), chunker edge cases in `test_chunking.py`, GraphML dict serialization in `test_integration.py`.

## Important Quirks

- **ENCODER_ONLY bug:** vLLM 0.6.6 `rocm_flash_attn.py` doesn't handle `ENCODER_ONLY` attention type. The embedding SLURM script patches this by bind-mounting a modified file. See `slurm/vllm_embedding_v1.slurm`.
- **GraphML sanitization:** `builder.py:save_graph()` strips None, dict, list, and XML-illegal characters from node/edge attributes before writing GraphML. Without this, `lxml` raises `ValueError`.
- **Node hostnames change:** Every SLURM job gets a different compute node. After `sbatch`, check `squeue -u $USER` and update `.env` with new hostnames.
- **Login node mismatch:** The API server must run on the same login node as your SSH tunnel target. Use `hostname` to verify.
- **Scratch path:** Cosmos uses `/cosmos/vast/scratch/$USER/`, not `/ddn_scratch/` (which is Barnacle2).
