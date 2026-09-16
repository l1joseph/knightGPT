# KnightGPT

A Retrieval-Augmented Generation (RAG) system for microbiome research, with knowledge graph-based retrieval, a real tool-calling agent loop, and domain-specific tools. Runs on SDSC Cosmos (AMD MI300A) for ingestion, and deploys as a Docker Compose stack (API + Open WebUI + Postgres) for day-to-day use.

## Architecture

```
Paper Sources (DOI lists, Zotero, RSS, mmc_datasheet.tsv, Qiita/redbiom)
    │
    ▼
Download & Convert (Unpaywall/PMC → PDF → marker-pdf/PyMuPDF4LLM → Markdown)
    │
    ▼
Semantic Chunking (paragraph-level, 500 token max, section-aware)
    │
    ▼
Embedding (NRP-hosted qwen3-embedding, 4096-dim vectors)
    │
    ▼
Storage: Postgres (pgGraph extension) + embedded DuckDB (vector search)
├── papers / chunks / chunk_edges   -- pgGraph-registered, graph traversal via SQL
├── qiita_studies (+ publications/paper links) -- Qiita cross-reference project
└── embeddings.duckdb               -- nearest-neighbor vector search
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  FastAPI Server (port 8080)                                │
│  ├── /api/v1/chat          RAG chat (streaming SSE)        │
│  ├── /api/v1/search        Semantic search + citations     │
│  ├── /api/v1/agent/chat    Tool-calling agent loop          │
│  ├── /api/v1/ingest        PDF/RSS/briefing ingestion       │
│  └── /v1/chat/completions  OpenAI-compatible (Open WebUI)   │
└────────────────────────────────────────────────────────────┘
    │
    ▼
NRP-hosted LLM endpoint (https://ellm.nrp-nautilus.io/v1)
├── Embedding: qwen3-embedding (4096-dim)
└── Inference: qwen3 (real OpenAI-style function-calling)
```

Self-hosted vLLM on Cosmos MI300A (SLURM + Singularity) remains available as an alternative to the NRP endpoint -- see [Self-Hosted vLLM (Cosmos MI300A)](#self-hosted-vllm-cosmos-mi300a-alternative) below.

## Quick Start (Docker Compose deployment)

This is the primary way to run KnightGPT day-to-day: API + Open WebUI + Postgres as a single stack, talking to NRP's hosted LLM endpoint.

### 1. Configure

```bash
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT
cp .env.example .env
```

Edit `.env`:
- `VLLM_API_KEY` -- get a token at https://ellm.nrp-nautilus.io/llmtoken
- `VLLM_EMBEDDING_DIM=4096` (must match `qwen3-embedding`'s real output size, or `DuckDBStore`'s dimension guard raises at startup)
- `POSTGRES_PASSWORD` -- pick a real one for anything beyond local testing
- `WEBUI_SECRET_KEY` -- pick a real one; `WEBUI_AUTH=true` is on by default

`docker/docker-compose.yaml`'s `postgres` service also bind-mounts a data directory and a restore dump at host-specific absolute paths -- override those for your host (see `k8s/nrp/README.md` for the current known-good paths and the S3 backup location).

### 2. Bring up the stack

```bash
cd docker
docker compose up -d --build
docker compose ps        # wait for postgres and api to report healthy
```

On first boot with an empty `pgdata` volume, Postgres restores from the mounted backup dump, truncates `papers`/`chunks`/`chunk_edges` (they were embedded with the old self-hosted model and are incompatible with `qwen3-embedding`'s dimension), and re-registers pgGraph against the restored tables -- see `docker/postgres/init/02-restore-backup.sh`.

### 3. Use it

- Open WebUI: `http://<host>:3000` (create an account on first visit; `WEBUI_AUTH=true`)
- API directly: `http://<host>:8080/health`, `http://<host>:8080/docs`
- For a public subdomain, uncomment/configure the `cloudflared` service (`--profile production`) with a real `CLOUDFLARE_TUNNEL_TOKEN`.

### 4. Ingest papers

Ingestion runs outside the Compose stack, against the same NRP endpoint and Postgres instance:

```bash
conda activate knightGPT
python scripts/nrp_batch_ingest.py --input data/paper_lists/initial_papers.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Embedding/inference/Postgres health, chunk/graph counts |
| `/api/v1/chat` | POST | RAG chat with citations (supports `stream: true`) |
| `/api/v1/search` | POST | Semantic search with similarity scores |
| `/api/v1/agent/chat` | POST | Tool-calling agent loop (PubMed, OpenAlex, KEGG, QIIME2) |
| `/api/v1/ingest` | POST | PDF ingestion (background task) |
| `/api/v1/ingest/rss` | POST | RSS feed discovery + ingestion |
| `/api/v1/ingest/briefing` | POST | Briefing text parsing |
| `/api/v1/webhook/briefing` | POST | Briefing bot webhook (shared-secret auth) |
| `/api/v1/webhook/google-form` | POST | Google Form submission webhook |
| `/v1/chat/completions` | POST | OpenAI-compatible chat, backed by the real agent loop |
| `/v1/models` | GET | List models (OpenAI-compatible) |

### Example: RAG Chat

```bash
curl -s http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How does QIIME2 handle diversity analysis?","top_k":5,"max_tokens":500}' \
  | python3 -m json.tool
```

### Example: Agent Chat (tool-calling)

```bash
curl -s http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What KEGG pathways are involved in butyrate production by gut bacteria?","top_k":5}' \
  | python3 -m json.tool
```

Response shape: `{"answer": str, "tools_used": [str, ...], "tool_results_count": int}`.

### Example: Semantic Search

```bash
curl -s http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"UniFrac distance","top_k":5}' \
  | python3 -m json.tool
```

## Agent System

`/api/v1/agent/chat` and `/v1/chat/completions` are both backed by `AgentOrchestrator.run()` (`src/agents/orchestrator.py`): a real OpenAI-style function-calling loop, not a fixed multi-stage pipeline. The model is offered tool schemas on every round and decides whether/which tools to call; results are fed back as tool-role messages until it returns a final answer or a safety cap (`max_tool_rounds`, default 5) is hit, at which point one final answer is forced with no further tools offered.

The orchestrator emits structured, transport-agnostic lifecycle events (`token`, `tool_call`, `tool_result`, `error`, `done`) via an `on_event` callback rather than knowing anything about SSE -- `src/api/sse_adapter.py` is a pure function that translates those events into OpenAI `chat.completion.chunk` dicts for `/v1/chat/completions`' streaming response.

Available tools:
- `pubmed_search` -- PubMed biomedical literature search
- `openalex_search` -- OpenAlex academic works with citation data
- `kegg_lookup` -- KEGG metabolic pathways and compounds
- `qiime2_docs` -- QIIME 2 methods and plugins

## Zotero Integration

The `ZoteroTool` (`src/tools/zotero.py`) connects to Zotero libraries for paper discovery (used by ingestion scripts, not part of the live agent tool set above):

```python
from src.tools.zotero import ZoteroTool

zt = ZoteroTool()  # reads ZOTERO_* from .env
zt.execute(action="collections")          # list collections
zt.execute(action="items", collection_id="BPKGGFEA")  # browse collection
zt.execute(action="search", query="microbiome")        # search library
zt.execute(action="dois")                 # extract DOIs for pipeline
zt.save_dois_to_file(output_path=Path("data/paper_lists/zotero_papers.txt"))
```

The `scripts/populate_zotero.py` script:
1. Searches OpenAlex for papers across 5 microbiome topics
2. Creates Zotero collections and adds items
3. Extracts DOIs via `ZoteroTool`
4. Downloads PDFs and runs the full ingestion pipeline

## Data Pipeline

### Paper Sources

| Source | File | DOIs |
|--------|------|------|
| Knight Lab core | `data/paper_lists/initial_papers.txt` | 112 |
| Zotero/OpenAlex | `data/paper_lists/zotero_papers.txt` | 213 |
| MMC dataset | `data/paper_lists/mmc_papers.txt` | 584 |

### Pipeline Steps

```bash
# Full pipeline against the NRP endpoint: download -> convert -> chunk -> embed -> insert
python scripts/nrp_batch_ingest.py --input data/paper_lists/initial_papers.txt

# Or, for a self-hosted vLLM embedding server instead (see below):
python scripts/download_papers.py --input data/paper_lists/initial_papers.txt --run-pipeline
```

### Storage

Postgres (via `pgGraph`) is the source of truth for text and graph structure; DuckDB holds only vectors for nearest-neighbor search. There is no file-based `graph.graphml`/`chunks_with_emb.json` artifact in the current architecture -- those were an earlier NetworkX-based design (`src/graph/builder.py`, `GraphRAGRetriever`) that the live API no longer uses; the live retrieval path is `HybridRetriever` (`src/retrieval/hybrid_retriever.py`), combining a Postgres/pgGraph traversal with a DuckDB vector search.

Tables (`sql/schema.sql`):
- `papers(doi, title, metadata)`
- `chunks(id, paper_doi, text, section, token_count)`
- `chunk_edges(src_chunk_id, dst_chunk_id, similarity)` -- pgGraph-registered as a `similar_to` edge relationship over `chunks`
- `qiita_studies(study_id, sample_count, contexts, title, abstract, ...)` -- Qiita study registry (`scripts/qiita_registry_ingest.py`), Stage 1 of a separate paper<->microbiome-metadata cross-reference project

The restored Postgres backup also carries `qiita_study_publications` and `paper_study_links` tables from that same cross-reference project; their schema isn't currently tracked in `sql/schema.sql`.

## Self-Hosted vLLM (Cosmos MI300A, alternative)

For offline work or when the NRP endpoint isn't available, KnightGPT can run against self-hosted vLLM servers on Cosmos instead of `https://ellm.nrp-nautilus.io/v1`.

```bash
conda env create -f environment.mi300a.yml   # or:
conda create -n knightGPT python=3.10 -y && conda activate knightGPT
pip install -r requirements.txt

sbatch slurm/vllm_embedding_v1.slurm    # gte-Qwen2-7B-instruct, 1 GPU, port 8001
sbatch slurm/vllm_inference_v1.slurm    # Qwen2.5-72B-Instruct, 4 GPU TP, port 8000
squeue -u $USER

# Update .env with the assigned node hostnames:
# VLLM_EMBEDDING_URL=http://<embedding-node>:8001/v1
# VLLM_EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct
# VLLM_EMBEDDING_DIM=3584
# VLLM_INFERENCE_URL=http://<inference-node>:8000/v1
# VLLM_INFERENCE_MODEL=Qwen/Qwen2.5-72B-Instruct

curl -s http://<embedding-node>:8001/v1/models
curl -s http://<inference-node>:8000/v1/models

python -m src.api.main --host 0.0.0.0 --port 8080
```

vLLM 0.6.6's `rocm_flash_attn.py` doesn't handle the `ENCODER_ONLY` attention type that embedding models need. `slurm/vllm_embedding_v1.slurm` patches this at runtime: extracts the file from the Singularity container, adds an `ENCODER_ONLY` branch (uses `seq_lens` like DECODER, but `causal_mask=False`), and bind-mounts the patched file over the original. Container: `~/vllm_rocm_0.6.6.sif` (vLLM 0.6.6, ROCm 6.3.1, Python 3.12), ROCm arch `gfx942` (MI300A).

`slurm/singularity_stack_proof_test.slurm` builds the same Postgres+pgGraph stack from source via SingularityPRO, for clusters (like Cosmos) where Docker/rootless Podman aren't available -- see that script's own header comments for why plain `docker compose` and rootless Podman are both structurally blocked there.

## Project Structure

```
knightGPT/
├── src/
│   ├── api/main.py              # FastAPI app (11 endpoints)
│   ├── api/sse_adapter.py       # orchestrator event -> OpenAI SSE chunk translator
│   ├── agents/orchestrator.py   # Tool-calling agent loop
│   ├── retrieval/hybrid_retriever.py  # Postgres/pgGraph + DuckDB retrieval (live path)
│   ├── retrieval/retriever.py   # GraphRAGRetriever + RAGEngine (legacy NetworkX path)
│   ├── embedding/embedder.py    # VLLMEmbedder (batch, retry, async; NRP or self-hosted)
│   ├── chunking/chunker.py      # SemanticChunker (tiktoken)
│   ├── graph/duckdb_store.py    # DuckDB vector store (embedded, single-writer)
│   ├── graph/builder.py         # KnowledgeGraphBuilder (NetworkX, legacy)
│   ├── tools/                   # PubMed, OpenAlex, KEGG, QIIME2, Zotero
│   ├── ingestion/                # PDF, RSS, web scraper, briefing parser
│   ├── utils/config.py          # Pydantic BaseSettings
│   └── cli.py                   # Rich terminal chat interface
├── scripts/
│   ├── nrp_batch_ingest.py      # DOI -> PDF -> chunk -> embed -> insert, against NRP
│   ├── download_papers.py       # DOI -> PDF -> pipeline (self-hosted vLLM path)
│   ├── populate_zotero.py       # Zotero + OpenAlex discovery
│   ├── ingest_pipeline.py       # 5-stage ingestion orchestrator
│   ├── qiita_registry_ingest.py # Qiita study registry (Stage 1)
│   ├── migrate_to_postgres.py   # legacy JSON artifacts -> Postgres/DuckDB
│   └── apply_schema.py          # apply sql/schema.sql + verify pgGraph registration
├── slurm/
│   ├── vllm_embedding_v1.slurm  # Self-hosted embedding server (ENCODER_ONLY patch)
│   ├── vllm_inference_v1.slurm  # Self-hosted inference server
│   ├── singularity_stack_proof_test.slurm  # Full stack via SingularityPRO
│   └── weekly_ingest.slurm      # Automated weekly ingestion
├── data/paper_lists/            # DOI lists for ingestion
├── docker/
│   ├── docker-compose.yaml      # API + Open WebUI + Postgres + Watchtower + Cloudflare Tunnel
│   ├── Dockerfile.api
│   └── postgres/                # Postgres + pgGraph image, restore-from-backup init script
├── sql/schema.sql               # papers/chunks/chunk_edges/qiita_studies + pgGraph registration
├── k8s/nrp/                     # Kubernetes manifests + deployment runbooks (NRP cluster)
├── tests/                       # pytest (unit, integration, api markers)
├── .env                         # Configuration (not in git)
├── .env.example                 # Template
└── requirements.txt             # Python dependencies
```

## Configuration

All settings via `.env` file (see `.env.example`). Key variables:

| Variable | Description |
|----------|--------------|
| `VLLM_EMBEDDING_URL` | Embedding endpoint (default: NRP-hosted) |
| `VLLM_INFERENCE_URL` | Inference endpoint (default: NRP-hosted) |
| `VLLM_EMBEDDING_MODEL` | `qwen3-embedding` (NRP) or `Alibaba-NLP/gte-Qwen2-7B-instruct` (self-hosted) |
| `VLLM_INFERENCE_MODEL` | `qwen3` (NRP) or `Qwen/Qwen2.5-72B-Instruct` (self-hosted) |
| `VLLM_API_KEY` | Required for the NRP endpoint -- get one at https://ellm.nrp-nautilus.io/llmtoken |
| `VLLM_EMBEDDING_DIM` | Must match the real output size of `VLLM_EMBEDDING_MODEL` (4096 for `qwen3-embedding`, 3584 for `gte-Qwen2-7B-instruct`) |
| `POSTGRES_DSN` | Postgres connection string (asyncpg format) |
| `GRAPH_SIMILARITY_THRESHOLD` | Edge creation threshold (default: 0.7) |
| `INGEST_DUCKDB_PATH` | DuckDB vector store file path |
| `INGEST_PROCESSED_DIR` | Artifact storage on scratch |
| `ZOTERO_LIBRARY_ID` / `ZOTERO_API_KEY` | Zotero credentials |
| `HF_TOKEN` | HuggingFace token (for gated models, self-hosted path) |

## Testing

```bash
conda activate knightGPT
pytest tests/ -v              # 120 pass, 2 skipped, 9 known failures
pytest tests/ -v -m unit      # unit tests only
pytest tests/ -v -m api       # API tests
```

Known failures are pre-existing mock-target/edge-case issues in `test_chunking.py`/`test_embedding.py`, unrelated to the current agent/deploy architecture (see `CLAUDE.md`).

## Authors

- Leo [@l1joseph](https://github.com/l1joseph)
- Dani [@drahmanucsd](https://github.com/drahmanucsd)

## Acknowledgments

- SDSC Cosmos Cluster team
- National Research Platform (NRP)
- Knight Lab @ UCSD
- vLLM and ROCm communities
