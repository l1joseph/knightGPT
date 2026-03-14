# KnightGPT

A Retrieval-Augmented Generation (RAG) system for microbiome research, with knowledge graph-based retrieval, multi-agent reasoning, and domain-specific tools. Runs on SDSC Cosmos cluster (AMD MI300A APUs) using vLLM for inference.

## Architecture

```
Paper Sources (DOI lists, Zotero, RSS, mmc_datasheet.tsv)
    │
    ▼
Download & Convert (Unpaywall/PMC → PDF → marker-pdf/PyMuPDF4LLM → Markdown)
    │
    ▼
Semantic Chunking (paragraph-level, 500 token max, section-aware)
    │
    ▼
Embedding (vLLM: gte-Qwen2-7B-instruct, 3584-dim vectors)
    │
    ▼
Knowledge Graph (NetworkX, cosine similarity ≥ 0.7, max 10 neighbors)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI Server (port 8080)                              │
│  ├── /api/v1/chat         RAG chat (streaming SSE)       │
│  ├── /api/v1/search       Semantic search + citations    │
│  ├── /api/v1/agent/chat   Multi-agent pipeline           │
│  ├── /api/v1/ingest       PDF/RSS ingestion              │
│  └── /v1/chat/completions OpenAI-compatible (Open WebUI) │
└──────────────────────────────────────────────────────────┘
    │
    ▼
vLLM Servers (SLURM + Singularity on MI300A)
├── Embedding: gte-Qwen2-7B-instruct (1 GPU, port 8001)
└── Inference: Qwen2.5-72B-Instruct  (4 GPU TP, port 8000)
```

## Quick Start (Cosmos Cluster)

### 1. Environment Setup

```bash
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT

# Create conda environment
conda env create -f environment.mi300a.yml   # or:
conda create -n knightGPT python=3.10 -y && conda activate knightGPT
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set paths, Zotero keys, HF_TOKEN, etc.
# Cosmos scratch paths: /cosmos/vast/scratch/$USER/knightgpt/data/...
```

### 3. Start vLLM Servers

```bash
sbatch slurm/vllm_embedding_v1.slurm    # Embedding server
sbatch slurm/vllm_inference_v1.slurm     # Inference server

# Wait ~5 min for model loading, check status:
squeue -u $USER

# Update .env with assigned node hostnames:
# VLLM_EMBEDDING_URL=http://<embedding-node>:8001/v1
# VLLM_INFERENCE_URL=http://<inference-node>:8000/v1

# Verify servers:
curl -s http://<embedding-node>:8001/v1/models
curl -s http://<inference-node>:8000/v1/models
```

### 4. Ingest Papers

```bash
conda activate knightGPT

# From DOI list (downloads OA PDFs, converts, chunks, embeds, builds graph)
python scripts/download_papers.py \
  --input data/paper_lists/initial_papers.txt --run-pipeline

# From Zotero library (discovers papers via OpenAlex, adds to Zotero, ingests)
python scripts/populate_zotero.py --run-pipeline

# From TSV dataset
python scripts/download_papers.py \
  --input data/paper_lists/mmc_papers.txt --run-pipeline
```

### 5. Start API Server

```bash
python -m src.api.main --host 0.0.0.0 --port 8080
```

### 6. Access from Local Machine

```bash
# SSH tunnel (match the login node where API is running — check with `hostname`)
ssh -L 8080:localhost:8080 l1joseph@cosmos02.cosmos.sdsc.edu

# Then open in browser:
# http://localhost:8080/health    — health check
# http://localhost:8080/docs      — Swagger UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, chunk/graph counts |
| `/api/v1/chat` | POST | RAG chat with citations (supports `stream: true`) |
| `/api/v1/search` | POST | Semantic search with similarity scores |
| `/api/v1/agent/chat` | POST | Multi-agent: Plan → Tools → Verify → Generate |
| `/api/v1/ingest` | POST | PDF ingestion (background task) |
| `/api/v1/ingest/rss` | POST | RSS feed discovery + ingestion |
| `/api/v1/ingest/briefing` | POST | Briefing text parsing |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/models` | GET | List models |
| `/api/v1/webhook/google-form` | POST | Google Form webhook |

### Example: RAG Chat

```bash
curl -s http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How does QIIME2 handle diversity analysis?","top_k":5,"max_tokens":500}' \
  | python3 -m json.tool
```

### Example: Multi-Agent Chat

```bash
curl -s http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What KEGG pathways are involved in butyrate production by gut bacteria?","top_k":5}' \
  | python3 -m json.tool
```

### Example: Semantic Search

```bash
curl -s http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"UniFrac distance","top_k":5}' \
  | python3 -m json.tool
```

## Multi-Agent System (Phase 6)

The `/api/v1/agent/chat` endpoint uses a 4-stage pipeline:

1. **Planner** — LLM analyzes the query, selects tools, creates sub-queries
2. **Executor** — Runs selected tools + RAG retrieval in parallel
3. **Verifier** — Checks citation relevance
4. **Generator** — Produces final answer with verified citations

Available tools:
- `pubmed_search` — PubMed biomedical literature search
- `openalex_search` — OpenAlex academic works with citation data
- `kegg_lookup` — KEGG metabolic pathways and compounds
- `qiime2_docs` — QIIME 2 methods and plugins

## Zotero Integration

The `ZoteroTool` (`src/tools/zotero.py`) connects to Zotero libraries for paper discovery:

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

| Source | File | Papers |
|--------|------|--------|
| Knight Lab core | `data/paper_lists/initial_papers.txt` | 28 DOIs |
| Zotero/OpenAlex | `data/paper_lists/zotero_papers.txt` | 70 DOIs |
| MMC dataset | `data/paper_lists/mmc_papers.txt` | 194 DOIs |

### Pipeline Steps

```bash
# Full pipeline: download → convert → chunk → embed → graph
python scripts/download_papers.py --input data/paper_lists/initial_papers.txt --run-pipeline

# Or step by step:
python scripts/ingest_pipeline.py \
  --input /cosmos/vast/scratch/l1joseph/knightgpt/data/raw_pdfs \
  --output /cosmos/vast/scratch/l1joseph/knightgpt/data/processed \
  --threshold 0.7
```

### Output Artifacts (on scratch)

| File | Size | Description |
|------|------|-------------|
| `chunks.json` | 11 MB | 6,179 text chunks with metadata |
| `chunks_with_emb.json` | 578 MB | Chunks + 3584-dim embeddings |
| `graph.graphml` | 17 MB | 6,173 nodes, 37,286 edges |

## vLLM on MI300A (ROCm)

### ENCODER_ONLY Bug Fix

vLLM 0.6.6's `rocm_flash_attn.py` doesn't handle `ENCODER_ONLY` attention type, which embedding models use with `--task embedding`. The embedding SLURM script (`slurm/vllm_embedding_v1.slurm`) patches this at runtime by:

1. Extracting `rocm_flash_attn.py` from the Singularity container
2. Adding an `ENCODER_ONLY` branch (uses `seq_lens` like DECODER, but `causal_mask=False`)
3. Bind-mounting the patched file over the original

### Container

- Image: `~/vllm_rocm_0.6.6.sif` (vLLM 0.6.6, ROCm 6.3.1, Python 3.12)
- ROCm arch: `gfx942` (MI300A)

## Project Structure

```
knightGPT/
├── src/
│   ├── api/main.py              # FastAPI app (11 endpoints)
│   ├── agents/orchestrator.py   # Multi-agent pipeline (4 stages)
│   ├── retrieval/retriever.py   # GraphRAGRetriever + RAGEngine
│   ├── embedding/embedder.py    # VLLMEmbedder (batch, retry, async)
│   ├── chunking/chunker.py      # SemanticChunker (tiktoken)
│   ├── graph/builder.py         # KnowledgeGraphBuilder (NetworkX)
│   ├── tools/                   # PubMed, OpenAlex, KEGG, QIIME2, Zotero
│   ├── ingestion/               # PDF, RSS, web scraper, briefing parser
│   ├── storage/storage.py       # Neo4j persistence (optional)
│   ├── utils/config.py          # Pydantic BaseSettings
│   └── cli.py                   # Rich terminal chat interface
├── scripts/
│   ├── download_papers.py       # DOI → PDF → pipeline
│   ├── populate_zotero.py       # Zotero + OpenAlex discovery
│   ├── ingest_pipeline.py       # 5-stage ingestion orchestrator
│   ├── healthcheck.py           # Service health monitoring
│   └── auto_ingest.py           # Automated ingestion
├── slurm/
│   ├── vllm_embedding_v1.slurm  # Embedding server (with ENCODER_ONLY patch)
│   ├── vllm_inference_v1.slurm  # Inference server (Qwen2.5-72B, TP=4)
│   └── weekly_ingest.slurm      # Automated weekly ingestion
├── data/paper_lists/            # DOI lists for ingestion
├── docker/docker-compose.yaml   # Full stack deployment
├── tests/                       # pytest (unit, integration, api markers)
├── .env                         # Configuration (not in git)
├── .env.example                 # Template
└── requirements.txt             # Python dependencies
```

## Configuration

All settings via `.env` file (see `.env.example`). Key variables:

| Variable | Description |
|----------|-------------|
| `VLLM_EMBEDDING_URL` | Embedding server URL (update after sbatch) |
| `VLLM_INFERENCE_URL` | Inference server URL (update after sbatch) |
| `VLLM_INFERENCE_MODEL` | `Qwen/Qwen2.5-72B-Instruct` |
| `GRAPH_SIMILARITY_THRESHOLD` | Edge creation threshold (default: 0.7) |
| `INGEST_RAW_PDF_DIR` | PDF storage on scratch |
| `INGEST_PROCESSED_DIR` | Artifact storage on scratch |
| `ZOTERO_LIBRARY_ID` | Zotero user library ID |
| `ZOTERO_API_KEY` | Zotero API key (read+write) |
| `HF_TOKEN` | HuggingFace token (for gated models) |

## Testing

```bash
conda activate knightGPT
pytest tests/ -v              # 30 pass, 10 known failures
pytest tests/ -v -m unit      # unit tests only
pytest tests/ -v -m api       # API tests
```

## Restart Procedure

```bash
# 1. Submit vLLM servers
sbatch slurm/vllm_embedding_v1.slurm
sbatch slurm/vllm_inference_v1.slurm

# 2. Wait ~5 min, get node hostnames
squeue -u $USER

# 3. Update .env with new hostnames

# 4. Start API
conda activate knightGPT && python -m src.api.main --host 0.0.0.0 --port 8080

# 5. SSH tunnel from local machine
ssh -L 8080:localhost:8080 l1joseph@<login-node>.cosmos.sdsc.edu
```

## Authors

- Leo [@l1joseph](https://github.com/l1joseph)
- Dani [@drahmanucsd](https://github.com/drahmanucsd)

## Acknowledgments

- SDSC Cosmos Cluster team
- Knight Lab @ UCSD
- vLLM and ROCm communities
