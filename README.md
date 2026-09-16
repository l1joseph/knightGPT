# KnightGPT

KnightGPT is a Retrieval-Augmented Generation (RAG) assistant for microbiome research. It ingests scientific papers, builds a searchable knowledge graph over them, and answers questions through a chat interface backed by a real tool-calling AI agent — with live lookups against PubMed, OpenAlex, KEGG, and QIIME 2 documentation alongside your own paper corpus.

## Features

- **Chat with your paper corpus** — ask questions in plain language and get answers grounded in the papers you've ingested, with citations.
- **Agent tools** — the assistant can search PubMed and OpenAlex, look up KEGG pathways, and reference QIIME 2 docs while answering.
- **Semantic search** — query the corpus directly and get back the most relevant passages, ranked by similarity.
- **Flexible ingestion** — add papers from a list of DOIs, a Zotero library, RSS feeds, or pasted text/links.
- **Works with any OpenAI-compatible chat UI** — ships with [Open WebUI](https://github.com/open-webui/open-webui) out of the box.

## Getting Started

The fastest way to run KnightGPT is with Docker Compose: one command brings up the API, a Postgres database, and a web chat interface.

### 1. Clone and configure

```bash
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT
cp .env.example .env
```

Open `.env` and set:

| Variable | What to put there |
|----------|--------------------|
| `VLLM_API_KEY` | An API key for your LLM provider. The defaults point at [NRP's hosted endpoint](https://ellm.nrp-nautilus.io/llmtoken) (free to request); any OpenAI-compatible endpoint works. |
| `POSTGRES_PASSWORD` | A real password, if this isn't just for local testing. |
| `WEBUI_SECRET_KEY` | A random secret string for the web UI's session auth. |

### 2. Start it up

```bash
cd docker
docker compose up -d --build
docker compose ps   # wait until postgres and api both report "healthy"
```

### 3. Chat

Open `http://localhost:3000` in a browser, create an account (this is your own private instance), and start asking questions. The API itself is also available directly at `http://localhost:8080` — see `http://localhost:8080/docs` for interactive API docs.

### 4. Add your own papers

```bash
python scripts/nrp_batch_ingest.py --input data/paper_lists/initial_papers.txt
```

Point `--input` at your own text file of one DOI per line. See [Adding Papers](#adding-papers) below for other ways to bring in content.

## Using the API

| Endpoint | Method | What it does |
|----------|--------|----------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint (what the web UI uses) |
| `/api/v1/chat` | POST | RAG chat with citations; supports `"stream": true` |
| `/api/v1/agent/chat` | POST | Chat with full tool access (PubMed, OpenAlex, KEGG, QIIME 2) |
| `/api/v1/search` | POST | Semantic search only, no chat generation |
| `/api/v1/ingest` | POST | Ingest a PDF or a directory of PDFs |
| `/api/v1/ingest/rss` | POST | Discover and ingest new papers from RSS feeds |
| `/health` | GET | Service status |

A couple of examples:

```bash
# Ask a question grounded in your paper corpus
curl -s http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How does QIIME2 handle diversity analysis?","top_k":5}' \
  | python3 -m json.tool

# Ask something that needs a live lookup, not just your corpus
curl -s http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What KEGG pathways are involved in butyrate production by gut bacteria?"}' \
  | python3 -m json.tool

# Search without generating an answer
curl -s http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"UniFrac distance","top_k":5}' \
  | python3 -m json.tool
```

## Adding Papers

KnightGPT can pull papers in from several places:

- **A DOI list** — a plain text file, one DOI per line (see `data/paper_lists/` for examples). Run `python scripts/nrp_batch_ingest.py --input your_list.txt`.
- **A Zotero library** — `python scripts/populate_zotero.py` discovers papers on given topics via OpenAlex, adds them to a Zotero collection, and ingests them.
- **RSS feeds** — `POST /api/v1/ingest/rss` checks configured feeds for new papers and ingests anything matching your keywords.
- **A PDF you already have** — `POST /api/v1/ingest` with a file path, or drop PDFs in a directory and point the endpoint at it.

Every paper goes through the same pipeline: download (if needed) → convert to text → split into semantically coherent chunks → embed → store, so it becomes searchable and citable in chat.

## Configuration

All configuration lives in `.env` (copy from `.env.example` to start). The most relevant settings:

| Variable | Purpose |
|----------|---------|
| `VLLM_EMBEDDING_URL` / `VLLM_INFERENCE_URL` | Where to send embedding/chat requests (any OpenAI-compatible endpoint) |
| `VLLM_EMBEDDING_MODEL` / `VLLM_INFERENCE_MODEL` | Which models to use |
| `VLLM_API_KEY` | Your API key for the above |
| `VLLM_EMBEDDING_DIM` | The embedding model's output size (must match exactly) |
| `POSTGRES_DSN` | Database connection string |
| `GRAPH_SIMILARITY_THRESHOLD` | How similar two chunks must be to link them (0.0-1.0, default 0.7) |
| `ZOTERO_LIBRARY_ID` / `ZOTERO_API_KEY` | Only needed for Zotero-based ingestion |

## Running Your Own Models

By default KnightGPT talks to a hosted LLM endpoint, but it works with anything that speaks the OpenAI API — including a self-hosted [vLLM](https://github.com/vllm-project/vllm) server. If you have your own GPU cluster (KnightGPT's own deployment runs this way on an AMD MI300A cluster via SLURM), see `slurm/vllm_embedding_v1.slurm` and `slurm/vllm_inference_v1.slurm` for a working example, and point `VLLM_EMBEDDING_URL`/`VLLM_INFERENCE_URL` at your own server once it's up.

## Contributing

```bash
conda env create -f environment.mi300a.yml   # or: pip install -r requirements.txt
pytest tests/ -v
black .
ruff check .
```

Pull requests welcome. See `tests/` for existing coverage and conventions before adding new code.

## Authors

- Leo [@l1joseph](https://github.com/l1joseph)
- Dani [@drahmanucsd](https://github.com/drahmanucsd)

## Acknowledgments

- SDSC Cosmos Cluster team
- National Research Platform (NRP)
- Knight Lab @ UCSD
- vLLM and ROCm communities
