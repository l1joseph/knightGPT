# KnightGPT - Microbiome Knowledge Graph RAG System

A Retrieval-Augmented Generation (RAG) system specialized in microbiome research, using knowledge graph-based retrieval with vLLM for inference on AMD MI300A APUs (SDSC Cosmos Cluster).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Pipeline                       │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│  Google Form    │   Web Scraper   │      Manual Upload              │
│   Webhook       │    (Scrapy)     │      (FastAPI)                  │
└────────┬────────┴────────┬────────┴────────┬────────────────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PDF → Markdown Conversion                        │
│                   (marker-pdf / PyMuPDF4LLM)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Semantic Chunking                               │
│               (Paragraph-level with metadata)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding Generation                              │
│        vLLM Server (Alibaba-NLP/gte-Qwen2-7B-instruct)             │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Knowledge Graph                                  │
│              (NetworkX + Neo4j for persistence)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Inference API                                    │
│   vLLM OpenAI-Compatible Server (meta-llama/Llama-3.3-70B-Instruct)│
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Web Interface                                   │
│          Open WebUI (https://knight-lab-dev.org)                    │
│                 Cloudflare + Watchtower                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
knightGPT/
├── configs/
│   ├── vllm_embedding.yaml      # vLLM embedding server config
│   ├── vllm_inference.yaml      # vLLM inference server config
│   └── open_webui.yaml          # Open WebUI configuration
│
├── data/
│   ├── raw_pdfs/                # Source PDF files
│   ├── markdown/                # Converted markdown files
│   └── processed/               # Processed chunks and embeddings
│
├── docs/
│   ├── DEPLOYMENT.md            # Deployment guide for Cosmos cluster
│   ├── API.md                   # API documentation
│   └── ARCHITECTURE.md          # System architecture details
│
├── scripts/
│   ├── setup_vllm_rocm.sh       # Setup vLLM for AMD MI300A
│   ├── ingest_pipeline.py       # Full ingestion pipeline runner
│   └── healthcheck.py           # Service health monitoring
│
├── slurm/
│   ├── vllm_embedding.slurm     # SLURM job for embedding server
│   ├── vllm_inference.slurm     # SLURM job for inference server
│   └── batch_ingest.slurm       # SLURM job for batch processing
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_reader.py        # PDF metadata extraction
│   │   ├── pdf_to_markdown.py   # PDF → Markdown conversion
│   │   ├── google_form_webhook.py # Google Form integration
│   │   └── web_scraper.py       # Web scraping module
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py           # Semantic paragraph chunking
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py          # vLLM embedding client
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── builder.py           # Knowledge graph construction
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── storage.py           # Neo4j persistence layer
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py         # Graph-based RAG retriever
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── routes.py            # API endpoints
│   │   └── models.py            # Pydantic models
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py           # Centralized logging
│   │   └── config.py            # Configuration management
│   │
│   └── cli.py                   # CLI interface
│
├── docker/
│   ├── Dockerfile.api           # API server container
│   ├── Dockerfile.ingest        # Ingestion worker container
│   └── docker-compose.yaml      # Full stack deployment
│
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Access to SDSC Cosmos cluster with MI300A APUs
- Python 3.10+
- ROCm 6.2+ (pre-installed on Cosmos)
- Neo4j (optional, for persistent graph storage)

### 1. Setup Environment on Cosmos

```bash
# Clone repository
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT

# Create conda environment
conda create -n knightgpt python=3.11 -y
conda activate knightgpt

# Install dependencies
pip install -r requirements.txt
```

### 2. Start vLLM Servers (via SLURM)

```bash
# Submit embedding server job
sbatch slurm/vllm_embedding.slurm

# Submit inference server job
sbatch slurm/vllm_inference.slurm
```

### 3. Build Knowledge Graph

```bash
# Convert PDFs to markdown
python -m src.ingestion.pdf_to_markdown data/raw_pdfs/ data/markdown/

# Run full ingestion pipeline
python scripts/ingest_pipeline.py \
    --input data/markdown/ \
    --output data/processed/ \
    --embedding-url http://localhost:8001/v1 \
    --threshold 0.7
```

### 4. Run Inference API

```bash
# Start FastAPI server
python -m src.api.main --host 0.0.0.0 --port 8000
```

### 5. Deploy Web Interface

```bash
# Using docker-compose
cd docker && docker-compose up -d
```

## Model Selection

### Embedding Model

**Alibaba-NLP/gte-Qwen2-7B-instruct** - 7B parameter embedding model

- High-quality embeddings for scientific text
- 8192 token context window
- Optimized for semantic similarity

### Inference Model

**meta-llama/Llama-3.3-70B-Instruct** - 70B parameter chat model

- State-of-the-art reasoning capabilities
- 128K context window
- Tensor parallelism across 4 MI300A APUs

## Configuration

### Environment Variables

```bash
# vLLM Servers
export VLLM_EMBEDDING_URL="http://localhost:8001/v1"
export VLLM_INFERENCE_URL="http://localhost:8000/v1"

# Neo4j (optional)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# AMD MI300A Optimizations
export VLLM_ROCM_USE_AITER=1
export PYTORCH_ROCM_ARCH="gfx942"
```

## Web Application

The web interface is hosted at `https://knight-lab-dev.org` using:

- **Open WebUI**: Self-hosted ChatGPT-like interface
- **Cloudflare**: SSL/CDN/DNS management
- **Watchtower**: Automatic container updates

### Features

- Multi-model chat interface
- RAG with citations
- Document upload
- Conversation history
- User authentication

## API Endpoints

| Endpoint         | Method | Description                  |
| ---------------- | ------ | ---------------------------- |
| `/api/v1/chat`   | POST   | RAG-enhanced chat completion |
| `/api/v1/ingest` | POST   | Ingest new documents         |
| `/api/v1/search` | POST   | Semantic search              |
| `/api/v1/health` | GET    | Service health status        |

## SDSC Cosmos Cluster Notes

### Resource Allocation

- Each MI300A has 128GB HBM3 memory
- 4 APUs per node (512GB total)
- Use `--tensor-parallel-size 4` for 70B models

### SLURM Configuration

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Authors:

- Leo [@l1joseph](https://github.com/l1joseph)
- Dani [@drahmanucsd](https://github.com/drahmanucsd)

## Acknowledgments

- SDSC Cosmos Cluster team
- Knight Lab @ UCSD
- vLLM and ROCm communities
