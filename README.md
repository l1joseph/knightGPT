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
├── data/
│   ├── raw_pdfs/                # Source PDF files
│   ├── markdown/                # Converted markdown files
│   └── processed/               # Processed chunks and embeddings
│
├── docs/
│   ├── DEPLOYMENT.md            # Deployment guide for Cosmos cluster
│   └── USAGE.md                 # Post-deployment and daily usage guide
│
├── scripts/
│   ├── ingest_pipeline.py       # Full ingestion pipeline runner
│   ├── bug_detection.py         # Comprehensive bug detection script
│   ├── healthcheck.py           # Service health monitoring
│   └── monitor.py               # Continuous monitoring daemon
│
├── slurm/
│   ├── vllm_embedding.slurm     # SLURM job for embedding server
│   └── vllm_inference.slurm     # SLURM job for inference server
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
│   │   └── main.py              # FastAPI application (all endpoints)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py           # Centralized logging
│   │   └── config.py            # Configuration management
│   │
│   └── cli.py                   # CLI interface
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_chunking.py         # Chunking module tests
│   ├── test_embedding.py        # Embedding module tests
│   ├── test_graph.py            # Graph module tests
│   ├── test_retrieval.py        # Retrieval module tests
│   ├── test_integration.py      # Integration tests
│   └── test_api.py              # API endpoint tests
│
├── docker/
│   ├── Dockerfile.api           # API server container
│   └── docker-compose.yaml      # Full stack deployment
│
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Access to SDSC Cosmos cluster with MI300A APUs (for production)
- Python 3.10+
- ROCm 6.2+ (pre-installed on Cosmos, or local setup for development)
- Neo4j (optional, for persistent graph storage)
- vLLM servers running (embedding and inference)

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

# Or start from PDFs
python scripts/ingest_pipeline.py \
    --input data/raw_pdfs/ \
    --output data/processed/ \
    --embedding-url http://localhost:8001/v1
```

### 4. Run Inference API

```bash
# Start FastAPI server
python -m src.api.main --host 0.0.0.0 --port 8080
```

### 5. Verify Installation

```bash
# Run health check
python scripts/healthcheck.py

# Run bug detection
python scripts/bug_detection.py

# Run tests
pytest tests/
```

### 6. Use CLI Interface (Alternative to API)

```bash
# Interactive chat interface
python -m src.cli \
    --chunks data/processed/chunks_with_emb.json \
    --graph data/processed/graph.graphml \
    --top-k 5
```

### 7. Deploy Web Interface

```bash
# Using docker-compose
cd docker && docker-compose up -d
```

## Key Features

- **Knowledge Graph RAG**: Graph-based retrieval for improved context understanding
- **Multiple Ingestion Methods**: PDF upload, Google Forms webhook, web scraping
- **Comprehensive Testing**: Unit, integration, and API tests with bug detection
- **Health Monitoring**: Real-time health checks and continuous monitoring
- **OpenAI-Compatible API**: Works with Open WebUI and other OpenAI-compatible clients
- **CLI Interface**: Terminal-based chat interface for quick queries
- **Production-Ready**: Error handling, input validation, and robust error recovery

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
| `/health`        | GET    | Service health status        |
| `/api/v1/chat`   | POST   | RAG-enhanced chat completion |
| `/api/v1/ingest` | POST   | Ingest new documents         |
| `/api/v1/search` | POST   | Semantic search              |
| `/api/v1/webhook/google-form` | POST | Google Form webhook |
| `/v1/models`     | GET    | List models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |

See [docs/USAGE.md](docs/USAGE.md) for detailed API usage examples.

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

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_chunking.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Monitoring

Monitor system health:

```bash
# One-time health check
python scripts/healthcheck.py

# Continuous monitoring
python scripts/monitor.py --interval 60

# Save health report
python scripts/healthcheck.py --output health_report.json
```

## Documentation

- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Complete deployment guide for SDSC Cosmos cluster
- **[USAGE.md](docs/USAGE.md)** - Post-deployment operations and day-to-day usage guide

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and bug detection: `pytest tests/ && python scripts/bug_detection.py`
4. Submit a pull request

## License

MIT License

## Authors:

- Leo [@l1joseph](https://github.com/l1joseph)
- Dani [@drahmanucsd](https://github.com/drahmanucsd)

## Acknowledgments

- SDSC Cosmos Cluster team
- Knight Lab @ UCSD
- vLLM and ROCm communities
