# knightGPT - Microbiome RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot specialized in microbiome research. It extracts text and metadata from scientific PDFs, builds a semantic paragraph-level knowledge graph, and enables interactive querying via local LLM (Ollama).

## Features

- PDF text and metadata extraction with DOI detection
- Paragraph-level text chunking with sentence boundary awareness
- Semantic embeddings using SentenceTransformers
- Graph-based context expansion for improved retrieval
- Local LLM inference via Ollama (no API keys required)
- Inline citations with document metadata

---

## Repository Structure

```
knightGPT/
├── data/
│   ├── raw_pdfs/              # Source PDF files
│   ├── processed_pages.json   # Output from pdf_reader.py
│   ├── chunks.json            # Paragraph chunks
│   ├── chunks_with_emb.json   # Chunks with embeddings
│   └── graph.graphml          # Semantic graph
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py             # RAGChatbot class & CLI
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_reader.py      # PDF → pages + metadata
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py         # Pages → paragraph chunks
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py        # Chunks → vector embeddings
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py         # Build semantic graph
│   │   └── storage.py         # Neo4j persistence (optional)
│   │
│   ├── metadata/
│   │   ├── __init__.py
│   │   └── extractor.py       # Standalone metadata extraction
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py       # Semantic search + graph expansion
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Centralized logging
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_pdf_reader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_app.py
│
├── docs/
│   └── architecture.md        # System architecture
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull the llama3 model (in a new terminal)
ollama pull llama3
```

---

## Building the Knowledge Graph

### Step 1: Extract PDF content

```bash
python src/ingestion/pdf_reader.py \
  data/raw_pdfs/your_paper.pdf \
  -o data/processed_pages.json
```

Output: JSON with `metadata` (title, authors, DOI, date) and `pages` array.

### Step 2: Chunk into paragraphs

```bash
python src/chunking/chunker.py \
  --pages data/processed_pages.json \
  --output data/chunks.json \
  --max_tokens 500
```

### Step 3: Generate embeddings

```bash
python src/embedding/embedder.py \
  --input data/chunks.json \
  --output data/chunks_with_emb.json \
  --model all-MiniLM-L6-v2
```

### Step 4: Build semantic graph

```bash
python src/graph/builder.py \
  --input data/chunks_with_emb.json \
  --output data/graph.graphml \
  --threshold 0.7
```

---

## Running the Chatbot

```bash
python src/api/app.py --top_k 5 --hops 1
```

**Parameters:**
- `--top_k`: Number of initial chunks to retrieve (default: 5)
- `--hops`: Graph hops for context expansion (default: 1)
- `--chunks`: Custom path to chunks JSON
- `--graph`: Custom path to graph file
- `--metadata`: Custom path to metadata JSON

**Example session:**
```
Microbiome RAG Chatbot CLI (type 'exit' to quit)

Enter your question: What role do gut bacteria play in digestion?

=== Answer (based on 'Microbiome Research' DOI: 10.1234/example) ===
Gut bacteria play a crucial role in digestion by... [^1] [^2]

=== Citations ===
[^1] Title: Microbiome Research, DOI: 10.1234/example, Page 3, Para 2, Chunk 1
[^2] Title: Microbiome Research, DOI: 10.1234/example, Page 5, Para 1, Chunk 1
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKS_PATH` | `data/chunks.json` | Path to chunks JSON |
| `GRAPH_PATH` | `data/graph.graphml` | Path to graph file |
| `GRAPH_FORMAT` | `graphml` | Graph format (graphml or gexf) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `METADATA_PATH` | `data/processed_pages.json` | Path to PDF metadata |
| `LLM_MODEL` | `llama3` | Ollama model name |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Programmatic Usage

```python
from src.api.app import RAGChatbot

# Create chatbot instance
chatbot = RAGChatbot(
    chunks_path="data/chunks_with_emb.json",
    graph_path="data/graph.graphml"
)

# Query
answer = chatbot.process_query(
    "What is the role of probiotics?",
    top_k=5,
    hops=1
)
```

---

## Troubleshooting

### ModuleNotFoundError
Run from the repository root or use the module syntax:
```bash
PYTHONPATH=. python src/api/app.py
# or
python -m src.api.app
```

### Ollama Connection Error
Ensure Ollama is running:
```bash
ollama serve  # Start the service
ollama list   # Verify models are available
```

### Empty or Poor Results
- Increase `--top_k` to retrieve more context
- Decrease similarity threshold in graph building
- Ensure PDF text was extracted correctly (check `processed_pages.json`)

---

## Security Notes

- PDF text is sanitized before LLM prompts to prevent prompt injection
- File paths are validated to prevent path traversal
- Input parameters are validated before processing

---

## Future Enhancements

- [ ] FastAPI HTTP endpoints
- [ ] FAISS/Pinecone vector store integration
- [ ] Figure and table extraction
- [ ] Streaming LLM responses
- [ ] Multi-document support
- [ ] Web UI

---

## License

MIT License

---

*Version 2.0 - Updated with security fixes and test suite*
