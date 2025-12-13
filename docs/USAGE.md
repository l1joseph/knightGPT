# KnightGPT Usage Guide

This guide covers post-deployment operations and day-to-day inference usage for KnightGPT.

## Table of Contents

1. [System Health Monitoring](#system-health-monitoring)
2. [Data Management](#data-management)
3. [Configuration Management](#configuration-management)
4. [Troubleshooting](#troubleshooting)
5. [API Usage](#api-usage)
6. [CLI Usage](#cli-usage)
7. [Web Interface](#web-interface)
8. [Best Practices](#best-practices)
9. [Integration Examples](#integration-examples)

## System Health Monitoring

### Health Check Endpoint

The API provides a health check endpoint to monitor system status:

```bash
# Check API health
curl http://localhost:8080/health

# Response:
{
  "status": "healthy",
  "embedding_server": true,
  "inference_server": true,
  "chunks_loaded": 1234,
  "graph_nodes": 5678
}
```

**Status Values:**
- `healthy`: All services operational
- `degraded`: One or more services unavailable but API can still function

### vLLM Server Status Checks

Check embedding server:
```bash
curl http://localhost:8001/v1/models
```

Check inference server:
```bash
curl http://localhost:8000/v1/models
```

### Log Monitoring

**View API logs:**
```bash
# If using Docker
docker logs knightgpt-api -f

# If running directly
tail -f logs/api.log
```

**View vLLM server logs (SLURM):**
```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/vllm_embed_*.out
tail -f logs/vllm_infer_*.out
```

**Log rotation:**
Logs are automatically rotated at 100MB with 1 week retention (configured in `src/utils/logging.py`).

### Resource Usage Monitoring

**GPU Usage (on Cosmos):**
```bash
# Check GPU utilization
rocm-smi

# Check job resource usage
sstat -j <job_id>
```

**Disk Space:**
```bash
# Check data directory
du -sh data/processed/
du -sh data/markdown/
```

**Memory Usage:**
```bash
# Check API process
ps aux | grep "src.api.main"

# Check vLLM processes
ps aux | grep vllm
```

## Data Management

### Adding New Documents via API

**Single PDF:**
```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/path/to/document.pdf",
    "force_ocr": false
  }'
```

**Batch Directory:**
```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_directory": "/path/to/pdfs/",
    "force_ocr": false
  }'
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Ingestion started in background"
}
```

The ingestion process runs asynchronously and will:
1. Convert PDFs to markdown
2. Chunk the documents
3. Generate embeddings
4. Rebuild the knowledge graph
5. Reload the RAG engine

### Batch Ingestion Workflows

**Using the ingestion pipeline script:**
```bash
python scripts/ingest_pipeline.py \
  --input data/raw_pdfs/ \
  --output data/processed/ \
  --embedding-url http://localhost:8001/v1 \
  --threshold 0.7 \
  --max-tokens 500
```

**Options:**
- `--skip-embedding`: Skip embedding generation (use existing)
- `--skip-graph`: Skip graph construction
- `--sync-neo4j`: Sync to Neo4j after processing
- `--force-ocr`: Force OCR on all PDFs

### Updating Embeddings and Graph

**Regenerate embeddings for existing chunks:**
```python
from src.chunking import load_chunks, save_chunks
from src.embedding import VLLMEmbedder

chunks = load_chunks(Path("data/processed/chunks.json"))
embedder = VLLMEmbedder()
chunks = embedder.embed_chunks(chunks)
save_chunks(chunks, Path("data/processed/chunks_with_emb.json"))
```

**Rebuild graph:**
```python
from src.graph import build_graph_from_chunks

build_graph_from_chunks(
    chunks_path=Path("data/processed/chunks_with_emb.json"),
    output_path=Path("data/processed/graph.graphml"),
    threshold=0.7,
    max_neighbors=10
)
```

**Reload API after updates:**
The API automatically reloads when new data is ingested via the `/api/v1/ingest` endpoint. For manual updates, restart the API server:

```bash
# Docker
docker restart knightgpt-api

# Direct
pkill -f "src.api.main"
python -m src.api.main
```

### Data Backup and Restoration

**Backup:**
```bash
# Backup processed data
tar -czf backup_$(date +%Y%m%d).tar.gz \
  data/processed/ \
  data/markdown/

# Backup Neo4j (if using)
docker exec knightgpt-neo4j neo4j-admin dump --to=/backup/neo4j.dump
```

**Restoration:**
```bash
# Extract backup
tar -xzf backup_20240101.tar.gz

# Restore Neo4j
docker exec -i knightgpt-neo4j neo4j-admin load --from=/backup/neo4j.dump --force
```

## Configuration Management

### Environment Variables

**vLLM Servers:**
```bash
export VLLM_EMBEDDING_URL="http://localhost:8001/v1"
export VLLM_INFERENCE_URL="http://localhost:8000/v1"
export VLLM_EMBEDDING_MODEL="Alibaba-NLP/gte-Qwen2-7B-instruct"
export VLLM_INFERENCE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
```

**Neo4j (optional):**
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

**Graph Settings:**
```bash
export GRAPH_SIMILARITY_THRESHOLD="0.7"
export GRAPH_MAX_NEIGHBORS="10"
```

**API Settings:**
```bash
export API_HOST="0.0.0.0"
export API_PORT="8080"
export API_CORS_ORIGINS='["*"]'
```

**Using .env file:**
Create a `.env` file in the project root:
```bash
VLLM_EMBEDDING_URL=http://localhost:8001/v1
VLLM_INFERENCE_URL=http://localhost:8000/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
```

### Model Switching Procedures

**1. Update environment variables:**
```bash
export VLLM_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
export VLLM_INFERENCE_MODEL="Qwen/Qwen2.5-72B-Instruct"
```

**2. Restart vLLM servers:**
```bash
# Cancel existing jobs
scancel <job_id>

# Submit new jobs with updated models
sbatch slurm/vllm_embedding.slurm
sbatch slurm/vllm_inference.slurm
```

**3. Regenerate embeddings (if embedding model changed):**
```bash
python scripts/ingest_pipeline.py \
  --input data/markdown/ \
  --output data/processed/ \
  --embedding-url http://localhost:8001/v1
```

### Threshold Tuning

**Similarity Threshold:**
- Lower (0.5-0.6): More edges, broader context, potentially more noise
- Higher (0.8-0.9): Fewer edges, more precise connections, may miss relevant chunks

**Adjust in code:**
```python
from src.graph import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(
    similarity_threshold=0.75,  # Adjust here
    max_neighbors=10
)
```

**Top-K Retrieval:**
- Lower (3-5): More focused, faster
- Higher (10-20): More comprehensive, slower

**Adjust in API request:**
```python
response = requests.post(
    "http://localhost:8080/api/v1/chat",
    json={"message": "query", "top_k": 10}  # Adjust here
)
```

### Performance Optimization

**Embedding Batch Size:**
```python
embedder = VLLMEmbedder(batch_size=64)  # Increase for faster processing
```

**Graph Hops:**
```python
retriever = GraphRAGRetriever(graph_hops=1)  # Reduce for faster retrieval
```

**API Concurrency:**
Use async clients for multiple requests:
```python
import asyncio
from openai import AsyncOpenAI

async def multiple_queries():
    client = AsyncOpenAI(base_url="http://localhost:8080/v1")
    tasks = [
        client.chat.completions.create(...),
        client.chat.completions.create(...),
    ]
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Error Messages

**"RAG engine not initialized"**
- **Cause**: No chunks loaded at startup
- **Solution**: Run ingestion pipeline first
```bash
python scripts/ingest_pipeline.py --input data/raw_pdfs/ --output data/processed/
```

**"Embedding server health check failed"**
- **Cause**: vLLM embedding server not running
- **Solution**: Check server status and restart if needed
```bash
curl http://localhost:8001/v1/models
sbatch slurm/vllm_embedding.slurm
```

**"Connection refused"**
- **Cause**: Service not running or wrong port
- **Solution**: Check service status
```bash
ss -tlnp | grep 8080
docker ps | grep knightgpt
```

**"Out of memory"**
- **Cause**: Too many chunks or large batch size
- **Solution**: Reduce batch size or increase memory
```python
embedder = VLLMEmbedder(batch_size=16)  # Reduce from default 32
```

**"Invalid embedding dimensions"**
- **Cause**: Mismatched embedding sizes
- **Solution**: Regenerate embeddings with consistent model
```bash
python scripts/ingest_pipeline.py --input data/markdown/ --output data/processed/
```

### Service Restart Procedures

**API Server:**
```bash
# Docker
docker restart knightgpt-api

# Direct
pkill -f "src.api.main"
python -m src.api.main --host 0.0.0.0 --port 8080
```

**vLLM Servers (SLURM):**
```bash
# Cancel and resubmit
scancel <job_id>
sbatch slurm/vllm_embedding.slurm
sbatch slurm/vllm_inference.slurm
```

**Neo4j:**
```bash
docker restart knightgpt-neo4j
```

### Data Corruption Recovery

**Corrupted chunks file:**
```bash
# Validate JSON
python -m json.tool data/processed/chunks_with_emb.json > /dev/null

# Restore from backup
cp backup/chunks_with_emb.json data/processed/
```

**Corrupted graph:**
```bash
# Rebuild from chunks
python -m src.graph.builder \
  --input data/processed/chunks_with_emb.json \
  --output data/processed/graph.graphml \
  --threshold 0.7
```

**Neo4j corruption:**
```bash
# Stop Neo4j
docker stop knightgpt-neo4j

# Restore from backup
docker run --rm \
  -v neo4j-data:/data \
  -v $(pwd)/backup:/backup \
  neo4j:5.16-community \
  neo4j-admin load --from=/backup/neo4j.dump --force

# Restart
docker start knightgpt-neo4j
```

### Performance Degradation Diagnosis

**Check system resources:**
```bash
# CPU/Memory
top
htop

# GPU
rocm-smi
nvidia-smi  # If using NVIDIA

# Disk I/O
iostat -x 1
```

**Check API response times:**
```bash
# Time a request
time curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

**Profile embedding generation:**
```python
import time
from src.embedding import VLLMEmbedder

embedder = VLLMEmbedder()
start = time.time()
embedding = embedder.embed_text("test query")
print(f"Time: {time.time() - start:.2f}s")
```

## API Usage

### Chat Completion (Non-Streaming)

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8080/api/v1/chat",
    json={
        "message": "What is the role of gut microbiota in human health?",
        "top_k": 5,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False
    }
)

result = response.json()
print(result["answer"])
for citation in result["citations"]:
    print(f"- {citation['source_file']}: {citation['similarity']:.3f}")
```

**cURL:**
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the role of gut microbiota in human health?",
    "top_k": 5,
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

### Chat Completion (Streaming)

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8080/api/v1/chat",
    json={
        "message": "Explain microbiome diversity",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith('data: '):
            data = decoded[6:]
            if data == '[DONE]':
                break
            print(data, end='', flush=True)
```

**JavaScript/TypeScript:**
```typescript
async function streamChat(message: string) {
  const response = await fetch('http://localhost:8080/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, stream: true })
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;
        process.stdout.write(data);
      }
    }
  }
}
```

### Semantic Search

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8080/api/v1/search",
    json={
        "query": "microbiome and immune system",
        "top_k": 10,
        "expand_context": True
    }
)

results = response.json()
for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Source: {result['source_file']}")
    print(f"Text: {result['text'][:200]}...")
    print("---")
```

**cURL:**
```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "microbiome and immune system",
    "top_k": 10,
    "expand_context": true
  }'
```

### OpenAI-Compatible Endpoint

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="knightgpt-rag",
    messages=[
        {"role": "user", "content": "What is the gut-brain axis?"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### Error Handling

**Python with retries:**
```python
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def query_with_retry(message: str):
    try:
        response = requests.post(
            "http://localhost:8080/api/v1/chat",
            json={"message": message},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise

result = query_with_retry("test query")
```

## CLI Usage

### Interactive Chat Interface

**Start CLI:**
```bash
python -m src.cli \
  --chunks data/processed/chunks_with_emb.json \
  --graph data/processed/graph.graphml \
  --top-k 5 \
  --hops 1
```

**Options:**
- `--chunks, -c`: Path to chunks JSON file
- `--graph, -g`: Path to graph GraphML file (optional)
- `--top-k, -k`: Number of chunks to retrieve (default: 5)
- `--hops, -n`: Graph hops for context expansion (default: 1)
- `--debug`: Enable debug logging

**Usage:**
```
You: What is the microbiome?
[Thinking...]

Assistant: The microbiome refers to the collection of microorganisms...

Sources:
  • paper1.pdf (Introduction) - similarity: 0.892
  • paper2.pdf (Methods) - similarity: 0.856
```

**Exit:**
Type `quit`, `exit`, or `q` to exit.

### Batch Query Processing

**Python script:**
```python
from pathlib import Path
from src.retrieval import GraphRAGRetriever, RAGEngine

retriever = GraphRAGRetriever(
    chunks_path=Path("data/processed/chunks_with_emb.json"),
    graph_path=Path("data/processed/graph.graphml"),
)
engine = RAGEngine(retriever=retriever)

queries = [
    "What is the gut microbiome?",
    "How does diet affect microbiota?",
    "What is dysbiosis?",
]

results = []
for query in queries:
    response = engine.query(query, top_k=5)
    results.append({
        "query": query,
        "answer": response.answer,
        "citations": len(response.citations)
    })

# Export to JSON
import json
with open("query_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Web Interface

### Accessing Open WebUI

Navigate to: `https://knight-lab-dev.org` (or your configured domain)

### User Authentication

**First-time setup:**
1. Create admin account on first visit
2. Configure authentication settings in environment:
```bash
export WEBUI_AUTH=true
export WEBUI_SECRET_KEY="your-secret-key"
```

**User management:**
- Admin can create/manage users through the web interface
- Users can register if registration is enabled

### Conversation Management

**Features:**
- Save conversations
- Export conversations (JSON, Markdown, PDF)
- Share conversations
- Delete conversations

**Conversation history:**
- Stored in `open-webui-data` Docker volume
- Accessible via web interface

### Model Selection

**Configured models:**
- `knightgpt-rag`: RAG-enhanced chat (default)
- Other models can be added via API configuration

**Switch models:**
- Select from model dropdown in chat interface
- Model list fetched from `/v1/models` endpoint

### RAG Configuration

**Settings (via environment):**
```bash
# Enable web search
ENABLE_RAG_WEB_SEARCH=true
RAG_WEB_SEARCH_ENGINE=searxng

# API endpoint
OPENAI_API_BASE_URL=http://api:8080/v1
OPENAI_API_KEY=EMPTY
```

## Best Practices

### Query Formulation

**Good queries:**
- Specific and focused: "What is the role of Bifidobacterium in infant gut development?"
- Context-aware: "How does antibiotic use affect gut microbiota diversity?"
- Scientific terminology: Use domain-specific terms for better retrieval

**Avoid:**
- Too vague: "Tell me about bacteria"
- Too broad: "Everything about the microbiome"
- Multiple unrelated topics in one query

### Understanding Citations

**Similarity scores:**
- 0.9-1.0: Very high relevance, likely directly answers query
- 0.7-0.9: High relevance, relevant context
- 0.5-0.7: Moderate relevance, may contain related information
- <0.5: Low relevance, may be noise

**Source files:**
- Check source file names for paper/document identification
- Section information indicates where in document chunk came from

**Text snippets:**
- Preview of actual chunk text
- Helps verify relevance before reading full document

### Interpreting Similarity Scores

**Cosine similarity:**
- Range: -1 to 1 (typically 0 to 1 for normalized embeddings)
- 1.0: Identical
- 0.0: Orthogonal (unrelated)
- Negative: Opposing concepts

**Thresholds:**
- Graph edges: Created when similarity >= threshold (default 0.7)
- Retrieval: Top-K chunks by similarity
- Filtering: Can filter results by minimum similarity

### Performance Optimization

**Query optimization:**
- Use specific terms
- Keep queries concise
- Batch multiple queries when possible

**Retrieval optimization:**
- Adjust `top_k` based on query complexity
- Use `expand_context=False` for faster retrieval
- Reduce `graph_hops` for faster graph traversal

**Embedding optimization:**
- Increase batch size for faster processing
- Use async embedding for concurrent requests
- Cache frequently used embeddings

**System optimization:**
- Monitor GPU utilization
- Adjust vLLM memory utilization
- Use quantization for smaller models

## Integration Examples

### Python Client

**Complete example:**
```python
import requests
from typing import List, Dict

class KnightGPTClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def chat(self, message: str, **kwargs) -> Dict:
        """Send chat message."""
        response = requests.post(
            f"{self.base_url}/api/v1/chat",
            json={"message": message, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search."""
        response = requests.post(
            f"{self.base_url}/api/v1/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict:
        """Check health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = KnightGPTClient()
result = client.chat("What is the microbiome?")
print(result["answer"])
```

### Jupyter Notebook

```python
# In Jupyter notebook
import requests
import pandas as pd
from IPython.display import Markdown

def query_knightgpt(question: str):
    response = requests.post(
        "http://localhost:8080/api/v1/chat",
        json={"message": question, "top_k": 5}
    )
    result = response.json()
    
    # Display answer
    display(Markdown(f"## Answer\n{result['answer']}"))
    
    # Display citations as table
    if result['citations']:
        df = pd.DataFrame(result['citations'])
        display(df[['source_file', 'section', 'similarity']])
    
    return result

# Use
result = query_knightgpt("Explain the gut-brain axis")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class KnightGPTClient {
  constructor(baseURL = 'http://localhost:8080') {
    this.baseURL = baseURL;
  }

  async chat(message, options = {}) {
    const response = await axios.post(
      `${this.baseURL}/api/v1/chat`,
      { message, ...options }
    );
    return response.data;
  }

  async search(query, topK = 10) {
    const response = await axios.post(
      `${this.baseURL}/api/v1/search`,
      { query, top_k: topK }
    );
    return response.data;
  }

  async health() {
    const response = await axios.get(`${this.baseURL}/health`);
    return response.data;
  }
}

// Usage
const client = new KnightGPTClient();
client.chat('What is the microbiome?')
  .then(result => console.log(result.answer))
  .catch(console.error);
```

### cURL Examples

**Health check:**
```bash
curl http://localhost:8080/health
```

**Chat:**
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the microbiome?"}'
```

**Search:**
```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "gut bacteria", "top_k": 5}'
```

**Streaming:**
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the microbiome", "stream": true}' \
  --no-buffer
```

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/v1/chat` | POST | RAG chat completion |
| `/api/v1/search` | POST | Semantic search |
| `/api/v1/ingest` | POST | Ingest documents |
| `/api/v1/webhook/google-form` | POST | Google Form webhook |
| `/v1/models` | GET | List models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_EMBEDDING_URL` | `http://localhost:8001/v1` | Embedding server URL |
| `VLLM_INFERENCE_URL` | `http://localhost:8000/v1` | Inference server URL |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `GRAPH_SIMILARITY_THRESHOLD` | `0.7` | Graph edge threshold |
| `API_PORT` | `8080` | API server port |

### Common Commands

```bash
# Start API
python -m src.api.main

# Run ingestion
python scripts/ingest_pipeline.py --input data/raw_pdfs/ --output data/processed/

# Run CLI
python -m src.cli --chunks data/processed/chunks_with_emb.json

# Check health
curl http://localhost:8080/health

# Run bug detection
python scripts/bug_detection.py
```

---

For more information, see:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [README.md](../README.md) - Project overview

