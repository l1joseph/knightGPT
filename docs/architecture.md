# knightGPT Architecture

This document describes the system architecture for the knightGPT Retrieval-Augmented Generation (RAG) chatbot specialized in microbiome research.

---

## 1. System Overview

knightGPT is a local-first RAG system that:
1. Extracts text and metadata from scientific PDFs
2. Chunks text into semantic paragraphs
3. Generates vector embeddings for similarity search
4. Builds a semantic graph connecting related chunks
5. Retrieves relevant context using similarity + graph expansion
6. Generates answers using a local LLM (Ollama)

---

## 2. Component Architecture

| Layer | Component | File | Responsibility |
|-------|-----------|------|----------------|
| **Ingestion** | `PDFReader` | `src/ingestion/pdf_reader.py` | Extract text and metadata from PDFs, DOI detection, LLM fallback for missing metadata |
| **Chunking** | `Chunker` | `src/chunking/chunker.py` | Split pages into paragraph chunks with token limits, sentence boundary awareness |
| **Embedding** | `Embedder` | `src/embedding/embedder.py` | Generate vector embeddings using SentenceTransformers |
| **Graph** | `GraphBuilder` | `src/graph/builder.py` | Build semantic graph with cosine similarity edges |
| **Storage** | `GraphStorage` | `src/graph/storage.py` | Optional Neo4j persistence |
| **Retrieval** | `Retriever` | `src/retrieval/retriever.py` | Semantic search with graph-based context expansion |
| **API** | `RAGChatbot` | `src/api/app.py` | CLI interface, orchestration, LLM integration |
| **Utils** | `configure_logging` | `src/utils/logging.py` | Centralized logging configuration |

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

   PDF File
       │
       ▼
┌──────────────────┐     ┌────────────────────────────────────────┐
│   PDFReader      │────▶│ processed_pages.json                   │
│                  │     │ {                                      │
│ • pdfminer.six   │     │   "metadata": {title, authors, doi},   │
│ • DOI regex      │     │   "pages": ["page1 text", ...]         │
│ • LLM fallback   │     │ }                                      │
└──────────────────┘     └────────────────────────────────────────┘
                                        │
                                        ▼
                         ┌──────────────────┐     ┌───────────────┐
                         │    Chunker       │────▶│ chunks.json   │
                         │                  │     │ [{            │
                         │ • Split on ¶     │     │   node_id,    │
                         │ • Token limits   │     │   page,       │
                         │ • Sentence aware │     │   text, ...   │
                         └──────────────────┘     │ }]            │
                                                  └───────────────┘
                                        │
                                        ▼
                         ┌──────────────────┐     ┌───────────────────┐
                         │    Embedder      │────▶│chunks_with_emb.json│
                         │                  │     │ [{                │
                         │ • MiniLM-L6-v2   │     │   ...,            │
                         │ • 384-dim vecs   │     │   embedding: []   │
                         │ • Batch encode   │     │ }]                │
                         └──────────────────┘     └───────────────────┘
                                        │
                                        ▼
                         ┌──────────────────┐     ┌───────────────┐
                         │  GraphBuilder    │────▶│ graph.graphml │
                         │                  │     │               │
                         │ • Cosine sim     │     │ Nodes: chunks │
                         │ • Threshold 0.7  │     │ Edges: similar│
                         │ • NetworkX       │     │               │
                         └──────────────────┘     └───────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                               │
└─────────────────────────────────────────────────────────────────────┘

   User Query
       │
       ▼
┌──────────────────┐
│   RAGChatbot     │
│                  │
│ • Input validate │
│ • Lazy loading   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────┐
│    Retriever     │     │ 1. Embed query (same model as chunks)   │
│                  │────▶│ 2. Cosine similarity → top-K chunks     │
│ • Load chunks    │     │ 3. Graph expansion (N hops)             │
│ • Load graph     │     │ 4. Return sorted results                │
│ • SentenceTransf │     └─────────────────────────────────────────┘
└──────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────┐
│   LLM (Ollama)   │     │ System: "You are an expert researcher"  │
│                  │────▶│ User: "Context: [^1]...\nQuestion:..."  │
│ • llama3 model   │     │ Response: Answer with citations         │
│ • Local inference│     └─────────────────────────────────────────┘
└──────────────────┘
         │
         ▼
   Answer + Citations
```

---

## 4. Key Design Decisions

### 4.1 Local-First Architecture
- **No cloud APIs required**: Uses Ollama for LLM inference
- **Privacy**: All data stays on local machine
- **Cost**: No per-query API costs

### 4.2 Lazy Initialization
- `RAGChatbot` uses lazy loading for retriever and metadata
- Module can be imported without triggering file I/O
- Enables unit testing without data files

### 4.3 Graph-Based Context Expansion
- Beyond top-K similarity, traverses graph to find related chunks
- Captures semantic relationships not apparent from query similarity
- Configurable hop count for precision vs. recall tradeoff

### 4.4 Security Measures
- **Prompt injection protection**: PDF text sanitized before LLM prompts
- **Input validation**: All parameters validated before processing
- **Error handling**: Graceful degradation with user-friendly messages

---

## 5. Data Schemas

### 5.1 Processed Pages (processed_pages.json)
```json
{
  "metadata": {
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "publication_date": "2024-01-15",
    "doi": "10.1234/example"
  },
  "pages": [
    "Text of page 1...",
    "Text of page 2..."
  ]
}
```

### 5.2 Chunks (chunks.json)
```json
[
  {
    "node_id": "uuid-string",
    "page": 1,
    "paragraph_index": 1,
    "chunk_index": 1,
    "text": "Paragraph text..."
  }
]
```

### 5.3 Chunks with Embeddings (chunks_with_emb.json)
```json
[
  {
    "node_id": "uuid-string",
    "page": 1,
    "paragraph_index": 1,
    "chunk_index": 1,
    "text": "Paragraph text...",
    "embedding": [0.1, 0.2, ...]  // 384 dimensions
  }
]
```

---

## 6. Configuration

### 6.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKS_PATH` | `data/chunks.json` | Chunks file path |
| `GRAPH_PATH` | `data/graph.graphml` | Graph file path |
| `GRAPH_FORMAT` | `graphml` | Graph format (graphml/gexf) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `METADATA_PATH` | `data/processed_pages.json` | Metadata file |
| `LLM_MODEL` | `llama3` | Ollama model name |
| `LOG_LEVEL` | `INFO` | Logging level |

### 6.2 Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 500 | Max tokens per chunk |
| `threshold` | 0.7 | Cosine similarity threshold for edges |
| `top_k` | 5 | Initial chunks to retrieve |
| `hops` | 1 | Graph expansion depth |

---

## 7. Error Handling

### 7.1 Custom Exceptions
- `RetrieverError`: Base exception for retrieval errors
- `DataFileError`: File loading/parsing errors

### 7.2 Graceful Degradation
- Missing metadata → fallback to "Unknown" values
- LLM connection failure → clear error message to user
- Invalid inputs → validation with specific error messages

---

## 8. Testing Strategy

### 8.1 Unit Tests
- `test_pdf_reader.py`: Sanitization, JSON extraction, DOI regex
- `test_chunker.py`: Paragraph splitting, token limits
- `test_retriever.py`: File loading, validation, retrieval logic
- `test_app.py`: Chatbot initialization, query processing

### 8.2 Mocking Strategy
- Mock `SentenceTransformer` to avoid model downloads
- Mock Ollama responses for LLM tests
- Use temporary files for file I/O tests

---

## 9. Deployment Options

### 9.1 Local Development
```bash
python src/api/app.py --top_k 5 --hops 1
```

### 9.2 Docker (Future)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "src/api/app.py"]
```

### 9.3 Neo4j Integration (Optional)
```bash
# Write graph to Neo4j
python src/graph/storage.py \
  --action write \
  --input data/chunks_with_emb.json \
  --uri bolt://localhost:7687 \
  --user neo4j --password secret
```

---

## 10. Future Enhancements

| Feature | Priority | Description |
|---------|----------|-------------|
| FastAPI endpoints | High | HTTP API for web integration |
| Vector database | High | FAISS/Pinecone for scalability |
| Multi-document | Medium | Support multiple PDFs per knowledge base |
| Streaming | Medium | Stream LLM responses |
| Figure extraction | Low | Extract images and diagrams |
| Web UI | Low | Browser-based interface |

---

## 11. Performance Considerations

### 11.1 Memory Usage
- Embeddings loaded into RAM (~1.5KB per chunk for 384-dim floats)
- For 10,000 chunks: ~15MB embeddings
- Graph structure adds minimal overhead

### 11.2 Scaling Recommendations
- **< 10K chunks**: File-based storage sufficient
- **10K - 100K chunks**: Consider FAISS for similarity search
- **> 100K chunks**: Use vector database (Pinecone, Weaviate)

---

*Document version 2.0 - Updated to reflect current implementation*
