#!/usr/bin/env python3
"""
Full ingestion pipeline for KnightGPT.

Processes PDFs through the complete pipeline:
1. PDF to Markdown conversion
2. Semantic chunking
3. Embedding generation (via vLLM)
4. Knowledge graph construction
5. Optional Neo4j sync
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, get_settings, setup_logging
from src.ingestion import batch_convert_pdfs
from src.chunking import SemanticChunker, save_chunks
from src.embedding import VLLMEmbedder
from src.graph import build_graph_from_chunks

logger = get_logger(__name__)


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    embedding_url: str = None,
    embedding_model: str = None,
    similarity_threshold: float = 0.7,
    max_tokens: int = 500,
    force_ocr: bool = False,
    skip_embedding: bool = False,
    skip_graph: bool = False,
    sync_neo4j: bool = False,
) -> dict:
    """Run the complete ingestion pipeline."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "start_time": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    
    markdown_dir = output_dir / "markdown"
    chunks_file = output_dir / "chunks.json"
    embedded_chunks_file = output_dir / "chunks_with_emb.json"
    graph_file = output_dir / "graph.graphml"
    
    # Step 1: PDF to Markdown
    logger.info("Step 1: Converting PDFs to Markdown")
    pdf_files = list(input_dir.rglob("*.pdf"))
    
    if pdf_files:
        results = batch_convert_pdfs(input_dir, markdown_dir, force_ocr=force_ocr)
        stats["pdfs_processed"] = len(results)
        stats["pdfs_successful"] = sum(1 for r in results if r.get("success"))
    
    # Step 2: Chunking
    logger.info("Step 2: Semantic Chunking")
    chunker = SemanticChunker(max_tokens=max_tokens)
    all_chunks = chunker.chunk_directory(markdown_dir, chunks_file)
    stats["chunks_created"] = len(all_chunks)
    
    # Step 3: Embedding
    if not skip_embedding:
        logger.info("Step 3: Generating Embeddings")
        try:
            embedder = VLLMEmbedder(api_base=embedding_url, model=embedding_model)
            if embedder.check_health():
                all_chunks = embedder.embed_chunks(all_chunks)
                save_chunks(all_chunks, embedded_chunks_file)
                stats["chunks_embedded"] = sum(1 for c in all_chunks if c.embedding)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            stats["embedding_error"] = str(e)
    
    # Step 4: Graph Construction
    if not skip_graph and embedded_chunks_file.exists():
        logger.info("Step 4: Building Knowledge Graph")
        try:
            graph = build_graph_from_chunks(embedded_chunks_file, graph_file, similarity_threshold)
            stats["graph_nodes"] = graph.number_of_nodes()
            stats["graph_edges"] = graph.number_of_edges()
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
    
    # Step 5: Neo4j Sync
    if sync_neo4j:
        logger.info("Step 5: Syncing to Neo4j")
        try:
            from src.storage import sync_to_neo4j
            result = sync_to_neo4j(embedded_chunks_file, graph_file)
            stats["neo4j_sync"] = result
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}")
    
    stats["end_time"] = datetime.now().isoformat()
    with open(output_dir / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Pipeline Complete!")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run KnightGPT ingestion pipeline")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--threshold", "-t", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--sync-neo4j", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    
    stats = run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        similarity_threshold=args.threshold,
        max_tokens=args.max_tokens,
        force_ocr=args.force_ocr,
        skip_embedding=args.skip_embedding,
        skip_graph=args.skip_graph,
        sync_neo4j=args.sync_neo4j,
    )
    
    print("\nPipeline Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
