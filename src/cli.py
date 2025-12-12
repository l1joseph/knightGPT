#!/usr/bin/env python3
"""
KnightGPT CLI - Terminal-based chatbot interface.

Usage:
    python -m src.cli --chunks data/processed/chunks_with_emb.json
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .retrieval import GraphRAGRetriever, RAGEngine
from .utils import get_logger, setup_logging

logger = get_logger(__name__)
console = Console()


def run_cli(
    chunks_path: Path,
    graph_path: Path = None,
    top_k: int = 5,
    hops: int = 1,
):
    """
    Run interactive CLI chat interface.
    
    Args:
        chunks_path: Path to chunks JSON with embeddings
        graph_path: Path to graph GraphML file
        top_k: Number of chunks to retrieve
        hops: Graph hops for context expansion
    """
    console.print(Panel.fit(
        "[bold blue]KnightGPT[/bold blue] - Microbiome Research Assistant\n"
        "Type your question and press Enter. Type 'quit' to exit.",
        title="Welcome",
    ))
    
    # Initialize retriever and engine
    console.print("[dim]Loading knowledge base...[/dim]")
    
    try:
        retriever = GraphRAGRetriever(
            chunks_path=chunks_path,
            graph_path=graph_path,
            top_k=top_k,
            graph_hops=hops,
        )
        
        engine = RAGEngine(retriever=retriever)
        
        console.print(f"[green]✓ Loaded {len(retriever.chunks)} chunks[/green]")
        console.print(f"[green]✓ Graph has {retriever.graph_builder.graph.number_of_nodes()} nodes[/green]")
        
    except Exception as e:
        console.print(f"[red]Error loading knowledge base: {e}[/red]")
        sys.exit(1)
    
    # Chat loop
    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if question.lower() in ["quit", "exit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if not question.strip():
                continue
            
            # Process query
            console.print("[dim]Thinking...[/dim]")
            
            response = engine.query(
                question=question,
                top_k=top_k,
            )
            
            # Display response
            console.print("\n[bold green]Assistant[/bold green]")
            console.print(Markdown(response.answer))
            
            # Display citations
            if response.citations:
                console.print("\n[bold yellow]Sources:[/bold yellow]")
                for cit in response.citations[:5]:
                    source = Path(cit.source_file).stem if cit.source_file else "Unknown"
                    section = cit.section or "General"
                    console.print(f"  • {source} ({section}) - similarity: {cit.similarity:.3f}")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("CLI error")


def main():
    parser = argparse.ArgumentParser(
        description="KnightGPT CLI - Terminal-based microbiome research assistant"
    )
    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        default=Path("data/processed/chunks_with_emb.json"),
        help="Path to chunks JSON file",
    )
    parser.add_argument(
        "--graph", "-g",
        type=Path,
        default=None,
        help="Path to graph GraphML file",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--hops", "-n",
        type=int,
        default=1,
        help="Graph hops for context expansion",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="WARNING")
    
    # Verify files exist
    if not args.chunks.exists():
        console.print(f"[red]Chunks file not found: {args.chunks}[/red]")
        console.print("[dim]Run the ingestion pipeline first:[/dim]")
        console.print("  python scripts/ingest_pipeline.py --input data/raw_pdfs/ --output data/processed/")
        sys.exit(1)
    
    run_cli(
        chunks_path=args.chunks,
        graph_path=args.graph,
        top_k=args.top_k,
        hops=args.hops,
    )


if __name__ == "__main__":
    main()
