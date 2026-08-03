import json
import logging
import os
from typing import List, Dict, Optional, Set

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Base exception for Retriever errors."""
    pass


class DataFileError(RetrieverError):
    """Error loading data files."""
    pass


class Retriever:
    """
    Retriever loads preprocessed chunks with embeddings and a semantic graph,
    then retrieves relevant chunks for a query based on embedding similarity
    and optional graph-based neighborhood expansion.
    """

    SUPPORTED_FORMATS = ('graphml', 'gexf')

    def __init__(
        self,
        chunks_path: str,
        graph_path: str,
        graph_format: str = 'graphml',
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the Retriever with data files and embedding model.

        :param chunks_path: Path to JSON file with chunk dicts (must include 'node_id' and 'embedding')
        :param graph_path: Path to graph file (GraphML or GEXF)
        :param graph_format: 'graphml' or 'gexf'
        :param model_name: Sentence-Transformers model name for query embedding
        :raises DataFileError: If data files cannot be loaded
        :raises ValueError: If graph format is unsupported
        """
        # Validate graph format first (before doing any I/O)
        graph_format = graph_format.lower()
        if graph_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported graph format: {graph_format}. Supported: {self.SUPPORTED_FORMATS}")

        # Load chunks with error handling
        self.chunks = self._load_chunks(chunks_path)

        # Validate chunks have required fields
        self._validate_chunks(self.chunks)

        # Extract IDs and embeddings
        self.ids = [chunk['node_id'] for chunk in self.chunks]
        self.embeddings = np.array([chunk['embedding'] for chunk in self.chunks])

        # Normalize embeddings (with warning for zero-norm vectors)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        zero_norm_count = np.sum(norms < 1e-8)
        if zero_norm_count > 0:
            logger.warning(f"Found {zero_norm_count} chunks with near-zero embedding norms")
        self.normed = self.embeddings / np.clip(norms, a_min=1e-8, a_max=None)

        # Load graph with error handling
        self.graph = self._load_graph(graph_path, graph_format)

        # Validate graph-chunk consistency
        self._validate_graph_chunk_consistency()

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise DataFileError(f"Failed to load embedding model '{model_name}': {e}")

    def _load_chunks(self, chunks_path: str) -> List[Dict]:
        """Load chunks from JSON file with error handling."""
        logger.info(f"Loading chunks from {chunks_path}")

        if not os.path.exists(chunks_path):
            raise DataFileError(f"Chunks file not found: {chunks_path}")

        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise DataFileError(f"Invalid JSON in chunks file {chunks_path}: {e}")
        except PermissionError:
            raise DataFileError(f"Permission denied reading chunks file: {chunks_path}")
        except Exception as e:
            raise DataFileError(f"Error reading chunks file {chunks_path}: {e}")

        if not isinstance(chunks, list):
            raise DataFileError(f"Chunks file must contain a JSON array, got {type(chunks).__name__}")

        if len(chunks) == 0:
            logger.warning(f"Chunks file is empty: {chunks_path}")

        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    def _validate_chunks(self, chunks: List[Dict]) -> None:
        """Validate that chunks have required fields."""
        required_fields = {'node_id', 'embedding', 'text'}
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise DataFileError(f"Chunk {i} is not a dict: {type(chunk).__name__}")

            missing = required_fields - set(chunk.keys())
            if missing:
                raise DataFileError(f"Chunk {i} missing required fields: {missing}")

    def _load_graph(self, graph_path: str, graph_format: str) -> nx.Graph:
        """Load graph from file with error handling."""
        logger.info(f"Loading graph from {graph_path} ({graph_format})")

        if not os.path.exists(graph_path):
            raise DataFileError(f"Graph file not found: {graph_path}")

        try:
            if graph_format == 'graphml':
                graph = nx.read_graphml(graph_path)
            else:  # gexf
                graph = nx.read_gexf(graph_path)
        except Exception as e:
            raise DataFileError(f"Error reading graph file {graph_path}: {e}")

        logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def _validate_graph_chunk_consistency(self) -> None:
        """Warn about mismatches between graph nodes and chunk IDs."""
        chunk_ids = set(self.ids)
        graph_nodes = set(self.graph.nodes())

        # Chunks not in graph
        missing_in_graph = chunk_ids - graph_nodes
        if missing_in_graph:
            logger.warning(f"{len(missing_in_graph)} chunk IDs not found in graph")

        # Graph nodes not in chunks
        missing_in_chunks = graph_nodes - chunk_ids
        if missing_in_chunks:
            logger.warning(f"{len(missing_in_chunks)} graph nodes not found in chunks")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed the input query into the same vector space as chunks.
        """
        vec = self.model.encode([query], convert_to_numpy=True)
        # normalize
        norm = np.linalg.norm(vec)
        return (vec / max(norm, 1e-8)).squeeze()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 1
    ) -> List[Dict]:
        """
        Retrieve top_k chunks semantically closest to query and expand via graph.

        :param query: User query string
        :param top_k: Number of initial chunks to retrieve (must be >= 1)
        :param hops: Number of graph hops to expand context (must be >= 0)
        :return: List of retrieved chunk dicts with metadata
        :raises ValueError: If top_k < 1 or hops < 0
        """
        # Validate inputs
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if hops < 0:
            raise ValueError(f"hops must be >= 0, got {hops}")
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        # Embed query and compute similarities
        q_vec = self.embed_query(query)
        sims = np.dot(self.normed, q_vec)

        # Get top_k indices (handle case where we have fewer chunks than top_k)
        actual_top_k = min(top_k, len(self.ids))
        best_idx = np.argsort(sims)[::-1][:actual_top_k]
        selected_ids = [self.ids[i] for i in best_idx]
        logger.info(f"Top {actual_top_k} nodes by similarity: {selected_ids[:3]}...")

        # Expand via graph neighbors
        all_ids: Set[str] = set(selected_ids)
        if hops > 0:
            for node_id in selected_ids:
                # Check if node exists in graph before traversal
                if node_id not in self.graph:
                    logger.debug(f"Node {node_id} not in graph, skipping expansion")
                    continue
                try:
                    lengths = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=hops)
                    all_ids.update(lengths.keys())
                except nx.NetworkXError as e:
                    logger.warning(f"Graph traversal error for node {node_id}: {e}")
            logger.info(f"Expanded to {len(all_ids)} nodes with hops={hops}")

        # Collect chunk metadata (with warning for missing chunks)
        id_to_chunk = {c['node_id']: c for c in self.chunks}
        results = []
        missing_count = 0
        for nid in all_ids:
            if nid in id_to_chunk:
                results.append(id_to_chunk[nid])
            else:
                missing_count += 1

        if missing_count > 0:
            logger.warning(f"{missing_count} node IDs from graph expansion not found in chunks")

        # Sort results by semantic similarity descending
        # Build a lookup for similarity scores
        id_to_sim = {self.ids[i]: sims[i] for i in range(len(self.ids))}
        results.sort(key=lambda c: id_to_sim.get(c['node_id'], 0.0), reverse=True)

        return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrieve relevant chunks for a query')
    parser.add_argument('--chunks', type=str, required=True,
                        help='Path to JSON chunks file')
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to graph file')
    parser.add_argument('--format', type=str, choices=['graphml','gexf'], default='graphml',
                        help='Graph file format')
    parser.add_argument('--query', type=str, required=True,
                        help='Query text')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of initial chunks to retrieve')
    parser.add_argument('--hops', type=int, default=1,
                        help='Number of graph hops for expansion')
    args = parser.parse_args()

    retriever = Retriever(
        chunks_path=args.chunks,
        graph_path=args.graph,
        graph_format=args.format
    )
    results = retriever.retrieve(args.query, top_k=args.top_k, hops=args.hops)
    # Output simplified JSON
    output = [
        {
            'node_id': c['node_id'],
            'page': c.get('page'),
            'paragraph_index': c.get('paragraph_index'),
            'chunk_index': c.get('chunk_index'),
            'text': c.get('text')
        } for c in results
    ]
    print(json.dumps(output, indent=2))
