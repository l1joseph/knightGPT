import logging
from typing import List, Dict, Optional

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    GraphBuilder constructs a semantic graph from text chunks with embeddings.
    Nodes represent chunks, edges connect semantically similar chunks.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        :param similarity_threshold: Cosine similarity cutoff for creating edges
        """
        self.similarity_threshold = similarity_threshold
        self.graph = nx.Graph()

    def add_nodes(self, chunks: List[Dict]) -> None:
        """
        Adds each chunk as a node in the graph.
        Node attributes include text and metadata except embedding.
        """
        for chunk in chunks:
            node_id = chunk['node_id']
            # copy metadata excluding embedding
            attrs = {k: v for k, v in chunk.items() if k != 'embedding'}
            self.graph.add_node(node_id, **attrs)
        logger.info(f"Added {len(chunks)} nodes to the graph")

    def add_edges(self, chunks: List[Dict]) -> None:
        """
        Computes pairwise cosine similarity and adds edges for pairs above threshold.
        Stores similarity score as edge weight.
        """
        ids = [chunk['node_id'] for chunk in chunks]
        embeddings = np.array([chunk['embedding'] for chunk in chunks])

        # Normalize embeddings to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.clip(norms, a_min=1e-8, a_max=None)

        sim_matrix = cosine_similarity(normed)
        num_nodes = len(ids)
        edge_count = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                score = float(sim_matrix[i, j])
                if score >= self.similarity_threshold:
                    self.graph.add_edge(ids[i], ids[j], weight=score)
                    edge_count += 1
        logger.info(f"Added {edge_count} edges (threshold={self.similarity_threshold})")

    def build(self, chunks: List[Dict]) -> nx.Graph:
        """
        End-to-end graph construction: add nodes and edges.
        """
        self.add_nodes(chunks)
        self.add_edges(chunks)
        logger.info(f"Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def save_graph(self, path: str, format: str = 'graphml') -> None:
        """
        Saves the graph to disk in the specified format ('graphml' or 'gexf').
        """
        if format == 'graphml':
            nx.write_graphml(self.graph, path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info(f"Graph saved to {path} (format={format})")

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Build semantic graph from chunks')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to JSON file of chunk dicts with embeddings')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save graph file')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Cosine similarity threshold for edges')
    parser.add_argument('--format', type=str, choices=['graphml', 'gexf'], default='graphml',
                        help='Output file format')
    args = parser.parse_args()

    # Load chunks
    with open(args.input, 'r') as f:
        chunks = json.load(f)

    builder = GraphBuilder(similarity_threshold=args.threshold)
    graph = builder.build(chunks)
    builder.save_graph(args.output, format=args.format)
