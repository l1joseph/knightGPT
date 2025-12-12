"""Knowledge graph construction from embedded chunks."""

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

from ..chunking import Chunk, load_chunks
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


class KnowledgeGraphBuilder:
    """
    Build a semantic knowledge graph from embedded chunks.
    
    The graph connects chunks based on cosine similarity of their
    embeddings. Edges are created when similarity exceeds a threshold.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
    ):
        """
        Initialize graph builder.
        
        Args:
            similarity_threshold: Minimum cosine similarity for edges
            max_neighbors: Maximum edges per node
        """
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.graph = nx.Graph()

    def build_graph(self, chunks: list[Chunk]) -> nx.Graph:
        """
        Build knowledge graph from chunks.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            NetworkX graph
        """
        logger.info(f"Building graph from {len(chunks)} chunks")
        
        # Filter chunks with embeddings
        chunks_with_emb = [c for c in chunks if c.embedding is not None]
        
        if not chunks_with_emb:
            logger.warning("No chunks with embeddings found")
            return self.graph
        
        # Add nodes
        for chunk in chunks_with_emb:
            self.graph.add_node(
                chunk.id,
                text=chunk.text,
                source_file=chunk.source_file,
                section=chunk.section,
                metadata=chunk.metadata,
                token_count=chunk.token_count,
            )
        
        # Compute similarity matrix
        embeddings = np.array([c.embedding for c in chunks_with_emb])
        n = len(chunks_with_emb)
        
        logger.info("Computing similarity matrix...")
        
        # Add edges based on similarity
        edge_count = 0
        for i in range(n):
            similarities = []
            
            for j in range(n):
                if i != j:
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    if sim >= self.similarity_threshold:
                        similarities.append((j, sim))
            
            # Keep only top neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for j, sim in similarities[:self.max_neighbors]:
                chunk_i = chunks_with_emb[i]
                chunk_j = chunks_with_emb[j]
                
                if not self.graph.has_edge(chunk_i.id, chunk_j.id):
                    self.graph.add_edge(
                        chunk_i.id,
                        chunk_j.id,
                        weight=sim,
                        similarity=sim,
                    )
                    edge_count += 1
        
        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
    ) -> list[str]:
        """
        Get neighbors within N hops.
        
        Args:
            node_id: Starting node ID
            hops: Number of hops to traverse
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor != node_id:
                        neighbors.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
        
        return list(neighbors)

    def find_similar_chunks(
        self,
        query_embedding: list[float],
        chunks: list[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Find chunks most similar to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            chunks: List of chunks to search
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity) tuples
        """
        query_emb = np.array(query_embedding)
        results = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                sim = 1 - cosine(query_emb, np.array(chunk.embedding))
                results.append((chunk, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def expand_context(
        self,
        initial_chunks: list[Chunk],
        all_chunks: list[Chunk],
        hops: int = 1,
    ) -> list[Chunk]:
        """
        Expand initial chunks with graph neighbors.
        
        Args:
            initial_chunks: Starting chunks
            all_chunks: All available chunks
            hops: Number of graph hops
            
        Returns:
            Expanded list of chunks
        """
        chunk_map = {c.id: c for c in all_chunks}
        result_ids = {c.id for c in initial_chunks}
        
        for chunk in initial_chunks:
            neighbor_ids = self.get_neighbors(chunk.id, hops)
            result_ids.update(neighbor_ids)
        
        return [chunk_map[cid] for cid in result_ids if cid in chunk_map]

    def save_graph(self, path: Path) -> None:
        """
        Save graph to GraphML file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_graphml(self.graph, str(path))
        logger.info(f"Graph saved to {path}")

    def load_graph(self, path: Path) -> nx.Graph:
        """
        Load graph from GraphML file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded graph
        """
        self.graph = nx.read_graphml(str(path))
        logger.info(f"Graph loaded from {path}: {self.graph.number_of_nodes()} nodes")
        return self.graph

    def get_graph_stats(self) -> dict:
        """Get graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
        }


def build_graph_from_chunks(
    chunks_path: Path,
    output_path: Path,
    threshold: float = 0.7,
    max_neighbors: int = 10,
) -> nx.Graph:
    """
    Build and save graph from chunks file.
    
    Args:
        chunks_path: Path to chunks JSON with embeddings
        output_path: Output GraphML file path
        threshold: Similarity threshold
        max_neighbors: Maximum neighbors per node
        
    Returns:
        Built graph
    """
    chunks = load_chunks(chunks_path)
    
    builder = KnowledgeGraphBuilder(
        similarity_threshold=threshold,
        max_neighbors=max_neighbors,
    )
    
    graph = builder.build_graph(chunks)
    builder.save_graph(output_path)
    
    stats = builder.get_graph_stats()
    logger.info(f"Graph stats: {stats}")
    
    return graph


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge graph")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--threshold", "-t", type=float, default=0.7)
    parser.add_argument("--max-neighbors", "-n", type=int, default=10)
    
    args = parser.parse_args()
    
    build_graph_from_chunks(
        chunks_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        max_neighbors=args.max_neighbors,
    )
