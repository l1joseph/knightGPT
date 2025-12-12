"""Neo4j storage layer for persistent knowledge graph."""

from pathlib import Path
from typing import Optional

from ..chunking import Chunk, load_chunks
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


class Neo4jStorage:
    """
    Neo4j storage layer for knowledge graph persistence.
    
    Stores chunks and their relationships in Neo4j for
    production-grade graph queries.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required: pip install neo4j")
        
        self.uri = uri or settings.neo4j.uri
        self.user = user or settings.neo4j.user
        self.password = password or settings.neo4j.password
        self.database = database or settings.neo4j.database
        
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        
        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes."""
        with self.driver.session(database=self.database) as session:
            # Index on chunk ID
            session.run(
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
            )
            # Index on source file
            session.run(
                "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source_file)"
            )
            # Full-text index for search
            try:
                session.run("""
                    CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.text]
                """)
            except Exception:
                pass  # May already exist

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def write_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> int:
        """
        Write chunks to Neo4j.
        
        Args:
            chunks: List of chunks to write
            batch_size: Batch size for writes
            
        Returns:
            Number of chunks written
        """
        logger.info(f"Writing {len(chunks)} chunks to Neo4j")
        
        with self.driver.session(database=self.database) as session:
            written = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare batch data
                chunk_data = []
                for chunk in batch:
                    data = {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "section": chunk.section,
                        "token_count": chunk.token_count,
                        "embedding": chunk.embedding,
                    }
                    # Add metadata fields
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            data[f"meta_{key}"] = value
                    chunk_data.append(data)
                
                # Write batch
                session.run("""
                    UNWIND $chunks AS chunk
                    MERGE (c:Chunk {id: chunk.id})
                    SET c += chunk
                """, chunks=chunk_data)
                
                written += len(batch)
                logger.info(f"Written {written}/{len(chunks)} chunks")
        
        return written

    def write_edges(
        self,
        edges: list[tuple[str, str, float]],
        batch_size: int = 500,
    ) -> int:
        """
        Write similarity edges to Neo4j.
        
        Args:
            edges: List of (source_id, target_id, similarity) tuples
            batch_size: Batch size
            
        Returns:
            Number of edges written
        """
        logger.info(f"Writing {len(edges)} edges to Neo4j")
        
        with self.driver.session(database=self.database) as session:
            written = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                
                edge_data = [
                    {"source": s, "target": t, "similarity": sim}
                    for s, t, sim in batch
                ]
                
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (s:Chunk {id: edge.source})
                    MATCH (t:Chunk {id: edge.target})
                    MERGE (s)-[r:SIMILAR_TO]->(t)
                    SET r.similarity = edge.similarity
                """, edges=edge_data)
                
                written += len(batch)
        
        return written

    def find_similar(
        self,
        chunk_id: str,
        limit: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """
        Find similar chunks by graph traversal.
        
        Args:
            chunk_id: Starting chunk ID
            limit: Maximum results
            
        Returns:
            List of (chunk, similarity) tuples
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk {id: $id})-[r:SIMILAR_TO]-(similar:Chunk)
                RETURN similar, r.similarity as similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """, id=chunk_id, limit=limit)
            
            chunks = []
            for record in result:
                node = record["similar"]
                chunk = Chunk(
                    id=node["id"],
                    text=node["text"],
                    source_file=node["source_file"],
                    section=node.get("section"),
                    token_count=node.get("token_count", 0),
                    embedding=node.get("embedding"),
                )
                chunks.append((chunk, record["similarity"]))
            
            return chunks

    def search_fulltext(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Chunk]:
        """
        Full-text search on chunk content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching chunks
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("chunk_text", $query)
                YIELD node, score
                RETURN node, score
                ORDER BY score DESC
                LIMIT $limit
            """, query=query, limit=limit)
            
            chunks = []
            for record in result:
                node = record["node"]
                chunk = Chunk(
                    id=node["id"],
                    text=node["text"],
                    source_file=node["source_file"],
                    section=node.get("section"),
                    token_count=node.get("token_count", 0),
                )
                chunks.append(chunk)
            
            return chunks

    def get_context_subgraph(
        self,
        chunk_ids: list[str],
        hops: int = 1,
    ) -> list[Chunk]:
        """
        Get subgraph around specified chunks.
        
        Args:
            chunk_ids: Starting chunk IDs
            hops: Number of hops to traverse
            
        Returns:
            List of chunks in subgraph
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (start:Chunk)
                WHERE start.id IN $ids
                CALL apoc.path.subgraphNodes(start, {
                    maxLevel: $hops,
                    relationshipFilter: "SIMILAR_TO"
                }) YIELD node
                RETURN DISTINCT node
            """, ids=chunk_ids, hops=hops)
            
            chunks = []
            for record in result:
                node = record["node"]
                chunk = Chunk(
                    id=node["id"],
                    text=node["text"],
                    source_file=node["source_file"],
                    section=node.get("section"),
                    token_count=node.get("token_count", 0),
                    embedding=node.get("embedding"),
                )
                chunks.append(chunk)
            
            return chunks

    def clear_all(self):
        """Delete all data."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all Neo4j data")

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.driver.session(database=self.database) as session:
            node_count = session.run(
                "MATCH (c:Chunk) RETURN count(c) as count"
            ).single()["count"]
            
            edge_count = session.run(
                "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count"
            ).single()["count"]
            
            return {
                "chunks": node_count,
                "edges": edge_count,
            }


def sync_to_neo4j(
    chunks_path: Path,
    graph_path: Optional[Path] = None,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> dict:
    """
    Sync chunks and graph to Neo4j.
    
    Args:
        chunks_path: Path to chunks JSON
        graph_path: Path to graph GraphML (optional)
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
        
    Returns:
        Sync statistics
    """
    import networkx as nx
    
    storage = Neo4jStorage(uri=uri, user=user, password=password)
    
    # Load and write chunks
    chunks = load_chunks(chunks_path)
    chunks_written = storage.write_chunks(chunks)
    
    # Load and write edges from graph
    edges_written = 0
    if graph_path and Path(graph_path).exists():
        graph = nx.read_graphml(str(graph_path))
        edges = [
            (u, v, graph[u][v].get("similarity", 0.5))
            for u, v in graph.edges()
        ]
        edges_written = storage.write_edges(edges)
    
    storage.close()
    
    return {
        "chunks_written": chunks_written,
        "edges_written": edges_written,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo4j storage operations")
    parser.add_argument("action", choices=["write", "stats", "clear"])
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--uri", type=str)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)
    
    args = parser.parse_args()
    
    if args.action == "write":
        if not args.chunks:
            print("--chunks required for write action")
            exit(1)
        
        result = sync_to_neo4j(
            chunks_path=args.chunks,
            graph_path=args.graph,
            uri=args.uri,
            user=args.user,
            password=args.password,
        )
        print(f"Synced: {result}")
        
    elif args.action == "stats":
        storage = Neo4jStorage(
            uri=args.uri,
            user=args.user,
            password=args.password,
        )
        print(storage.get_stats())
        storage.close()
        
    elif args.action == "clear":
        storage = Neo4jStorage(
            uri=args.uri,
            user=args.user,
            password=args.password,
        )
        storage.clear_all()
        storage.close()
