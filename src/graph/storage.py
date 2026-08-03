import logging
from typing import Optional

import networkx as nx
from py2neo import Graph, Node, Relationship

logger = logging.getLogger(__name__)

class GraphStorage:
    """
    GraphStorage handles persisting NetworkX graphs to a Neo4j database
    and loading them back into memory.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: Optional[str] = None
    ):
        """
        :param uri: Neo4j Bolt URI
        :param user: Username for Neo4j
        :param password: Password for Neo4j
        :param database: Optional database name (Neo4j 4.x+)
        """
        auth = (user, password)
        if database:
            self.graph = Graph(uri, auth=auth, name=database)
        else:
            self.graph = Graph(uri, auth=auth)
        logger.info(f"Connected to Neo4j at {uri}, database={database}")

    def clear(self) -> None:
        """Deletes all nodes and relationships in the database."""
        self.graph.delete_all()
        logger.info("Cleared all data from Neo4j database")

    def write_graph(
        self,
        nx_graph: nx.Graph,
        node_label: str = "Chunk",
        rel_type: str = "SIMILAR"
    ) -> None:
        """
        Persists the given NetworkX graph into Neo4j.
        Nodes are labeled with `node_label` and have an 'id' property;
        edges are created with relationship type `rel_type` and a 'weight' property.

        :param nx_graph: NetworkX graph with node attributes and 'weight' on edges
        :param node_label: Label to assign to nodes in Neo4j
        :param rel_type: Relationship type for edges
        """
        tx = self.graph.begin()
        # Create or merge nodes
        for node_id, attrs in nx_graph.nodes(data=True):
            props = {**attrs, "id": node_id}
            node = Node(node_label, **props)
            tx.merge(node, node_label, "id")
        # Create relationships
        for u, v, data in nx_graph.edges(data=True):
            n1 = Node(node_label, id=u)
            n2 = Node(node_label, id=v)
            rel = Relationship(n1, rel_type, n2, weight=data.get("weight"))
            tx.merge(rel)
        tx.commit()
        logger.info(f"Written graph to Neo4j: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")

    def load_graph(
        self,
        node_label: str = "Chunk",
        rel_type: str = "SIMILAR"
    ) -> nx.Graph:
        """
        Loads a graph from Neo4j, reconstructing a NetworkX graph.

        :param node_label: Label of nodes to load
        :param rel_type: Relationship type of edges to load
        :return: NetworkX Graph with node attributes and 'weight' on edges
        """
        G = nx.Graph()
        # Load nodes and properties
        node_query = f"MATCH (n:{node_label}) RETURN n.id AS id, properties(n) AS props"
        for record in self.graph.run(node_query):
            node_id = record["id"]
            props = record["props"]
            # Remove redundant 'id' property
            props.pop("id", None)
            G.add_node(node_id, **props)
        # Load relationships
        rel_query = (
            f"MATCH (a:{node_label})-[r:{rel_type}]-(b:{node_label}) "
            "RETURN a.id AS id1, b.id AS id2, r.weight AS weight"
        )
        for record in self.graph.run(rel_query):
            id1 = record["id1"]
            id2 = record["id2"]
            weight = record.get("weight")
            G.add_edge(id1, id2, weight=weight)
        logger.info(f"Loaded graph from Neo4j: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

if __name__ == '__main__':
    import argparse
    import json
    import networkx as nx

    parser = argparse.ArgumentParser(description='Persist or load a graph to/from Neo4j')
    parser.add_argument('--action', choices=['write', 'load'], required=True,
                        help='Whether to write a JSON graph to Neo4j or load from Neo4j')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (JSON for write, ignored for load)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file (ignored for write, JSON for load)')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--password', type=str, required=True, help='Neo4j password')
    parser.add_argument('--database', type=str, default=None, help='Neo4j database name')
    parser.add_argument('--node_label', type=str, default='Chunk', help='Node label')
    parser.add_argument('--rel_type', type=str, default='SIMILAR', help='Relationship type')
    args = parser.parse_args()

    storage = GraphStorage(uri=args.uri, user=args.user, password=args.password, database=args.database)

    if args.action == 'write':
        with open(args.input, 'r') as f:
            chunks = json.load(f)
        # Convert list of chunks with embeddings to NetworkX graph
        G = nx.Graph()
        for chunk in chunks:
            node_id = chunk['node_id']
            data = {k: v for k, v in chunk.items() if k != 'embedding'}
            G.add_node(node_id, **data)
        # Attempt to add edges if embedded similarity exists
        # This assumes edges are precomputed, else build separately
        # Save graph
        storage.write_graph(G, node_label=args.node_label, rel_type=args.rel_type)
    else:
        G = storage.load_graph(node_label=args.node_label, rel_type=args.rel_type)
        # Serialize nodes and edges
        out = {
            'nodes': [ {'id': n, **G.nodes[n]} for n in G.nodes ],
            'edges': [ {'source': u, 'target': v, **G.edges[u,v]} for u,v in G.edges ]
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Exported graph to {args.output}")
