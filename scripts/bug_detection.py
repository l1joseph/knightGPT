#!/usr/bin/env python3
"""
Comprehensive bug detection script for KnightGPT.

Systematically tests all modules for:
- Null/None checks
- Index out of bounds
- Division by zero
- Missing error handling
- Resource leaks
- Type mismatches
- Edge cases
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.chunking import Chunk, SemanticChunker, load_chunks, save_chunks
from src.embedding import VLLMEmbedder
from src.graph import KnowledgeGraphBuilder
from src.retrieval import GraphRAGRetriever, RAGEngine
from src.storage import Neo4jStorage
from src.utils import get_logger, get_settings

logger = get_logger(__name__)


class BugDetector:
    """Systematic bug detection across all modules."""

    def __init__(self):
        self.bugs = []
        self.warnings = []
        self.test_results = {}

    def report_bug(self, module: str, location: str, issue: str, severity: str = "HIGH"):
        """Report a bug finding."""
        bug = {
            "module": module,
            "location": location,
            "issue": issue,
            "severity": severity,
        }
        self.bugs.append(bug)
        print(f"[{severity}] {module}:{location} - {issue}")

    def report_warning(self, module: str, location: str, issue: str):
        """Report a warning."""
        warning = {
            "module": module,
            "location": location,
            "issue": issue,
        }
        self.warnings.append(warning)
        print(f"[WARNING] {module}:{location} - {issue}")

    def test_chunking_module(self):
        """Test chunking module for bugs."""
        print("\n=== Testing Chunking Module ===")
        module = "chunking"

        try:
            # Test 1: Empty text
            chunker = SemanticChunker()
            chunks = chunker.chunk_text("", "test.md")
            if len(chunks) != 0:
                self.report_bug(module, "chunk_text", "Empty text should return empty chunks")

            # Test 2: Very long text
            long_text = "A" * 100000
            chunks = chunker.chunk_text(long_text, "test.md")
            if not chunks:
                self.report_bug(module, "chunk_text", "Very long text should still be chunked")

            # Test 3: None/None inputs
            try:
                chunker.chunk_text(None, "test.md")
                self.report_bug(module, "chunk_text", "Should handle None text input")
            except (TypeError, AttributeError):
                pass  # Expected

            # Test 4: Invalid file path
            try:
                chunker.chunk_markdown_file(Path("/nonexistent/file.md"))
                self.report_bug(module, "chunk_markdown_file", "Should handle non-existent files")
            except FileNotFoundError:
                pass  # Expected

            # Test 5: Chunk ID collision risk
            text1 = "Test text 1"
            text2 = "Test text 2"
            chunks1 = chunker.chunk_text(text1, "file1.md")
            chunks2 = chunker.chunk_text(text2, "file2.md")
            ids1 = {c.id for c in chunks1}
            ids2 = {c.id for c in chunks2}
            if ids1 & ids2:
                self.report_warning(module, "_create_chunk", "Potential ID collision detected")

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"

    def test_embedding_module(self):
        """Test embedding module for bugs."""
        print("\n=== Testing Embedding Module ===")
        module = "embedding"

        try:
            # Test 1: Empty text list
            embedder = VLLMEmbedder(api_base="http://invalid:8001/v1")
            try:
                result = embedder.embed_batch([])
                if result != []:
                    self.report_bug(module, "embed_batch", "Empty list should return empty list")
            except Exception:
                pass  # May fail due to invalid URL, which is expected

            # Test 2: None text
            try:
                embedder.embed_text(None)
                self.report_bug(module, "embed_text", "Should handle None input")
            except (TypeError, AttributeError):
                pass  # Expected

            # Test 3: Health check with invalid server
            health = embedder.check_health()
            if health:
                self.report_warning(module, "check_health", "Health check should fail for invalid server")

            # Test 4: Batch with None values
            try:
                embedder.embed_batch([None, "test"])
                self.report_bug(module, "embed_batch", "Should handle None in batch")
            except (TypeError, AttributeError):
                pass  # Expected

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"

    def test_graph_module(self):
        """Test graph builder module for bugs."""
        print("\n=== Testing Graph Module ===")
        module = "graph"

        try:
            # Test 1: Empty chunks
            builder = KnowledgeGraphBuilder()
            graph = builder.build_graph([])
            if graph.number_of_nodes() != 0:
                self.report_bug(module, "build_graph", "Empty chunks should create empty graph")

            # Test 2: Chunks without embeddings
            chunks_no_emb = [
                Chunk(id="1", text="Test", source_file="test.md"),
                Chunk(id="2", text="Test2", source_file="test.md"),
            ]
            graph = builder.build_graph(chunks_no_emb)
            if graph.number_of_nodes() != 0:
                self.report_warning(module, "build_graph", "Chunks without embeddings should be filtered")

            # Test 3: Division by zero in stats
            stats = builder.get_graph_stats()
            if "avg_degree" in stats and stats["avg_degree"] != 0:
                self.report_bug(module, "get_graph_stats", "Empty graph avg_degree should be 0")

            # Test 4: Invalid node ID in get_neighbors
            neighbors = builder.get_neighbors("nonexistent", hops=1)
            if neighbors:
                self.report_bug(module, "get_neighbors", "Non-existent node should return empty list")

            # Test 5: Invalid embedding dimensions
            chunk_bad_emb = Chunk(
                id="1",
                text="Test",
                source_file="test.md",
                embedding=[1.0, 2.0],  # Too short
            )
            chunk_good_emb = Chunk(
                id="2",
                text="Test2",
                source_file="test.md",
                embedding=[1.0] * 768,  # Normal size
            )
            try:
                builder.build_graph([chunk_bad_emb, chunk_good_emb])
                self.report_warning(module, "build_graph", "Should handle mismatched embedding dimensions")
            except Exception:
                pass  # May fail, which is acceptable

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"
            traceback.print_exc()

    def test_retrieval_module(self):
        """Test retrieval module for bugs."""
        print("\n=== Testing Retrieval Module ===")
        module = "retrieval"

        try:
            # Test 1: Empty chunks list
            retriever = GraphRAGRetriever(chunks=[])
            result = retriever.retrieve("test query")
            if result.chunks:
                self.report_bug(module, "retrieve", "Empty chunks should return empty result")

            # Test 2: None query
            try:
                retriever.retrieve(None)
                self.report_bug(module, "retrieve", "Should handle None query")
            except (TypeError, AttributeError):
                pass  # Expected

            # Test 3: Mismatched chunks and scores
            chunks = [
                Chunk(id="1", text="Test", source_file="test.md", embedding=[1.0] * 768),
            ]
            retriever = GraphRAGRetriever(chunks=chunks)
            result = retriever.retrieve("test")
            if len(result.chunks) != len(result.similarity_scores):
                self.report_bug(
                    module,
                    "retrieve",
                    "Chunks and similarity_scores should have same length",
                )

            # Test 4: Format context with empty chunks
            context = retriever.format_context([])
            if context != "":
                self.report_bug(module, "format_context", "Empty chunks should return empty context")

            # Test 5: Create citations with mismatched lengths
            try:
                citations = retriever.create_citations([chunks[0]], [0.5, 0.6])  # Mismatched
                self.report_bug(module, "create_citations", "Should handle mismatched lengths")
            except Exception:
                pass  # May fail, which is acceptable

            # Test 6: IndexError in expand_context
            retriever.graph_hops = 1
            result = retriever.retrieve("test", expand_context=True)
            # Should not raise IndexError

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"
            traceback.print_exc()

    def test_api_module(self):
        """Test API module for bugs."""
        print("\n=== Testing API Module ===")
        module = "api"

        # Note: We can't fully test FastAPI without running server,
        # but we can check for obvious issues in the code

        api_file = Path(__file__).parent.parent / "src" / "api" / "main.py"
        if api_file.exists():
            content = api_file.read_text()

            # Check for potential bugs
            if "_retriever.graph_builder" in content and "_retriever is None" not in content:
                # Check if there's proper None checking
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "graph_builder.graph" in line and i > 180:
                        # Check if _retriever is checked before this
                        prev_lines = "\n".join(lines[max(0, i - 10):i])
                        if "_retriever" in prev_lines and "if _retriever" not in prev_lines:
                            self.report_bug(
                                module,
                                f"line {i}",
                                "Accessing graph_builder without checking if _retriever is None",
                            )

        self.test_results[module] = "PASSED"

    def test_storage_module(self):
        """Test storage module for bugs."""
        print("\n=== Testing Storage Module ===")
        module = "storage"

        try:
            # Test 1: Invalid Neo4j connection
            try:
                storage = Neo4jStorage(uri="bolt://invalid:7687", user="test", password="test")
                storage.close()
                self.report_warning(module, "__init__", "Should validate connection on init")
            except Exception:
                pass  # Expected to fail

            # Test 2: Write empty chunks
            # Can't test without real connection, but check code logic

            # Test 3: Invalid chunk data
            # Check if metadata handling is safe
            storage_file = Path(__file__).parent.parent / "src" / "storage" / "storage.py"
            if storage_file.exists():
                content = storage_file.read_text()
                if "chunk.metadata.items()" in content:
                    # Check if there's type checking
                    if "isinstance(value" not in content:
                        self.report_warning(
                            module,
                            "write_chunks",
                            "Should validate metadata value types",
                        )

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"

    def test_file_operations(self):
        """Test file I/O operations for bugs."""
        print("\n=== Testing File Operations ===")
        module = "file_io"

        try:
            # Test 1: Load non-existent chunks file
            try:
                chunks = load_chunks(Path("/nonexistent/chunks.json"))
                self.report_bug(module, "load_chunks", "Should handle non-existent file")
            except FileNotFoundError:
                pass  # Expected

            # Test 2: Load invalid JSON
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write("invalid json {")
                temp_path = Path(f.name)

            try:
                chunks = load_chunks(temp_path)
                self.report_bug(module, "load_chunks", "Should handle invalid JSON")
            except json.JSONDecodeError:
                pass  # Expected
            finally:
                temp_path.unlink(missing_ok=True)

            # Test 3: Save chunks with None values
            chunks = [
                Chunk(id="1", text="Test", source_file="test.md", embedding=None),
            ]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                temp_path = Path(f.name)

            try:
                save_chunks(chunks, temp_path)
                # Should not raise exception
                loaded = load_chunks(temp_path)
                if len(loaded) != 1:
                    self.report_bug(module, "save_chunks/load_chunks", "Should preserve chunks with None embeddings")
            except Exception as e:
                self.report_bug(module, "save_chunks", f"Should handle None embeddings: {e}")
            finally:
                temp_path.unlink(missing_ok=True)

            self.test_results[module] = "PASSED"
        except Exception as e:
            self.report_bug(module, "general", f"Exception during testing: {e}")
            self.test_results[module] = "FAILED"

    def run_all_tests(self):
        """Run all bug detection tests."""
        print("=" * 60)
        print("KnightGPT Bug Detection Report")
        print("=" * 60)

        self.test_chunking_module()
        self.test_embedding_module()
        self.test_graph_module()
        self.test_retrieval_module()
        self.test_api_module()
        self.test_storage_module()
        self.test_file_operations()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Bugs found: {len(self.bugs)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"\nTest Results: {self.test_results}")

        return {
            "bugs": self.bugs,
            "warnings": self.warnings,
            "test_results": self.test_results,
        }

    def save_report(self, output_path: Path):
        """Save bug report to JSON file."""
        report = {
            "bugs": self.bugs,
            "warnings": self.warnings,
            "test_results": self.test_results,
            "summary": {
                "total_bugs": len(self.bugs),
                "total_warnings": len(self.warnings),
                "high_severity": len([b for b in self.bugs if b["severity"] == "HIGH"]),
                "medium_severity": len([b for b in self.bugs if b["severity"] == "MEDIUM"]),
                "low_severity": len([b for b in self.bugs if b["severity"] == "LOW"]),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="KnightGPT Bug Detection")
    parser.add_argument("--output", "-o", type=Path, default=Path("bug_report.json"))
    args = parser.parse_args()

    detector = BugDetector()
    detector.run_all_tests()
    detector.save_report(args.output)

    # Exit with error code if bugs found
    if detector.bugs:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

