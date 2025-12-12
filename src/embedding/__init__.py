"""Embedding generation modules."""

from .embedder import VLLMEmbedder, embed_chunks_file

__all__ = [
    "VLLMEmbedder",
    "embed_chunks_file",
]
