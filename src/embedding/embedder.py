import json
import logging
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Embedder:
    """
    Embedder encodes text chunks into vector embeddings using Sentence-Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        :param model_name: Pretrained Sentence-Transformers model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = 32
    ) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        :param texts: List of input strings
        :param batch_size: Batch size for model encoding
        :return: Numpy array of shape (len(texts), embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def embed_chunks(
        self,
        chunks: List[Dict],
        text_key: str = "text",
        embedding_key: str = "embedding",
        batch_size: Optional[int] = 32
    ) -> List[Dict]:
        """
        Adds embeddings to each chunk dict under `embedding_key`.

        :param chunks: List of dicts containing text entries
        :param text_key: Key in chunk dict to read text
        :param embedding_key: Key to write embedding vector
        :param batch_size: Batch size for embedding
        :return: List of chunk dicts with embeddings
        """
        texts = [chunk[text_key] for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        for chunk, emb in zip(chunks, embeddings):
            chunk[embedding_key] = emb.tolist()
        return chunks

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed JSON chunks into vector embeddings"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to JSON file containing list of chunk dicts"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file with embeddings added"
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding"
    )
    args = parser.parse_args()

    # Load chunks
    with open(args.input, 'r') as f:
        chunks = json.load(f)

    embedder = Embedder(model_name=args.model)
    enriched = embedder.embed_chunks(
        chunks,
        text_key="text",
        embedding_key="embedding",
        batch_size=args.batch_size
    )

    # Save enriched chunks
    with open(args.output, 'w') as f:
        json.dump(enriched, f, indent=2)
    logger.info(f"Saved {len(enriched)} chunks with embeddings to {args.output}")
