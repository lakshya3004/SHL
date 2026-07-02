import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from app.models.retrieval_models import SearchResult


class VectorStore:
    """
    Manages the FAISS vector index for semantic retrieval.
    Maps vector IDs back to assessment metadata.
    Handles missing index gracefully (returns empty results, no crash).
    """
    
    def __init__(
        self,
        index_path: str = "data/vectorstore/faiss.index",
        metadata_path: str = "data/vectorstore/metadata.json"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Creates a new FAISS index from a set of embeddings.
        Uses normalized Inner Product (equivalent to cosine similarity).
        """
        dimension = embeddings.shape[1]
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div by zero
        normalized = (embeddings / norms).astype("float32")
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized)
        self.metadata = metadata
        logger.info(f"Built FAISS index with {len(metadata)} documents.")

    def save_index(self):
        """Persists the index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save.")
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index to {self.index_path}")

    def load_index(self) -> bool:
        """Loads index and metadata from disk. Returns True on success."""
        if not os.path.exists(self.index_path):
            logger.warning(f"Index file not found: {self.index_path}")
            return False

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index with {len(self.metadata)} documents.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        """
        Performs semantic search using vector similarity.
        Returns empty list if index is not loaded (graceful degradation).
        """
        if self.index is None:
            logger.warning("Vector index not loaded; returning empty results.")
            return []

        if len(self.metadata) == 0:
            return []

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        actual_k = min(k, len(self.metadata))
        
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append(SearchResult(
                assessment_id=meta.get("id", str(idx)),
                name=meta.get("name", "Unknown"),
                score=float(score),
                metadata=meta
            ))

        return results

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.metadata) > 0
