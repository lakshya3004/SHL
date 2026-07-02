import pickle
import os
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from app.models.retrieval_models import SearchResult


class KeywordSearchEngine:
    """
    Handles keyword-based retrieval using TF-IDF.
    Provides sparse retrieval to complement semantic search, ensuring 
    exact skill and product name matches are captured.
    """
    
    def __init__(self, model_path: str = "data/vectorstore/tfidf.pkl"):
        self.model_path = model_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.metadata: List[Dict[str, Any]] = []

    def fit(self, texts: List[str], metadata: List[Dict[str, Any]]):
        """Trains the TF-IDF vectorizer on the catalog corpus."""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2), # Capture multi-word terms like 'numerical reasoning'
            min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.metadata = metadata
        logger.info(f"Fitted TF-IDF engine with {len(texts)} documents.")

    def save(self):
        """Saves the TF-IDF model and matrix."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'matrix': self.matrix,
                'metadata': self.metadata
            }, f)
        logger.info(f"Saved TF-IDF model to {self.model_path}")

    def load(self) -> bool:
        """Loads the TF-IDF model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"TF-IDF model not found: {self.model_path}")
            return False
            
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.matrix = data['matrix']
            self.metadata = data['metadata']
            
        logger.info("Loaded TF-IDF model.")
        return True

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Performs keyword search using TF-IDF cosine similarity.
        """
        if self.vectorizer is None:
            if not self.load():
                return []
                
        query_vec = self.vectorizer.transform([query])
        # Compute cosine similarity
        similarities = (self.matrix * query_vec.T).toarray().flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score <= 0:
                continue
                
            meta = self.metadata[idx]
            results.append(SearchResult(
                assessment_id=meta.get("id", str(idx)),
                name=meta.get("name", "Unknown"),
                score=float(score),
                metadata=meta
            ))
            
        return results
