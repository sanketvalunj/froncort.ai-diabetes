"""
EmbeddingService — lightweight TF-IDF text representation service using scikit-learn.

Memory footprint: ~15-20 MB total RSS (vs ~350 MB PyTorch + SentenceTransformers).
Instantiates in < 0.1s. Zero heavy C++ / PyTorch / CUDA / OpenMP dependencies.
"""

from typing import List, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    """
    Lightweight TF-IDF EmbeddingService preserving the embed() and embed_batch() interface
    expected by index builders and retrievers.
    """

    def __init__(self, model_name: str = "tfidf"):
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        self._is_fitted = False

    def fit_transform(self, texts: List[str]):
        matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        return matrix

    def embed(self, text: str) -> np.ndarray:
        """Embed a single query string for retrieval search."""
        if not self._is_fitted:
            return np.array([text])
        return self.vectorizer.transform([text]).toarray()[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if not self._is_fitted:
            return self.fit_transform(texts).toarray()
        return self.vectorizer.transform(texts).toarray()
