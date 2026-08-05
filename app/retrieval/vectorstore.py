# Manages the FAISS vector store for semantic search.
"""
FAISSVectorStore — lightweight TF-IDF + Cosine Similarity vector store using scikit-learn.

Retains the FAISSVectorStore class name and API (build, save, load, search)
so that EvidenceRetriever, scripts, and tests work without modification.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAISSVectorStore:
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        self.matrix = None
        self.metadata: List[Dict] = []

    def build(self, embeddings: Any, metadata: List[Dict]) -> None:
        self.metadata = metadata
        texts = [m.get("text", "") for m in metadata]
        if texts:
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def save(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.index_path / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.matrix, f)
        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        with open(self.index_path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.index_path / "tfidf_matrix.pkl", "rb") as f:
            self.matrix = pickle.load(f)
        with open(self.index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_input: Any, top_k: int) -> List[Tuple[Dict, float]]:
        if self.matrix is None or not self.metadata or self.matrix.shape[0] == 0:
            return []

        if isinstance(query_input, str):
            q_vec = self.vectorizer.transform([query_input])
        elif isinstance(query_input, np.ndarray) and query_input.dtype.kind in ("U", "S", "O"):
            q_vec = self.vectorizer.transform([str(x) for x in query_input])
        elif hasattr(query_input, "shape"):
            q_arr = np.asarray(query_input)
            if q_arr.ndim == 1:
                q_arr = q_arr.reshape(1, -1)
            if q_arr.shape[1] == self.matrix.shape[1]:
                q_vec = q_arr
            else:
                q_vec = self.vectorizer.transform([str(query_input)])
        else:
            q_vec = self.vectorizer.transform([str(query_input)])

        similarities = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Tuple[Dict, float]] = []
        for idx in top_indices:
            if 0 <= idx < len(self.metadata):
                sim = float(similarities[idx])
                dist = (1.0 - sim) / (sim + 1e-6) if sim > 0 else 1000.0
                results.append((self.metadata[idx], dist))

        return results
