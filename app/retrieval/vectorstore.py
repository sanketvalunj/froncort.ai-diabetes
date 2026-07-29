import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index = None
        self.metadata: List[Dict] = []

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata

    def save(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "trial_index.faiss"))
        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path / "trial_index.faiss"))
        with open(self.index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype("float32"), top_k
        )
        results: List[Tuple[Dict, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        return results
