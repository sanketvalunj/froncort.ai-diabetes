from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)
