"""
EmbeddingService — wraps SentenceTransformer with a process-level singleton cache.

The SentenceTransformer model is ~90 MB on disk and ~200 MB in RAM.  Loading it
more than once per process wastes memory and adds latency.  We store one model
instance per model-name in ``_MODEL_CACHE`` so every ``EmbeddingService``
created with the same model name shares the exact same object.
"""

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# Process-level cache: model_name → SentenceTransformer instance.
# Populated lazily on first embed() call; never recreated.
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return the cached model, loading it once if necessary."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Do NOT load the model here — defer until first use.

    @property
    def model(self) -> SentenceTransformer:
        """Lazy accessor — loads and caches the model on first call."""
        return _get_model(self.model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)
