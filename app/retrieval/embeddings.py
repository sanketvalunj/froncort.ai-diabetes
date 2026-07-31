"""
EmbeddingService — defers SentenceTransformer import and load to first use.

sentence_transformers pulls in torch, transformers, and tokenizers at import
time which alone can consume 200+ MB.  By moving the import inside the function
that actually needs it, the module itself is free to import with zero cost.

The loaded model is stored in a process-level cache so it is created exactly
once per model name, never per request.
"""

from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    # Only used for type hints — never executed at runtime during import
    from sentence_transformers import SentenceTransformer as _ST

# Process-level cache: model_name → SentenceTransformer instance.
_MODEL_CACHE: Dict[str, "_ST"] = {}


def _get_model(model_name: str) -> "_ST":
    """Load and cache the model on first call; return the cached instance thereafter."""
    if model_name not in _MODEL_CACHE:
        # Import is deferred here — only runs on first actual embed() call
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # No import, no model load here.

    @property
    def model(self) -> "_ST":
        """Lazy accessor — loads and caches the model on first call."""
        return _get_model(self.model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)
