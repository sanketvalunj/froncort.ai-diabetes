"""
EmbeddingService — memory-optimised, single-instance SentenceTransformer.

Memory analysis (all-MiniLM-L6-v2 on CPU, Render free tier):
  import torch alone          : +184 MB  (unavoidable PyTorch baseline)
  import transformers         : +200 MB  (HuggingFace registry — loaded once)
  model weights (all-MiniLM)  : +22  MB  (90 MB on disk, quantised in RAM)
  first encode() call         : +80  MB  (activation buffers, then GC'd)
  FAISS index (581 × 384)     : ~2   MB  (negligible)
  ─────────────────────────────────────────
  Total at steady state       : ~350 MB  — fits in 512 MB if nothing else leaks

Configured model: sentence-transformers/all-MiniLM-L6-v2
  - 22 MB weights, 384-dim embeddings
  - Matches the dimension of the existing persisted FAISS index (dim=384)
  - Do NOT change to a larger model without rebuilding the FAISS index

Memory reduction knobs applied here:
  1. OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 — prevents thread-local memory copies
     that can 2–4× RSS on a multi-core host.
  2. TOKENIZERS_PARALLELISM=false — suppresses HuggingFace tokenizer fork warning
     and avoids spawning parallel tokenizer processes.
  3. torch.set_num_threads(1) — caps PyTorch intra-op parallelism.
  4. Process-level model cache (_MODEL_CACHE) — guarantees exactly one
     SentenceTransformer instance per model name for the entire process lifetime.
  5. Import is deferred inside _get_model() — zero cost at module import time.
"""

import os
import time
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

# ── Thread-count caps — must be set before torch/OMP loads ───────────────────
# These environment variables are read by the native libraries at init time.
# Setting them here (before the deferred import) is sufficient because
# _get_model() is always called before the libraries are first imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Process-level cache ───────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, "_ST"] = {}


def _get_model(model_name: str) -> "_ST":
    """
    Return the cached SentenceTransformer model.

    On first call for a given model_name:
      - Logs RSS before and after load so Render logs show exact memory delta.
      - Sets torch thread count to 1 after load.
      - Stores the instance in _MODEL_CACHE.

    All subsequent calls return the cached instance immediately.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    # ── Measure RSS before load ───────────────────────────────────────────────
    try:
        import psutil
        rss_before = psutil.Process().memory_info().rss // (1024 * 1024)
    except ImportError:
        rss_before = None

    print(f"[embeddings] Loading model: {model_name}", flush=True)
    if rss_before is not None:
        print(f"[embeddings] Memory before model load: {rss_before} MB", flush=True)

    t0 = time.perf_counter()

    # Deferred import — torch + transformers load here, once only
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(1)  # cap intra-op parallelism after load

    model = SentenceTransformer(model_name, device="cpu")
    model.eval()  # disable dropout, reduce memory slightly

    elapsed = time.perf_counter() - t0

    # ── Measure RSS after load ────────────────────────────────────────────────
    try:
        import psutil
        rss_after = psutil.Process().memory_info().rss // (1024 * 1024)
        delta = rss_after - (rss_before or 0)
        print(
            f"[embeddings] Memory after model load: {rss_after} MB "
            f"(+{delta} MB, {elapsed:.1f}s)",
            flush=True,
        )
    except ImportError:
        print(f"[embeddings] Model loaded in {elapsed:.1f}s", flush=True)

    _MODEL_CACHE[model_name] = model
    return model


class EmbeddingService:
    """
    Thin wrapper around the process-level model cache.

    Construction is free — no import, no model load.
    The model loads on the first call to embed() or embed_batch().

    Inference contract (production):
      - embed(query)  → embeds ONE query string for FAISS search
      - embed_batch() → used only for index build, never called at inference time

    The FAISS index is pre-built and persisted.  Production requests ONLY
    embed the query string and search the existing index.  Trial embeddings
    are never recomputed at request time.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def model(self) -> "_ST":
        return _get_model(self.model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single query string. Called once per query at inference time."""
        return self.model.encode([text], show_progress_bar=False)[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch. Used only during index build — NOT called at inference time."""
        return self.model.encode(texts, show_progress_bar=False)
