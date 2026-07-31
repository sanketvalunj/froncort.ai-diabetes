"""
RetrievalService — Stage 2 of the pipeline.

The EvidenceRetriever (which owns both EmbeddingService and FAISSVectorStore) is
expensive to construct: first use loads the SentenceTransformer model (~200 MB)
and reads the persisted FAISS index from disk.

We keep one EvidenceRetriever per (vector_store_path, embedding_model) combination
in a module-level cache so the model and index are loaded exactly once for the
lifetime of the process, regardless of how many requests arrive or how many times
RetrievalService is instantiated.
"""

from typing import Dict

from app.models.state import AgentState
from app.retrieval.retriever import EvidenceRetriever
from app.utils.logger import get_logger

log = get_logger(__name__)

# Module-level cache: cache_key → EvidenceRetriever
# Populated lazily on first run(); never recreated.
_RETRIEVER_CACHE: Dict[str, EvidenceRetriever] = {}


def _get_retriever(settings) -> EvidenceRetriever:
    """Return the cached EvidenceRetriever, creating it once if needed."""
    cache_key = f"{settings.paths.vector_store}|{settings.embeddings.model}"
    if cache_key not in _RETRIEVER_CACHE:
        log.info("retriever_init", vector_store=str(settings.paths.vector_store),
                 embedding_model=settings.embeddings.model)
        _RETRIEVER_CACHE[cache_key] = EvidenceRetriever(settings)
    return _RETRIEVER_CACHE[cache_key]


class RetrievalService:
    def __init__(self, settings):
        self._settings = settings
        # Do NOT construct EvidenceRetriever here — defer to first run() call
        # so neither the SentenceTransformer nor the FAISS index loads at
        # service-construction time (which happens on every /screen request
        # inside make_nodes/build_workflow).

    def run(self, state: AgentState) -> AgentState:
        retriever = _get_retriever(self._settings)

        log.info("retrieval_start", trials=len(state.filtered_trials),
                 trace_id=state.trace_id)
        patient_evidence = retriever.extract_patient_evidence(state.patient)
        queries          = retriever.build_queries(state.patient)
        log.info("retrieval_queries_built", count=len(queries),
                 trace_id=state.trace_id)

        trial_evidence = {}
        for trial in state.filtered_trials:
            evidence = retriever.retrieve_for_trial(trial.id, queries)
            trial_evidence[trial.id] = evidence
            log.info("retrieval_trial_done", trial_id=trial.id,
                     evidence_count=len(evidence), trace_id=state.trace_id)

        log.info("retrieval_complete",
                 patient_evidence_count=len(patient_evidence),
                 trace_id=state.trace_id)
        return state.model_copy(update={"patient_evidence": patient_evidence,
                                        "trial_evidence": trial_evidence})
