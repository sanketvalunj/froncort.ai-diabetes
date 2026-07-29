from app.models.state import AgentState
from app.retrieval.retriever import EvidenceRetriever
from app.utils.logger import get_logger

log = get_logger(__name__)


class RetrievalService:
    def __init__(self, settings):
        self._settings  = settings
        self._retriever = EvidenceRetriever(settings)

    def run(self, state: AgentState) -> AgentState:
        log.info("retrieval_start", trials=len(state.filtered_trials),
                 trace_id=state.trace_id)
        patient_evidence = self._retriever.extract_patient_evidence(state.patient)
        queries          = self._retriever.build_queries(state.patient)
        log.info("retrieval_queries_built", count=len(queries),
                 trace_id=state.trace_id)
        trial_evidence = {}
        for trial in state.filtered_trials:
            evidence = self._retriever.retrieve_for_trial(trial.id, queries)
            trial_evidence[trial.id] = evidence
            log.info("retrieval_trial_done", trial_id=trial.id,
                     evidence_count=len(evidence), trace_id=state.trace_id)
        log.info("retrieval_complete",
                 patient_evidence_count=len(patient_evidence),
                 trace_id=state.trace_id)
        return state.model_copy(update={"patient_evidence": patient_evidence,
                                        "trial_evidence": trial_evidence})
