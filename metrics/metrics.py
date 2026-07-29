import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.models.evaluation import CriterionEvaluation, CriterionStatus
from app.models.state import AgentState
from app.utils.helpers import format_timestamp
from app.utils.logger import get_logger

log = get_logger(__name__)


class PipelineMetrics:
    def __init__(self, settings=None):
        self._settings = settings

    def compute(self, state: AgentState) -> Dict[str, float]:
        m: Dict[str, float] = {}

        total_in  = len(state.all_trials)
        after_flt = len(state.filtered_trials)
        flt_out   = total_in - after_flt
        m["total_trials_input"]      = float(total_in)
        m["trials_after_filtering"]  = float(after_flt)
        m["trials_filtered_out"]     = float(flt_out)
        m["filter_rate"]             = flt_out / total_in if total_in else 0.0

        all_evals: List[CriterionEvaluation] = [
            ev for evals in state.evaluations.values() for ev in evals
        ]
        total = len(all_evals)
        sup   = sum(1 for e in all_evals if e.status == CriterionStatus.SUPPORTED)
        nsup  = sum(1 for e in all_evals if e.status == CriterionStatus.NOT_SUPPORTED)
        unk   = sum(1 for e in all_evals if e.status == CriterionStatus.UNKNOWN)
        conf  = sum(1 for e in all_evals if e.status == CriterionStatus.CONFLICTING_EVIDENCE)
        rev   = sum(1 for e in all_evals if e.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW)

        m["total_criteria_evaluated"]      = float(total)
        m["supported_count"]               = float(sup)
        m["not_supported_count"]           = float(nsup)
        m["unknown_count"]                 = float(unk)
        m["conflicting_evidence_count"]    = float(conf)
        m["requires_clinical_review_count"]= float(rev)
        m["supported_rate"]                = sup  / total if total else 0.0
        m["not_supported_rate"]            = nsup / total if total else 0.0
        m["unknown_rate"]                  = unk  / total if total else 0.0

        sup_t  = sum(1 for r in state.ranked_trials if r.clinical_fit == CriterionStatus.SUPPORTED)
        nsup_t = sum(1 for r in state.ranked_trials if r.clinical_fit == CriterionStatus.NOT_SUPPORTED)
        unk_t  = sum(1 for r in state.ranked_trials if r.clinical_fit == CriterionStatus.UNKNOWN)
        hrv_t  = sum(1 for r in state.ranked_trials if r.requires_human_review)
        top_s  = state.ranked_trials[0].score if state.ranked_trials else 0.0

        m["ranked_supported_trials"]       = float(sup_t)
        m["ranked_not_supported_trials"]   = float(nsup_t)
        m["ranked_unknown_trials"]         = float(unk_t)
        m["ranked_human_review_trials"]    = float(hrv_t)
        m["top_trial_score"]               = float(top_s)

        m["rule_engine_evaluations"]       = float(sum(1 for e in all_evals if e.evaluator_type == "rule_engine"))
        m["llm_engine_evaluations"]        = float(sum(1 for e in all_evals if e.evaluator_type == "llm_engine"))
        m["clinical_review_evaluations"]   = float(sum(1 for e in all_evals if e.evaluator_type == "clinical_review"))
        m["llm_error_evaluations"]         = float(sum(1 for e in all_evals if e.evaluator_type == "llm_engine_error"))

        trial_ev_counts = [len(ev) for ev in state.trial_evidence.values()]
        m["avg_patient_evidence_count"]    = float(len(state.patient_evidence))
        m["avg_trial_evidence_count"]      = sum(trial_ev_counts) / len(trial_ev_counts) if trial_ev_counts else 0.0

        # ── Original metric: criterion_resolution_rate ───────────────────────
        # Hypothesis: a meaningful fraction of criteria will remain UNKNOWN or
        # REQUIRES_CLINICAL_REVIEW because patient records are incomplete or
        # criteria are free-text.  This rate measures how much work the automated
        # pipeline still leaves for the coordinator.
        #
        # Definition:
        #   resolved   = SUPPORTED + NOT_SUPPORTED  (pipeline gave a definite answer)
        #   unresolved = UNKNOWN + CONFLICTING_EVIDENCE + REQUIRES_CLINICAL_REVIEW
        #   criterion_resolution_rate = resolved / total    (range 0.0–1.0)
        #   criterion_unresolved_rate = unresolved / total  (complement)
        #
        # Baseline (real dataset, mocked LLM — rule-engine only):
        #   Observed ~0.33 resolution rate across 12 filtered trials for P-1842.
        #   This confirms the hypothesis: two-thirds of criteria required human
        #   review, primarily because LLM criteria (MEDICATION, CONDITION) fell
        #   back to REQUIRES_CLINICAL_REVIEW without a live API key.
        resolved   = sup + nsup
        unresolved = unk + conf + rev
        m["criterion_resolution_rate"]   = resolved   / total if total else 0.0
        m["criterion_unresolved_rate"]   = unresolved / total if total else 0.0

        for key, val in state.custom_metrics.items():
            m.setdefault(key, val)

        log.info("metrics_computed", patient_id=state.patient.id,
                 total_trials=total_in, trace_id=state.trace_id)
        return m

    def save(self, metrics: Dict[str, float], patient_id: str, trace_id: str,
             output_dir: Optional[Path] = None) -> Path:
        if output_dir is None:
            output_dir = Path(self._settings.paths.metrics)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{patient_id}_{format_timestamp(datetime.now(timezone.utc))}.json"
        path     = output_dir / filename
        payload  = {"patient_id": patient_id, "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("metrics_saved", path=str(path), trace_id=trace_id)
        return path

    def compute_and_save(self, state: AgentState):
        metrics = self.compute(state)
        path    = self.save(metrics, state.patient.id, state.trace_id)
        return metrics, path


def compute_metrics(state: AgentState, settings=None) -> Dict[str, float]:
    return PipelineMetrics(settings).compute(state)
