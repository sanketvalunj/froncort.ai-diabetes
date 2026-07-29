from typing import Dict, List

from app.evaluation.router import EvaluatorRouter
from app.models.evaluation import CriterionEvaluation
from app.models.state import AgentState
from app.utils.logger import get_logger

log = get_logger(__name__)


class EvaluationService:
    def __init__(self, settings, llm_client):
        self._settings = settings
        self._router   = EvaluatorRouter(settings, llm_client)

    def run(self, state: AgentState) -> AgentState:
        log.info("evaluation_start", trials=len(state.filtered_trials),
                 trace_id=state.trace_id)
        evaluations: Dict[str, List[CriterionEvaluation]] = {}
        for trial in state.filtered_trials:
            trial_evals = []
            trial_ev    = state.trial_evidence.get(trial.id, [])
            for criterion in trial.inclusion_criteria + trial.exclusion_criteria:
                result = self._router.evaluate(
                    criterion, state.patient, state.patient_evidence, trial_ev)
                trial_evals.append(result)
                log.debug("evaluation_criterion", trial_id=trial.id,
                          criterion_id=criterion.id, status=result.status.value,
                          evaluator=result.evaluator_type, trace_id=state.trace_id)
            evaluations[trial.id] = trial_evals
            log.info("evaluation_trial_done", trial_id=trial.id,
                     criteria_count=len(trial_evals), trace_id=state.trace_id)
        log.info("evaluation_complete",
                 total_trials_evaluated=len(evaluations), trace_id=state.trace_id)
        return state.model_copy(update={"evaluations": evaluations})
