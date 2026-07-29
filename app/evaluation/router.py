from typing import List

from app.evaluation.llm_engine import LLMEvaluator
from app.evaluation.rule_engine import RuleEngine
from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence
from app.models.patient import Patient
from app.models.trial import Criterion, CriterionType

_RULE_TYPES = {CriterionType.AGE, CriterionType.HBA1C,
               CriterionType.EGFR, CriterionType.RECRUITING}
_LLM_TYPES  = {CriterionType.MEDICATION, CriterionType.CONDITION}


class EvaluatorRouter:
    def __init__(self, settings, llm_client):
        self._rule_engine   = RuleEngine(settings)
        self._llm_evaluator = LLMEvaluator(settings, llm_client)

    def evaluate(
        self,
        criterion: Criterion,
        patient: Patient,
        patient_evidence: List[Evidence],
        trial_evidence: List[Evidence],
    ) -> CriterionEvaluation:
        if criterion.type in _RULE_TYPES:
            return self._rule_engine.evaluate(criterion, patient)

        if criterion.type in _LLM_TYPES:
            return self._llm_evaluator.evaluate(
                criterion, patient, patient_evidence + trial_evidence
            )

        # CriterionType.OTHER — cannot be automated; flag for clinical review
        return CriterionEvaluation(
            criterion_id=criterion.id,
            status=CriterionStatus.REQUIRES_CLINICAL_REVIEW,
            reasoning="Criterion type OTHER requires clinical judgement and cannot be automated.",
            evidence_used=[],
            confidence=0.0,
            evaluator_type="clinical_review",
            unanswered_questions=[
                "Please review this criterion manually with the study team."
            ],
        )
