"""
LLM-based criterion evaluator.

Handles: MEDICATION, CONDITION (and anything else routed to it).
Maps LLM response status strings to the assignment's CriterionStatus vocabulary.
Falls back to UNKNOWN on parse errors, REQUIRES_CLINICAL_REVIEW on other errors.
"""

from typing import List

from config.prompts import CRITERION_EVAL_PROMPT
from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence
from app.models.patient import Patient
from app.models.trial import Criterion

_EVALUATOR_TYPE       = "llm_engine"
_ERROR_EVALUATOR_TYPE = "llm_engine_error"

# Map the LLM response vocabulary → CriterionStatus
_STATUS_MAP = {
    "SUPPORTED":                CriterionStatus.SUPPORTED,
    "NOT_SUPPORTED":            CriterionStatus.NOT_SUPPORTED,
    "UNKNOWN":                  CriterionStatus.UNKNOWN,
    "CONFLICTING_EVIDENCE":     CriterionStatus.CONFLICTING_EVIDENCE,
    "REQUIRES_CLINICAL_REVIEW": CriterionStatus.REQUIRES_CLINICAL_REVIEW,
    # Graceful aliases the LLM may produce
    "ELIGIBLE":                 CriterionStatus.SUPPORTED,
    "INELIGIBLE":               CriterionStatus.NOT_SUPPORTED,
    "REQUIRES_REVIEW":          CriterionStatus.REQUIRES_CLINICAL_REVIEW,
}


class LLMEvaluator:
    def __init__(self, settings, llm_client):
        self._settings   = settings
        self._llm_client = llm_client

    def evaluate(
        self,
        criterion: Criterion,
        patient: Patient,
        evidence: List[Evidence],
    ) -> CriterionEvaluation:
        try:
            prompt = CRITERION_EVAL_PROMPT.format(
                criterion_description=criterion.description,
                criterion_type="Inclusion" if criterion.is_inclusion else "Exclusion",
                patient_summary=self._build_patient_summary(patient),
                evidence_text=self._build_evidence_text(evidence),
            )
            raw: dict = self._llm_client.generate_structured(prompt)

            status_raw = raw.get("status", "REQUIRES_CLINICAL_REVIEW").upper().strip()
            status     = _STATUS_MAP.get(status_raw, CriterionStatus.REQUIRES_CLINICAL_REVIEW)
            reasoning  = str(raw.get("reasoning", "No reasoning provided."))
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            questions  = raw.get("unanswered_questions", [])
            if isinstance(questions, str):
                questions = [questions]

            return CriterionEvaluation(
                criterion_id=criterion.id,
                status=status,
                reasoning=reasoning,
                evidence_used=evidence,
                confidence=confidence,
                evaluator_type=_EVALUATOR_TYPE,
                unanswered_questions=list(questions),
            )

        except Exception as exc:  # noqa: BLE001
            return CriterionEvaluation(
                criterion_id=criterion.id,
                status=CriterionStatus.REQUIRES_CLINICAL_REVIEW,
                reasoning=f"LLM evaluation failed: {exc}",
                evidence_used=evidence,
                confidence=0.0,
                evaluator_type=_ERROR_EVALUATOR_TYPE,
                unanswered_questions=[
                    "Automated evaluation unavailable — please review manually."
                ],
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_patient_summary(patient: Patient) -> str:
        lines = [f"Age: {patient.age}", f"Gender: {patient.gender}"]
        if patient.conditions:
            lines.append(f"Conditions: {', '.join(patient.conditions)}")
        if patient.medications:
            lines.append(f"Medications: {', '.join(patient.medications)}")
        key_labs = [
            f"{lr.test}: {lr.value} {lr.unit}"
            for lr in patient.lab_results
            if any(k in lr.test.lower() for k in ("hba1c", "a1c", "egfr", "gfr"))
        ]
        if key_labs:
            lines.append(f"Key labs: {'; '.join(key_labs)}")
        return "\n".join(lines)

    @staticmethod
    def _build_evidence_text(evidence: List[Evidence]) -> str:
        if not evidence:
            return "No supporting evidence available."
        return "\n".join(f"{i+1}. {e.text}" for i, e in enumerate(evidence))
