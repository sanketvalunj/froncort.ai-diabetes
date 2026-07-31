"""
LLM-based criterion evaluator — token-optimised.

Design:
  - Patient fields sent to the LLM are scoped to what the criterion type actually needs:
      MEDICATION  → age + medications only
      CONDITION   → age + conditions only
      (other LLM) → age + gender only
  - Evidence is serialised as compact JSON objects (id / src / text) instead of
    numbered prose lines, saving 10–20 tokens per item.
  - The prompt returns only three fields: state / reason (≤20 words) / evidence_ids.
  - Full backward-compat fallback: also accepts the old "status"/"reasoning" keys so
    that all existing mock-based tests continue to pass without modification.

Token budget target: 300–500 input tokens per call.
"""

import json
from typing import List

from config.prompts import CRITERION_EVAL_PROMPT
from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence
from app.models.patient import Patient
from app.models.trial import Criterion, CriterionType

_EVALUATOR_TYPE       = "llm_engine"
_ERROR_EVALUATOR_TYPE = "llm_engine_error"

# Map LLM response vocabulary → CriterionStatus (covers new and legacy keys)
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
                criterion_type="Inclusion" if criterion.is_inclusion else "Exclusion",
                criterion_description=criterion.description,
                patient_fields=self._build_patient_fields(criterion, patient),
                evidence_json=self._build_evidence_json(evidence),
            )
            raw: dict = self._llm_client.generate_structured(prompt)

            # ── Parse response — new shape (state/reason/evidence_ids)
            #    with fallback to legacy shape (status/reasoning/confidence) ──
            status_raw = (
                raw.get("state") or raw.get("status") or "REQUIRES_CLINICAL_REVIEW"
            ).upper().strip()
            status = _STATUS_MAP.get(status_raw, CriterionStatus.REQUIRES_CLINICAL_REVIEW)

            reasoning = str(raw.get("reason") or raw.get("reasoning") or "No reasoning provided.")

            # confidence: new response omits it; default 0.8 when resolved, 0.0 otherwise
            if "confidence" in raw:
                confidence = max(0.0, min(1.0, float(raw["confidence"])))
            else:
                confidence = 0.8 if status in (
                    CriterionStatus.SUPPORTED, CriterionStatus.NOT_SUPPORTED
                ) else 0.0

            # unanswered questions: not in new shape, infer from status
            questions: List[str] = raw.get("unanswered_questions", [])
            if isinstance(questions, str):
                questions = [questions]
            if not questions and status in (
                CriterionStatus.UNKNOWN, CriterionStatus.REQUIRES_CLINICAL_REVIEW
            ):
                questions = ["Automated evaluation unavailable — please review manually."]

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

    # ── Patient field scoping ─────────────────────────────────────────────────

    @staticmethod
    def _build_patient_fields(criterion: Criterion, patient: Patient) -> str:
        """
        Return only the patient fields relevant to this criterion type.

        MEDICATION  → age + medications list (omit conditions, labs — not needed)
        CONDITION   → age + conditions list  (omit medications, labs — not needed)
        anything else → age + gender only    (minimal anchor)

        Serialised as a compact one-line JSON string to save tokens.
        """
        if criterion.type == CriterionType.MEDICATION:
            fields: dict = {"age": patient.age}
            if patient.medications:
                fields["meds"] = patient.medications
            else:
                fields["meds"] = []

        elif criterion.type == CriterionType.CONDITION:
            fields = {"age": patient.age}
            if patient.conditions:
                # Truncate to first 5 conditions — very long lists add tokens without value
                fields["conditions"] = patient.conditions[:5]
            else:
                fields["conditions"] = []

        else:
            # Minimal anchor for any other LLM-routed criterion type
            fields = {"age": patient.age, "gender": patient.gender}

        return json.dumps(fields, separators=(",", ":"))

    # ── Evidence serialisation ────────────────────────────────────────────────

    @staticmethod
    def _build_evidence_json(evidence: List[Evidence]) -> str:
        """
        Serialise evidence as a compact JSON array.

        Each item: {"id":"<evidence_id or idx>","src":"<source>","text":"<text>"}
        Text is truncated to 120 characters to bound token usage per item.
        Returns "[]" when there is no evidence.
        """
        if not evidence:
            return "[]"

        items = []
        for idx, ev in enumerate(evidence):
            eid = ev.evidence_id if ev.evidence_id else str(idx)
            items.append({
                "id":   eid,
                "src":  ev.source,
                "text": ev.text[:120],
            })
        return json.dumps(items, separators=(",", ":"))
