# Evaluates deterministic criteria like age and simple lab results.
"""
Rule-based criterion evaluator.

Handles: AGE, HBA1C, EGFR, RECRUITING
Mapping:
  criterion met        → SUPPORTED
  criterion not met    → NOT_SUPPORTED
  lab/data missing     → UNKNOWN  (never a negative; raises an explicit question)
  unparseable text     → REQUIRES_CLINICAL_REVIEW
"""

import re
from typing import List, Optional

from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence
from app.models.patient import Patient
from app.models.trial import Criterion, CriterionType
from app.utils.helpers import compare_values

_EVALUATOR_TYPE = "rule_engine"


class RuleEngine:
    def __init__(self, settings=None):
        self._settings = settings

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate(self, criterion: Criterion, patient: Patient) -> CriterionEvaluation:
        dispatch = {
            CriterionType.AGE:        self._eval_age,
            CriterionType.HBA1C:      self._eval_lab("hba1c", ["hba1c", "a1c"]),
            CriterionType.EGFR:       self._eval_lab("egfr",  ["egfr", "gfr"]),
            CriterionType.RECRUITING: self._eval_recruiting,
        }
        handler = dispatch.get(criterion.type)
        if handler is None:
            return self._clinical_review(criterion.id, "Not handled by rule engine")
        return handler(criterion, patient)

    # ── AGE ───────────────────────────────────────────────────────────────────

    def _eval_age(self, criterion: Criterion, patient: Patient) -> CriterionEvaluation:
        desc = criterion.description

        # Range: "18-75", "18 to 75", "between 18 and 75"
        range_match = re.search(r"(\d+)\s*(?:[-–]|to|and)\s*(\d+)", desc)
        if range_match:
            low, high = int(range_match.group(1)), int(range_match.group(2))
            met = low <= patient.age <= high
            status = CriterionStatus.SUPPORTED if met else CriterionStatus.NOT_SUPPORTED
            reasoning = (
                f"Patient age {patient.age} is "
                f"{'within' if met else 'outside'} the required range {low}–{high}."
            )
            return self._make_eval(criterion.id, status, reasoning, 1.0)

        # Single bound: ">= 18", "<= 75"
        bound_match = re.search(r"([><]=?)\s*(\d+)", desc)
        if bound_match:
            op, thr = bound_match.group(1), float(bound_match.group(2))
            met = compare_values(float(patient.age), op, thr)
            status = CriterionStatus.SUPPORTED if met else CriterionStatus.NOT_SUPPORTED
            reasoning = (
                f"Patient age {patient.age} "
                f"{'meets' if met else 'does not meet'} criterion age {op} {int(thr)}."
            )
            return self._make_eval(criterion.id, status, reasoning, 1.0)

        # Cannot parse — needs clinical review, not a negative answer
        return self._clinical_review(
            criterion.id,
            f"Age criterion text '{desc}' could not be parsed automatically.",
        )

    # ── LAB (HBA1C / EGFR) ───────────────────────────────────────────────────

    def _eval_lab(self, lab_name: str, keywords: list):
        """Return a handler closure for numeric lab criteria."""

        def handler(criterion: Criterion, patient: Patient) -> CriterionEvaluation:
            # Find the relevant lab value
            lab_value: Optional[float] = None
            lab_date: str = ""
            lab_source_id: str = ""
            for lr in patient.lab_results:
                if any(kw in lr.test.lower() for kw in keywords):
                    lab_value     = lr.value
                    lab_date      = str(lr.date) if lr.date else ""
                    lab_source_id = lr.source_id
                    break

            # Missing lab → UNKNOWN with an explicit question
            if lab_value is None:
                return self._unknown(
                    criterion.id,
                    f"{lab_name.upper()} value not found in patient records.",
                    questions=[
                        f"What is the patient's most recent {lab_name.upper()} value and date?"
                    ],
                )

            desc = criterion.description

            # Range pattern: "7.0–12.0", "7 to 12", "between 7 and 12"
            range_match = re.search(r"(\d+\.?\d*)\s*(?:[-–]|to|and)\s*(\d+\.?\d*)", desc)
            if range_match:
                low, high = float(range_match.group(1)), float(range_match.group(2))
                in_range  = low <= lab_value <= high
                status    = CriterionStatus.SUPPORTED if in_range else CriterionStatus.NOT_SUPPORTED
                reasoning = (
                    f"Patient {lab_name.upper()} {lab_value} is "
                    f"{'within' if in_range else 'outside'} range {low}–{high}."
                )
                ev = self._lab_evidence(lab_name, lab_value, lab_date, lab_source_id)
                return self._make_eval(criterion.id, status, reasoning, 1.0, [ev])

            # Operator pattern: ">= 7.5", "< 30"
            op_match = re.search(r"([><]=?)\s*(\d+\.?\d*)", desc)
            if op_match:
                op, thr  = op_match.group(1), float(op_match.group(2))
                result   = compare_values(lab_value, op, thr)
                # For exclusion criteria: condition TRUE means patient is excluded
                eligible = result if criterion.is_inclusion else not result
                status   = CriterionStatus.SUPPORTED if eligible else CriterionStatus.NOT_SUPPORTED
                direction = "meets" if result else "does not meet"
                reasoning = (
                    f"Patient {lab_name.upper()} {lab_value} {direction} criterion {op} {thr}."
                )
                ev = self._lab_evidence(lab_name, lab_value, lab_date, lab_source_id)
                return self._make_eval(criterion.id, status, reasoning, 1.0, [ev])

            # Cannot parse the threshold
            return self._clinical_review(
                criterion.id,
                f"{lab_name.upper()} criterion text '{desc}' could not be parsed.",
            )

        return handler

    # ── RECRUITING ────────────────────────────────────────────────────────────

    def _eval_recruiting(self, criterion: Criterion, patient: Patient) -> CriterionEvaluation:
        return self._make_eval(
            criterion.id,
            CriterionStatus.SUPPORTED,
            "Trial recruiting status verified in the filtering stage.",
            1.0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _lab_evidence(lab_name: str, value: float, date: str, source_id: str = "") -> Evidence:
        return Evidence(
            text=f"{lab_name.upper()}: {value}",
            source="lab_results",
            retrieved_from="patient",
            relevance_score=1.0,
            evidence_id=source_id,
            date=date,
        )

    @staticmethod
    def _make_eval(
        criterion_id: str,
        status: CriterionStatus,
        reasoning: str,
        confidence: float,
        evidence: Optional[List[Evidence]] = None,
        questions: Optional[List[str]] = None,
    ) -> CriterionEvaluation:
        return CriterionEvaluation(
            criterion_id=criterion_id,
            status=status,
            reasoning=reasoning,
            evidence_used=evidence or [],
            confidence=confidence,
            evaluator_type=_EVALUATOR_TYPE,
            unanswered_questions=questions or [],
        )

    @staticmethod
    def _unknown(
        criterion_id: str,
        reasoning: str,
        questions: Optional[List[str]] = None,
    ) -> CriterionEvaluation:
        """Missing data — never a negative answer, always an explicit question."""
        return CriterionEvaluation(
            criterion_id=criterion_id,
            status=CriterionStatus.UNKNOWN,
            reasoning=reasoning,
            evidence_used=[],
            confidence=0.0,
            evaluator_type=_EVALUATOR_TYPE,
            unanswered_questions=questions or ["Missing data — please provide the relevant value."],
        )

    @staticmethod
    def _clinical_review(
        criterion_id: str,
        reasoning: str,
    ) -> CriterionEvaluation:
        """Criterion requires clinical judgement; cannot be automated."""
        return CriterionEvaluation(
            criterion_id=criterion_id,
            status=CriterionStatus.REQUIRES_CLINICAL_REVIEW,
            reasoning=reasoning,
            evidence_used=[],
            confidence=0.0,
            evaluator_type=_EVALUATOR_TYPE,
            unanswered_questions=["Clinical review required — please evaluate manually."],
        )
