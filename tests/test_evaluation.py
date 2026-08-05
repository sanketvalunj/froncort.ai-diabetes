# Unit tests for the evaluation engines.
"""Tests for the evaluation layer: RuleEngine, LLMEvaluator, EvaluatorRouter."""

import pytest
from unittest.mock import MagicMock

from app.evaluation.rule_engine import RuleEngine
from app.evaluation.llm_engine import LLMEvaluator
from app.evaluation.router import EvaluatorRouter
from app.models.evaluation import CriterionStatus, Evidence
from app.models.patient import Patient, LabResult
from app.models.trial import Criterion, CriterionType


@pytest.fixture
def rule_engine():
    return RuleEngine()


@pytest.fixture
def patient_62():
    return Patient(id="p_eval", age=62, gender="Male",
                   conditions=["Type 2 Diabetes"], medications=["metformin"],
                   lab_results=[LabResult(test="HbA1c", value=8.2, unit="%"),
                                 LabResult(test="eGFR",  value=55.0, unit="mL/min/1.73m2")])


def make_c(ctype, desc, is_inclusion=True, cid="c_test"):
    return Criterion(id=cid, type=ctype, description=desc, is_inclusion=is_inclusion)


# ── RuleEngine — AGE ──────────────────────────────────────────────────────────

def test_rule_age_supported_in_range(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.AGE, "Patients aged 18-75 years"), patient_62)
    assert r.status == CriterionStatus.SUPPORTED and r.confidence == 1.0

def test_rule_age_not_supported_below_range(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.AGE, "Patients aged 18-45 years"), patient_62)
    assert r.status == CriterionStatus.NOT_SUPPORTED

def test_rule_age_not_supported_above_range(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.AGE, "Must be between 18 and 55 years old"), patient_62)
    assert r.status == CriterionStatus.NOT_SUPPORTED

def test_rule_age_supported_single_bound(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.AGE, "Age >= 18 years"), patient_62)
    assert r.status == CriterionStatus.SUPPORTED

def test_rule_age_clinical_review_unparseable(rule_engine, patient_62):
    """Unparseable age text → REQUIRES_CLINICAL_REVIEW (not a negative answer)."""
    r = rule_engine.evaluate(make_c(CriterionType.AGE, "Adults only"), patient_62)
    assert r.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW
    assert len(r.unanswered_questions) > 0


# ── RuleEngine — HBA1C ────────────────────────────────────────────────────────

def test_rule_hba1c_supported(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.HBA1C, "HbA1c >= 7.5%"), patient_62)
    assert r.status == CriterionStatus.SUPPORTED

def test_rule_hba1c_not_supported(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.HBA1C, "HbA1c <= 7.0%"), patient_62)
    assert r.status == CriterionStatus.NOT_SUPPORTED

def test_rule_hba1c_in_range(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.HBA1C, "HbA1c between 7.0 and 12.0%"), patient_62)
    assert r.status == CriterionStatus.SUPPORTED

def test_rule_hba1c_unknown_when_missing(rule_engine):
    """Missing lab → UNKNOWN, not NOT_SUPPORTED."""
    r = rule_engine.evaluate(make_c(CriterionType.HBA1C, "HbA1c >= 7.5%"),
                              Patient(id="x", age=50, gender="F"))
    assert r.status == CriterionStatus.UNKNOWN
    assert any("HbA1c" in q or "hba1c" in q.lower() for q in r.unanswered_questions)


# ── RuleEngine — EGFR ────────────────────────────────────────────────────────

def test_rule_egfr_supported(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.EGFR, "eGFR >= 30", is_inclusion=True), patient_62)
    assert r.status == CriterionStatus.SUPPORTED

def test_rule_egfr_exclusion_not_triggered(rule_engine, patient_62):
    """eGFR 55 with exclusion 'eGFR < 30' — exclusion condition is FALSE → SUPPORTED."""
    r = rule_engine.evaluate(make_c(CriterionType.EGFR, "eGFR < 30 mL/min/1.73m2", is_inclusion=False), patient_62)
    assert r.status == CriterionStatus.SUPPORTED

def test_rule_egfr_exclusion_triggered(rule_engine):
    p = Patient(id="x", age=50, gender="F",
                lab_results=[LabResult(test="eGFR", value=25.0, unit="mL/min/1.73m2")])
    r = rule_engine.evaluate(make_c(CriterionType.EGFR, "eGFR < 30 mL/min/1.73m2", is_inclusion=False), p)
    assert r.status == CriterionStatus.NOT_SUPPORTED

def test_rule_egfr_unknown_when_missing(rule_engine):
    r = rule_engine.evaluate(make_c(CriterionType.EGFR, "eGFR >= 45"),
                              Patient(id="x", age=50, gender="M"))
    assert r.status == CriterionStatus.UNKNOWN


# ── RuleEngine — RECRUITING ───────────────────────────────────────────────────

def test_rule_recruiting_supported(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.RECRUITING, "Currently recruiting"), patient_62)
    assert r.status == CriterionStatus.SUPPORTED and r.confidence == 1.0


# ── RuleEngine — OTHER ────────────────────────────────────────────────────────

def test_rule_other_clinical_review(rule_engine, patient_62):
    r = rule_engine.evaluate(make_c(CriterionType.OTHER, "Signed informed consent"), patient_62)
    assert r.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW


# ── LLMEvaluator ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    c = MagicMock()
    c.generate_structured.return_value = {
        "status": "SUPPORTED", "reasoning": "Patient is on metformin.",
        "confidence": 0.9, "unanswered_questions": []
    }
    return c

def test_llm_evaluator_supported(mock_llm, patient_62):
    ev = LLMEvaluator(MagicMock(), mock_llm).evaluate(
        make_c(CriterionType.MEDICATION, "Must be on metformin"), patient_62, [])
    assert ev.status == CriterionStatus.SUPPORTED and ev.evaluator_type == "llm_engine"

def test_llm_evaluator_maps_eligible_alias(patient_62):
    """LLM may return old 'ELIGIBLE' string — should map to SUPPORTED."""
    c = MagicMock()
    c.generate_structured.return_value = {"status": "ELIGIBLE", "reasoning": "ok", "confidence": 0.8}
    ev = LLMEvaluator(MagicMock(), c).evaluate(
        make_c(CriterionType.MEDICATION, "On metformin"), patient_62, [])
    assert ev.status == CriterionStatus.SUPPORTED

def test_llm_evaluator_api_error(patient_62):
    bad = MagicMock()
    bad.generate_structured.side_effect = RuntimeError("timeout")
    ev = LLMEvaluator(MagicMock(), bad).evaluate(
        make_c(CriterionType.CONDITION, "T2D"), patient_62, [])
    assert ev.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW
    assert ev.evaluator_type == "llm_engine_error"
    assert len(ev.unanswered_questions) > 0

def test_llm_evaluator_parse_error(patient_62):
    from app.llm.client import LLMParseError
    bad = MagicMock()
    bad.generate_structured.side_effect = LLMParseError("bad json")
    ev = LLMEvaluator(MagicMock(), bad).evaluate(
        make_c(CriterionType.CONDITION, "T2D"), patient_62, [])
    assert ev.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW


# ── EvaluatorRouter ───────────────────────────────────────────────────────────

@pytest.fixture
def router(mock_llm):
    return EvaluatorRouter(MagicMock(), mock_llm)

@pytest.mark.parametrize("ctype,desc", [
    (CriterionType.AGE,        "Aged 18-75 years"),
    (CriterionType.HBA1C,      "HbA1c >= 7.5%"),
    (CriterionType.EGFR,       "eGFR >= 30"),
    (CriterionType.RECRUITING, "Currently recruiting"),
])
def test_router_rule_engine(router, patient_62, ctype, desc):
    assert router.evaluate(make_c(ctype, desc), patient_62, [], []).evaluator_type == "rule_engine"

@pytest.mark.parametrize("ctype,desc", [
    (CriterionType.MEDICATION, "Must be on metformin"),
    (CriterionType.CONDITION,  "Diagnosis of Type 2 Diabetes"),
])
def test_router_llm_engine(router, patient_62, ctype, desc, mock_llm):
    r = router.evaluate(make_c(ctype, desc), patient_62, [], [])
    assert r.evaluator_type == "llm_engine"

def test_router_other_clinical_review(router, patient_62):
    r = router.evaluate(make_c(CriterionType.OTHER, "consent"), patient_62, [], [])
    assert r.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW
    assert r.evaluator_type == "clinical_review"

def test_router_combines_evidence(patient_62, mock_llm):
    pev = [Evidence(text="metformin", source="medications")]
    tev = [Evidence(text="Inclusion: metformin use", source="inclusion", retrieved_from="trial")]
    r   = EvaluatorRouter(MagicMock(), mock_llm).evaluate(
        make_c(CriterionType.MEDICATION, "On metformin"), patient_62, pev, tev)
    assert len(r.evidence_used) == 2
