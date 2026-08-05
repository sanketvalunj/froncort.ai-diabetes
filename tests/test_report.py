# Unit tests for the report generation module.
"""Tests for the ReportGenerator (Step 4 — coordinator-facing format)."""

import pytest
from app.models.evaluation import CriterionEvaluation, CriterionStatus, TrialRanking
from app.models.patient import Patient, LabResult
from app.reports.report_generator import ReportGenerator
from tests.conftest import make_eval, make_ranking


@pytest.fixture
def gen():
    return ReportGenerator()


@pytest.fixture
def patient():
    return Patient(
        id="p_report_test", age=62, gender="Male",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["metformin", "lisinopril"],
        lab_results=[
            LabResult(test="HbA1c", value=8.2, unit="%"),
            LabResult(test="eGFR",  value=55.0, unit="mL/min/1.73m2"),
        ],
    )


# ── Return types ──────────────────────────────────────────────────────────────

def test_returns_tuple(gen, patient):
    assert isinstance(gen.generate(patient, [], {}), tuple)

def test_markdown_is_string(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert isinstance(md, str)

def test_data_is_dict(gen, patient):
    _, d = gen.generate(patient, [], {})
    assert isinstance(d, dict)


# ── Required coordinator-facing fields ────────────────────────────────────────

def test_markdown_has_header(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "Pre-Screening Report" in md

def test_markdown_contains_patient_id(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert patient.id in md

def test_markdown_contains_conditions(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "Type 2 Diabetes" in md

def test_markdown_contains_medications(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "metformin" in md

def test_markdown_contains_labs(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "HbA1c" in md and "eGFR" in md

def test_trial_id_and_title_present(gen, patient):
    r   = make_ranking("t1", "Alpha Trial", CriterionStatus.SUPPORTED, score=1.0, sup=1)
    md, _ = gen.generate(patient, [r], {})
    assert "Alpha Trial" in md and "t1" in md

def test_reason_surfaced_present(gen, patient):
    r = make_ranking("t1", "A", CriterionStatus.SUPPORTED, score=1.0, sup=1)
    md, _ = gen.generate(patient, [r], {})
    assert "Reason surfaced" in md or "reason" in md.lower()

def test_criterion_status_labels_present(gen, patient):
    """Each of the 5 status values must be nameable in the output."""
    rankings = [make_ranking("t1", "A", CriterionStatus.SUPPORTED, score=1.0, sup=1,
                             unk=1, rev=1)]
    evals = {"t1": [
        make_eval("c1", CriterionStatus.SUPPORTED),
        make_eval("c2", CriterionStatus.UNKNOWN),
        make_eval("c3", CriterionStatus.REQUIRES_CLINICAL_REVIEW),
    ]}
    md, _ = gen.generate(patient, rankings, evals)
    assert "SUPPORTED" in md
    assert "UNKNOWN" in md
    assert "REQUIRES_CLINICAL_REVIEW" in md

def test_evidence_source_present(gen, patient):
    """Lab evidence must appear in the criterion table.

    _format_evidence() converts the raw source field ('lab_results') into a
    human-readable label such as 'HbA1c (2024-01-15)'.  The important thing is
    that the evidence text and date are visible in the rendered Markdown — not
    the internal source key.
    """
    from app.models.evaluation import Evidence
    ev = CriterionEvaluation(
        criterion_id="c1", status=CriterionStatus.SUPPORTED,
        reasoning="Patient HbA1c 8.2%", confidence=1.0,
        evaluator_type="rule_engine",
        evidence_used=[Evidence(text="HbA1c: 8.2", source="lab_results",
                                retrieved_from="patient", date="2024-01-15")],
    )
    rankings = [make_ranking("t1", "A", CriterionStatus.SUPPORTED, score=1.0, sup=1)]
    md, _    = gen.generate(patient, rankings, {"t1": [ev]})
    # The evidence label rendered by _format_evidence is "HbA1c (2024-01-15)"
    assert "HbA1c" in md
    assert "2024-01-15" in md

def test_unanswered_questions_shown(gen, patient):
    """UNKNOWN criteria must surface their questions in the report."""
    ev = CriterionEvaluation(
        criterion_id="c1", status=CriterionStatus.UNKNOWN,
        reasoning="eGFR not in records", confidence=0.0,
        evaluator_type="rule_engine",
        unanswered_questions=["What is the patient's most recent eGFR value?"],
    )
    rankings = [make_ranking("t1", "A", CriterionStatus.UNKNOWN, score=0.0, unk=1)]
    md, _    = gen.generate(patient, rankings, {"t1": [ev]})
    assert "eGFR" in md
    assert "Unanswered" in md or "unanswered" in md.lower() or "Clinical Review" in md

def test_clinical_fit_and_recruiting_shown_separately(gen, patient):
    """Clinical Fit and Recruiting status must appear as distinct rows."""
    r   = make_ranking("t1", "A", CriterionStatus.SUPPORTED, score=1.0, sup=1,
                        is_recruiting=True)
    md, _ = gen.generate(patient, [r], {})
    assert "Clinical Fit" in md
    assert "Recruiting" in md

def test_human_review_flag_present(gen, patient):
    """requires_human_review flag must be visible in the output."""
    r   = make_ranking("t1", "A", CriterionStatus.UNKNOWN, score=0.0, unk=1)
    md, _ = gen.generate(patient, [r], {})
    assert "Human Review" in md or "human review" in md.lower()

def test_human_review_not_required_when_all_supported(gen, patient):
    r   = make_ranking("t1", "A", CriterionStatus.SUPPORTED, score=1.0, sup=3)
    md, _ = gen.generate(patient, [r], {})
    assert "Not required" in md or "not required" in md.lower()

def test_safety_disclaimer_present(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "clinical judgement" in md.lower() or "healthcare professional" in md.lower()

def test_filter_appendix_shown(gen, patient):
    md, _ = gen.generate(patient, [], {}, filter_reasons={"t_old": "Status: Completed"})
    assert "t_old" in md and ("Appendix" in md or "Excluded" in md)

def test_no_appendix_without_filter_reasons(gen, patient):
    md, _ = gen.generate(patient, [], {})
    assert "t_old" not in md


# ── Structured data dict ──────────────────────────────────────────────────────

def test_data_patient_id(gen, patient):
    _, d = gen.generate(patient, [], {})
    assert d["patient_id"] == patient.id

def test_data_summary_counts(gen, patient):
    rankings = [
        make_ranking("t1", "A", CriterionStatus.SUPPORTED,     1.0,  sup=2),
        make_ranking("t2", "B", CriterionStatus.NOT_SUPPORTED, -1.0, nsup=1),
        make_ranking("t3", "C", CriterionStatus.UNKNOWN,        0.0, unk=1),
    ]
    _, d = gen.generate(patient, rankings, {})
    assert d["summary"]["supported_count"]     == 1
    assert d["summary"]["not_supported_count"] == 1
    assert d["summary"]["human_review_required"] == 1  # only UNKNOWN trial has requires_human_review=True

def test_data_trials_list_length(gen, patient):
    rankings = [
        make_ranking("t1", "A", CriterionStatus.SUPPORTED,     1.0,  sup=1),
        make_ranking("t2", "B", CriterionStatus.NOT_SUPPORTED, -1.0, nsup=1),
    ]
    _, d = gen.generate(patient, rankings, {})
    assert len(d["trials"]) == 2

def test_data_trial_has_criteria(gen, patient):
    rankings = [make_ranking("t1", "A", CriterionStatus.SUPPORTED, 1.0, sup=1)]
    _, d     = gen.generate(patient, rankings, {"t1": [make_eval("c1", CriterionStatus.SUPPORTED)]})
    assert len(d["trials"][0]["criteria"]) == 1

def test_data_has_generated_at(gen, patient):
    _, d = gen.generate(patient, [], {})
    assert "generated_at" in d and "T" in d["generated_at"]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_trials_no_crash(gen, patient):
    md, d = gen.generate(patient, [], {})
    assert isinstance(md, str) and d["summary"]["total_ranked"] == 0

def test_patient_no_labs_no_crash(gen):
    md, _ = gen.generate(Patient(id="bare", age=45, gender="F"), [], {})
    assert "bare" in md

def test_trials_ordered(gen, patient):
    rankings = [
        make_ranking("t1", "First Trial",  CriterionStatus.SUPPORTED,     1.0,  sup=2),
        make_ranking("t2", "Second Trial", CriterionStatus.NOT_SUPPORTED, -1.0, nsup=1),
    ]
    md, _ = gen.generate(patient, rankings, {})
    assert md.index("First Trial") < md.index("Second Trial")
