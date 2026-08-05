# Pytest configuration and shared fixtures for testing.
import pytest
from app.models.patient import Patient, LabResult
from app.models.trial import Trial, Criterion, CriterionType
from app.models.evaluation import CriterionStatus, CriterionEvaluation, TrialRanking
from app.models.state import AgentState


@pytest.fixture
def sample_patient() -> Patient:
    return Patient(
        id="patient_001", age=62, gender="Male",
        conditions=["Type 2 Diabetes", "Hypertension"],
        medications=["metformin", "lisinopril"],
        lab_results=[
            LabResult(test="HbA1c", value=8.2, unit="%"),
            LabResult(test="eGFR",  value=55.0, unit="mL/min/1.73m2"),
            LabResult(test="Fasting Glucose", value=156.0, unit="mg/dL"),
        ],
        medical_history="Diagnosed with Type 2 Diabetes 8 years ago.",
    )


@pytest.fixture
def sample_trials() -> list:
    trial_a = Trial(
        id="trial_001", title="Metformin Optimisation in T2D",
        phase="Phase 3", status="Recruiting",
        description="Evaluating dose optimisation of metformin.",
        inclusion_criteria=[
            Criterion(id="inc_0", type=CriterionType.AGE,
                      description="Patients aged 18-75 years", is_inclusion=True),
            Criterion(id="inc_1", type=CriterionType.HBA1C,
                      description="HbA1c >= 7.0%", is_inclusion=True),
            Criterion(id="inc_2", type=CriterionType.CONDITION,
                      description="Diagnosis of Type 2 Diabetes for at least 1 year",
                      is_inclusion=True),
        ],
        exclusion_criteria=[
            Criterion(id="exc_0", type=CriterionType.EGFR,
                      description="eGFR < 30 mL/min/1.73m2", is_inclusion=False),
        ],
    )
    trial_b = Trial(
        id="trial_002", title="GLP-1 Receptor Agonist Study",
        phase="Phase 2", status="Recruiting",
        description="Assessing GLP-1 receptor agonist efficacy.",
        inclusion_criteria=[
            Criterion(id="inc_0", type=CriterionType.AGE,
                      description="Patients aged 18-45 years", is_inclusion=True),
            Criterion(id="inc_1", type=CriterionType.HBA1C,
                      description="HbA1c between 7.5% and 12.0%", is_inclusion=True),
        ],
        exclusion_criteria=[
            Criterion(id="exc_0", type=CriterionType.MEDICATION,
                      description="Current insulin treatment", is_inclusion=False),
        ],
    )
    trial_c = Trial(
        id="trial_003", title="Renal Outcomes in T2D — Closed",
        phase="Phase 3", status="Completed",
        description="Completed study on renal outcomes.",
        inclusion_criteria=[
            Criterion(id="inc_0", type=CriterionType.EGFR,
                      description="eGFR >= 45 mL/min/1.73m2", is_inclusion=True),
        ],
    )
    return [trial_a, trial_b, trial_c]


@pytest.fixture
def sample_state(sample_patient, sample_trials) -> AgentState:
    return AgentState(patient=sample_patient, all_trials=sample_trials)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_eval(criterion_id: str, status: CriterionStatus,
              evaluator: str = "rule_engine") -> CriterionEvaluation:
    return CriterionEvaluation(criterion_id=criterion_id, status=status,
                               reasoning="test", confidence=1.0,
                               evaluator_type=evaluator)


def make_ranking(trial_id: str, title: str, clinical_fit: CriterionStatus,
                 score: float = 0.0, is_recruiting: bool = True,
                 sup: int = 0, nsup: int = 0, unk: int = 0,
                 conf: int = 0, rev: int = 0) -> TrialRanking:
    total = sup + nsup + unk + conf + rev
    return TrialRanking(
        trial_id=trial_id, title=title,
        clinical_fit=clinical_fit, is_recruiting=is_recruiting,
        score=score,
        supported_count=sup, not_supported_count=nsup,
        unknown_count=unk, conflicting_count=conf, review_count=rev,
        total_criteria=total,
        requires_human_review=(unk + conf + rev) > 0,
        reason_surfaced="test",
    )
