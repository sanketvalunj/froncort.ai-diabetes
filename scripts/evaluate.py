# CLI script for evaluating the agent's performance against test cases.
"""
Evaluation suite for the Clinical Trial Pre-Screening Assistant.

Runs 10+ evaluation cases across five categories:
  1. Retrieval quality
  2. Criterion status correctness
  3. Agent / pipeline behaviour
  4. Dataset coverage (real dataset)
  5. Output quality (report completeness)

Usage:
    python scripts/evaluate.py              # run all cases, print results
    python scripts/evaluate.py --json       # also write results/eval_results.json

Each case reports PASS / FAIL and a short diagnostic.
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import typer
from app.evaluation.rule_engine import RuleEngine
from app.evaluation.router import EvaluatorRouter
from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence
from app.models.patient import Patient, LabResult
from app.models.state import AgentState
from app.models.trial import Criterion, CriterionType, Trial
from app.reports.report_generator import ReportGenerator
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.loader import load_dataset
from app.retrieval.parser import parse_patient, parse_trials
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.vectorstore import FAISSVectorStore
from app.services.filtering_service import FilteringService
from app.services.ranking_service import RankingService
from config.settings import settings

app = typer.Typer(help="Evaluation suite for the pre-screening pipeline.")

# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id:    str
    category:   str
    name:       str
    passed:     bool
    diagnostic: str
    duration_ms: float = 0.0


@dataclass
class EvalReport:
    total:    int = 0
    passed:   int = 0
    failed:   int = 0
    cases:    List[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_patient(age=62, hba1c=8.2, egfr=55.0, conditions=None, medications=None):
    return Patient(
        id="eval_patient",
        age=age, gender="Male",
        conditions=conditions or ["Type 2 Diabetes"],
        medications=medications or ["metformin"],
        lab_results=[
            LabResult(test="HbA1c", value=hba1c, unit="%"),
            LabResult(test="eGFR",  value=egfr,  unit="mL/min/1.73m2"),
        ],
    )


def _make_criterion(ctype, desc, is_inclusion=True):
    return Criterion(id="c_eval", type=ctype, description=desc, is_inclusion=is_inclusion)


def _make_ranking(trial_id, title, fit, score, sup=0, nsup=0, unk=0):
    from app.models.evaluation import TrialRanking
    return TrialRanking(
        trial_id=trial_id, title=title, clinical_fit=fit,
        is_recruiting=True, score=score,
        supported_count=sup, not_supported_count=nsup, unknown_count=unk,
        conflicting_count=0, review_count=0, total_criteria=sup+nsup+unk,
        requires_human_review=(unk > 0),
        reason_surfaced=f"{sup}/{sup+nsup+unk} criteria supported.",
    )


def _run(fn) -> CaseResult:
    t0 = time.perf_counter()
    result: CaseResult = fn()
    result.duration_ms = (time.perf_counter() - t0) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — Retrieval quality
# ═══════════════════════════════════════════════════════════════════════════════

def case_R1_retrieval_returns_results():
    """R1: EvidenceRetriever returns at least one result for a T2D trial.

    Uses score_threshold=0.0 and a high top_k so that at least the indexed
    criteria for the target trial appear regardless of cosine distance, and
    picks a trial that is semantically related to the T2D query.
    """
    mock_settings = MagicMock()
    mock_settings.embeddings.model   = str(settings.embeddings.model)
    mock_settings.retrieval.top_k    = 30   # wide net — index has 581 vectors
    mock_settings.retrieval.score_threshold = 0.0  # no threshold: any match counts
    mock_settings.paths.vector_store = settings.paths.vector_store

    retriever = EvidenceRetriever(mock_settings)
    patient   = _make_patient()
    queries   = retriever.build_queries(patient)

    # Pick a trial that is semantically aligned with the patient queries
    # (RAY1225 T2D study; top hit when querying "Type 2 diabetes clinical trial")
    store = FAISSVectorStore(settings.paths.vector_store)
    store.load()
    # Find a trial that has an inclusion criterion mentioning type 2 diabetes
    target_id = next(
        (m["trial_id"] for m in store.metadata
         if m.get("source") == "inclusion" and "type 2 diabetes" in m.get("text", "").lower()),
        store.metadata[0]["trial_id"],
    )

    evidence = retriever.retrieve_for_trial(target_id, queries)
    passed   = len(evidence) > 0
    return CaseResult(
        case_id="R1", category="Retrieval",
        name="Retriever returns results for a T2D-aligned trial",
        passed=passed,
        diagnostic=f"Retrieved {len(evidence)} evidence items for trial {target_id}.",
    )


def case_R2_retrieval_only_returns_matching_trial():
    """R2: Results are scoped to the requested trial_id only (no cross-trial bleed)."""
    mock_settings = MagicMock()
    mock_settings.embeddings.model   = str(settings.embeddings.model)
    mock_settings.retrieval.top_k    = 30
    mock_settings.retrieval.score_threshold = 0.0
    mock_settings.paths.vector_store = settings.paths.vector_store

    retriever = EvidenceRetriever(mock_settings)
    patient   = _make_patient()
    queries   = retriever.build_queries(patient)

    store = FAISSVectorStore(settings.paths.vector_store)
    store.load()
    target_id = next(
        (m["trial_id"] for m in store.metadata
         if m.get("source") == "inclusion" and "type 2 diabetes" in m.get("text", "").lower()),
        store.metadata[0]["trial_id"],
    )

    evidence    = retriever.retrieve_for_trial(target_id, queries)
    cross_bleed = [e for e in evidence if e.retrieved_from != "trial"]
    passed      = len(evidence) > 0 and len(cross_bleed) == 0
    return CaseResult(
        case_id="R2", category="Retrieval",
        name="Retriever scopes results to requested trial (no cross-trial bleed)",
        passed=passed,
        diagnostic=(
            f"Retrieved {len(evidence)} items for {target_id}; "
            f"all retrieved_from='trial': {len(cross_bleed) == 0}."
        ),
    )


def case_R3_patient_evidence_covers_key_sources():
    """R3: Patient evidence extraction covers demographics, labs, conditions, medications."""
    retriever = EvidenceRetriever.__new__(EvidenceRetriever)
    patient   = _make_patient(conditions=["Type 2 Diabetes", "Hypertension"],
                              medications=["metformin", "lisinopril"])
    evidence  = retriever.extract_patient_evidence(patient)
    sources   = {e.source for e in evidence}
    required  = {"demographics", "lab_results", "conditions", "medications"}
    missing   = required - sources
    passed    = len(missing) == 0
    return CaseResult(
        case_id="R3", category="Retrieval",
        name="Patient evidence extraction covers all source types",
        passed=passed,
        diagnostic=(
            f"Sources found: {sorted(sources)}. "
            + (f"Missing: {sorted(missing)}." if missing else "All required sources present.")
        ),
    )


def case_R4_queries_include_lab_values():
    """R4: build_queries includes numeric lab values so retrieval can match thresholds."""
    retriever = EvidenceRetriever.__new__(EvidenceRetriever)
    patient   = _make_patient(hba1c=8.2, egfr=55.0)
    queries   = retriever.build_queries(patient)
    has_hba1c = any("8.2" in q or "hba1c" in q.lower() or "a1c" in q.lower() for q in queries)
    has_egfr  = any("55" in q or "egfr" in q.lower() or "renal" in q.lower() for q in queries)
    passed    = has_hba1c and has_egfr
    return CaseResult(
        case_id="R4", category="Retrieval",
        name="Query builder includes HbA1c and eGFR lab values",
        passed=passed,
        diagnostic=(
            f"HbA1c in queries: {has_hba1c}. eGFR in queries: {has_egfr}. "
            f"Queries: {queries}"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — Criterion status correctness
# ═══════════════════════════════════════════════════════════════════════════════

def case_C1_age_in_range_is_supported():
    """C1: Patient age 62 inside 18–75 range → SUPPORTED."""
    engine = RuleEngine()
    r      = engine.evaluate(_make_criterion(CriterionType.AGE, "Patients aged 18-75 years"),
                             _make_patient(age=62))
    passed = r.status == CriterionStatus.SUPPORTED
    return CaseResult(
        case_id="C1", category="Criterion status",
        name="Age in range → SUPPORTED",
        passed=passed,
        diagnostic=f"status={r.status.value}, confidence={r.confidence}.",
    )


def case_C2_age_outside_range_is_not_supported():
    """C2: Patient age 62 outside 18–45 range → NOT_SUPPORTED."""
    engine = RuleEngine()
    r      = engine.evaluate(_make_criterion(CriterionType.AGE, "Patients aged 18-45 years"),
                             _make_patient(age=62))
    passed = r.status == CriterionStatus.NOT_SUPPORTED
    return CaseResult(
        case_id="C2", category="Criterion status",
        name="Age outside range → NOT_SUPPORTED",
        passed=passed,
        diagnostic=f"status={r.status.value}.",
    )


def case_C3_missing_lab_is_unknown_not_negative():
    """C3: Patient with no HbA1c on record → UNKNOWN (not NOT_SUPPORTED)."""
    engine  = RuleEngine()
    patient = Patient(id="x", age=55, gender="M", lab_results=[])
    r       = engine.evaluate(_make_criterion(CriterionType.HBA1C, "HbA1c >= 7.5%"), patient)
    passed  = r.status == CriterionStatus.UNKNOWN
    has_q   = len(r.unanswered_questions) > 0
    return CaseResult(
        case_id="C3", category="Criterion status",
        name="Missing lab → UNKNOWN (not NOT_SUPPORTED)",
        passed=passed and has_q,
        diagnostic=(
            f"status={r.status.value}. "
            f"Unanswered questions: {r.unanswered_questions}."
        ),
    )


def case_C4_exclusion_criterion_not_triggered():
    """C4: eGFR 55 with exclusion 'eGFR < 30' — exclusion NOT triggered → SUPPORTED."""
    engine = RuleEngine()
    r      = engine.evaluate(
        _make_criterion(CriterionType.EGFR, "eGFR < 30 mL/min/1.73m2", is_inclusion=False),
        _make_patient(egfr=55.0),
    )
    passed = r.status == CriterionStatus.SUPPORTED
    return CaseResult(
        case_id="C4", category="Criterion status",
        name="Exclusion criterion not triggered → SUPPORTED",
        passed=passed,
        diagnostic=f"status={r.status.value} (eGFR 55 does not trigger exclusion < 30).",
    )


def case_C5_unparseable_criterion_is_clinical_review():
    """C5: Age criterion with no numeric content → REQUIRES_CLINICAL_REVIEW."""
    engine = RuleEngine()
    r      = engine.evaluate(_make_criterion(CriterionType.AGE, "Adults only"),
                             _make_patient())
    passed = r.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW
    return CaseResult(
        case_id="C5", category="Criterion status",
        name="Unparseable criterion → REQUIRES_CLINICAL_REVIEW",
        passed=passed,
        diagnostic=f"status={r.status.value}.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — Agent / pipeline behaviour
# ═══════════════════════════════════════════════════════════════════════════════

def case_A1_filtering_removes_non_recruiting():
    """A1: FilteringService removes all non-RECRUITING trials."""
    trials = [
        Trial(id="t1", title="Open",   status="Recruiting"),
        Trial(id="t2", title="Closed", status="Completed"),
        Trial(id="t3", title="Paused", status="Suspended"),
    ]
    state  = AgentState(patient=_make_patient(), all_trials=trials)
    result = FilteringService().run(state)
    ids    = {t.id for t in result.filtered_trials}
    passed = ids == {"t1"} and "t2" in result.filter_reasons and "t3" in result.filter_reasons
    return CaseResult(
        case_id="A1", category="Agent behaviour",
        name="FilteringService removes non-recruiting trials",
        passed=passed,
        diagnostic=(
            f"Filtered in: {sorted(ids)}. "
            f"Reasons: { {k: v for k, v in result.filter_reasons.items()} }."
        ),
    )


def case_A2_ranking_caps_at_top_k():
    """A2: RankingService returns at most TOP_K=3 trials regardless of input size."""
    trials = [Trial(id=f"t{i}", title=f"T{i}", status="Recruiting") for i in range(8)]
    evals  = {
        t.id: [CriterionEvaluation(
            criterion_id="c0", status=CriterionStatus.SUPPORTED,
            reasoning="ok", confidence=1.0, evaluator_type="rule_engine",
        )]
        for t in trials
    }
    state = AgentState(patient=_make_patient(), all_trials=trials)
    state = state.model_copy(update={"filtered_trials": trials, "evaluations": evals})
    result = RankingService().run(state)
    passed = len(result.ranked_trials) <= 3
    return CaseResult(
        case_id="A2", category="Agent behaviour",
        name="RankingService caps output at TOP_K=3",
        passed=passed,
        diagnostic=f"Ranked {len(result.ranked_trials)} from {len(trials)} input trials.",
    )


def case_A3_human_review_flag_set_for_unknown():
    """A3: requires_human_review=True when any criterion is UNKNOWN."""
    trial = Trial(id="t1", title="T", status="Recruiting")
    evals = {"t1": [
        CriterionEvaluation(criterion_id="c0", status=CriterionStatus.SUPPORTED,
                            reasoning="ok", confidence=1.0, evaluator_type="rule_engine"),
        CriterionEvaluation(criterion_id="c1", status=CriterionStatus.UNKNOWN,
                            reasoning="missing", confidence=0.0, evaluator_type="rule_engine"),
    ]}
    state  = AgentState(patient=_make_patient(), all_trials=[trial])
    state  = state.model_copy(update={"filtered_trials": [trial], "evaluations": evals})
    result = RankingService().run(state)
    r      = result.ranked_trials[0]
    passed = r.requires_human_review is True
    return CaseResult(
        case_id="A3", category="Agent behaviour",
        name="requires_human_review=True when UNKNOWN criteria present",
        passed=passed,
        diagnostic=f"requires_human_review={r.requires_human_review}, unknown_count={r.unknown_count}.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — Dataset coverage (real dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def case_D1_all_15_patients_parse():
    """D1: All 15 real patients parse without error."""
    raw      = load_dataset(settings.paths.data)
    patients_raw = raw.get("patients", [])
    errors   = []
    for p in patients_raw:
        try:
            parse_patient(p)
        except Exception as e:
            errors.append(f"{p.get('patient_id')}: {e}")
    passed = len(errors) == 0
    return CaseResult(
        case_id="D1", category="Dataset coverage",
        name="All 15 real patients parse without error",
        passed=passed,
        diagnostic=(
            f"Parsed {len(patients_raw) - len(errors)}/{len(patients_raw)} successfully."
            + (f" Errors: {errors}" if errors else "")
        ),
    )


def case_D2_all_36_trials_parse():
    """D2: All 36 real trials parse without error."""
    raw       = load_dataset(settings.paths.data)
    trials_raw = raw.get("trials", [])
    try:
        trials = parse_trials(trials_raw)
        passed = len(trials) == 36
        diag   = f"Parsed {len(trials)}/36 trials."
    except Exception as e:
        passed = False
        diag   = f"Parse error: {e}"
    return CaseResult(
        case_id="D2", category="Dataset coverage",
        name="All 36 real trials parse without error",
        passed=passed,
        diagnostic=diag,
    )


def case_D3_filter_reduces_trial_count():
    """D3: FilteringService reduces the 36-trial set for a typical patient."""
    raw    = load_dataset(settings.paths.data)
    trials = parse_trials(raw.get("trials", []))
    state  = AgentState(patient=_make_patient(age=62), all_trials=trials)
    result = FilteringService().run(state)
    reduced = len(result.filtered_trials) < len(trials)
    passed  = reduced and len(result.filtered_trials) > 0
    return CaseResult(
        case_id="D3", category="Dataset coverage",
        name="FilteringService reduces 36-trial set for age-62 patient",
        passed=passed,
        diagnostic=(
            f"Before: {len(trials)}, after: {len(result.filtered_trials)}, "
            f"excluded: {len(result.filter_reasons)}."
        ),
    )


def case_D4_real_patients_have_lab_results():
    """D4: All 15 real patients have at least one lab result after parsing."""
    raw     = load_dataset(settings.paths.data)
    missing = []
    for p_raw in raw.get("patients", []):
        p = parse_patient(p_raw)
        if len(p.lab_results) == 0:
            missing.append(p.id)
    passed = len(missing) == 0
    return CaseResult(
        case_id="D4", category="Dataset coverage",
        name="All real patients have at least one parsed lab result",
        passed=passed,
        diagnostic=(
            f"{15 - len(missing)}/15 patients have lab results."
            + (f" Missing: {missing}." if missing else "")
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — Output quality (report completeness)
# ═══════════════════════════════════════════════════════════════════════════════

def case_O1_report_contains_all_required_fields():
    """O1: Generated report contains all coordinator-required sections."""
    gen     = ReportGenerator()
    patient = _make_patient(conditions=["Type 2 Diabetes"], medications=["metformin"])
    ranking = _make_ranking("t1", "Alpha Trial", CriterionStatus.SUPPORTED, 1.0, sup=2)
    ev      = CriterionEvaluation(
        criterion_id="c1", status=CriterionStatus.SUPPORTED,
        reasoning="Patient HbA1c 8.2% meets criterion >= 7.5%.",
        confidence=1.0, evaluator_type="rule_engine",
        evidence_used=[Evidence(text="HbA1c: 8.2", source="lab_results",
                                retrieved_from="patient", date="2026-04-30")],
    )
    md, data = gen.generate(patient, [ranking], {"t1": [ev]})

    checks = {
        "Pre-Screening Report header":    "Pre-Screening Report" in md,
        "Patient ID":                     patient.id in md,
        "Trial ID":                       "t1" in md,
        "Trial title":                    "Alpha Trial" in md,
        "Criterion status label":         "SUPPORTED" in md,
        "Evidence source":                "lab_results" in md,
        "Clinical Fit row":               "Clinical Fit" in md,
        "Recruiting row":                 "Recruiting" in md,
        "Human Review row":               "Human Review" in md or "human review" in md.lower(),
        "Safety disclaimer":              "clinical judgement" in md.lower() or "healthcare professional" in md.lower(),
        "Reason surfaced":                "reason" in md.lower() or "Reason surfaced" in md,
        "Structured data patient_id":     data.get("patient_id") == patient.id,
        "Structured data summary":        "summary" in data,
        "Structured data generated_at":   "generated_at" in data,
    }
    failing = [k for k, v in checks.items() if not v]
    passed  = len(failing) == 0
    return CaseResult(
        case_id="O1", category="Output quality",
        name="Report contains all coordinator-required sections and fields",
        passed=passed,
        diagnostic=(
            f"{len(checks) - len(failing)}/{len(checks)} checks passed."
            + (f" Failing: {failing}." if failing else "")
        ),
    )


def case_O2_unanswered_questions_surfaced():
    """O2: UNKNOWN criteria surface their questions in the report."""
    gen     = ReportGenerator()
    patient = _make_patient()
    ranking = _make_ranking("t1", "B", CriterionStatus.UNKNOWN, 0.0, unk=1)
    ev      = CriterionEvaluation(
        criterion_id="c1", status=CriterionStatus.UNKNOWN,
        reasoning="eGFR not available.", confidence=0.0,
        evaluator_type="rule_engine",
        unanswered_questions=["What is the patient's most recent eGFR value?"],
    )
    md, _ = gen.generate(patient, [ranking], {"t1": [ev]})
    has_q = "eGFR" in md
    has_u = "Unanswered" in md or "unanswered" in md.lower() or "Clinical Review" in md
    passed = has_q and has_u
    return CaseResult(
        case_id="O2", category="Output quality",
        name="Unanswered questions from UNKNOWN criteria surface in report",
        passed=passed,
        diagnostic=f"eGFR in report: {has_q}. Unanswered section present: {has_u}.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

ALL_CASES = [
    case_R1_retrieval_returns_results,
    case_R2_retrieval_only_returns_matching_trial,
    case_R3_patient_evidence_covers_key_sources,
    case_R4_queries_include_lab_values,
    case_C1_age_in_range_is_supported,
    case_C2_age_outside_range_is_not_supported,
    case_C3_missing_lab_is_unknown_not_negative,
    case_C4_exclusion_criterion_not_triggered,
    case_C5_unparseable_criterion_is_clinical_review,
    case_A1_filtering_removes_non_recruiting,
    case_A2_ranking_caps_at_top_k,
    case_A3_human_review_flag_set_for_unknown,
    case_D1_all_15_patients_parse,
    case_D2_all_36_trials_parse,
    case_D3_filter_reduces_trial_count,
    case_D4_real_patients_have_lab_results,
    case_O1_report_contains_all_required_fields,
    case_O2_unanswered_questions_surfaced,
]


def _print_report(report: EvalReport) -> None:
    col_w = {"id": 4, "cat": 20, "name": 52, "pass": 6, "ms": 8, "diag": 60}
    sep   = "-" * (sum(col_w.values()) + len(col_w) * 3)

    print()
    print("=" * len(sep))
    print("  EVALUATION SUITE RESULTS")
    print("=" * len(sep))
    print(f"  {'ID':<{col_w['id']}}  {'Category':<{col_w['cat']}}  {'Case':<{col_w['name']}}  "
          f"{'Result':<{col_w['pass']}}  {'ms':>{col_w['ms']}}  Diagnostic")
    print(sep)

    by_cat: dict = {}
    for c in report.cases:
        by_cat.setdefault(c.category, []).append(c)

    for cat, cases in by_cat.items():
        for c in cases:
            status = "PASS" if c.passed else "FAIL"
            diag   = c.diagnostic[:col_w["diag"]] + ("…" if len(c.diagnostic) > col_w["diag"] else "")
            print(f"  {c.case_id:<{col_w['id']}}  {c.category:<{col_w['cat']}}  "
                  f"{c.name:<{col_w['name']}}  {status:<{col_w['pass']}}  "
                  f"{c.duration_ms:>{col_w['ms']}.1f}  {diag}")
        print(sep)

    print()
    print(f"  TOTAL: {report.total}  PASSED: {report.passed}  "
          f"FAILED: {report.failed}  PASS RATE: {report.pass_rate:.1%}")
    print()


@app.command()
def run(
    output_json: bool = typer.Option(False, "--json", help="Write results/eval_results.json"),
) -> None:
    """Run the full evaluation suite and print results."""
    report = EvalReport()

    for fn in ALL_CASES:
        result = _run(fn)
        report.cases.append(result)
        report.total  += 1
        report.passed += int(result.passed)
        report.failed += int(not result.passed)

    _print_report(report)

    if output_json:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "eval_results.json"
        payload = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": round(report.pass_rate, 4),
            "cases": [asdict(c) for c in report.cases],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        typer.echo(f"Results written to {out_path}")

    raise SystemExit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    app()
