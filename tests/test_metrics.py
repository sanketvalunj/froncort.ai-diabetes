# Unit tests for the metrics module.
"""Tests for the metrics module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence, TrialRanking
from app.models.state import AgentState
from app.models.trial import Trial
from metrics.metrics import PipelineMetrics, compute_metrics
from tests.conftest import make_eval, make_ranking


def _state(patient, all_trials, filtered_trials=None, evaluations=None,
           ranked_trials=None, patient_evidence=None, trial_evidence=None,
           custom_metrics=None):
    return AgentState(patient=patient, all_trials=all_trials,
                      filtered_trials=filtered_trials or [],
                      evaluations=evaluations or {},
                      ranked_trials=ranked_trials or [],
                      patient_evidence=patient_evidence or [],
                      trial_evidence=trial_evidence or {},
                      custom_metrics=custom_metrics or {})


class TestThroughput:
    def test_total_trials_input(self, sample_patient, sample_trials):
        assert compute_metrics(_state(sample_patient, sample_trials))["total_trials_input"] == 3.0

    def test_trials_filtered_out(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials,
                                   filtered_trials=sample_trials[:2]))
        assert m["trials_filtered_out"] == 1.0

    def test_filter_rate(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials,
                                   filtered_trials=sample_trials[:1]))
        assert abs(m["filter_rate"] - 2/3) < 1e-9

    def test_filter_rate_zero_no_trials(self, sample_patient):
        assert compute_metrics(_state(sample_patient, []))["filter_rate"] == 0.0


class TestEvalQuality:
    def _s(self, patient, trials):
        evals = {
            "trial_001": [make_eval("c0", CriterionStatus.SUPPORTED),
                          make_eval("c1", CriterionStatus.SUPPORTED),
                          make_eval("c2", CriterionStatus.NOT_SUPPORTED)],
            "trial_002": [make_eval("c3", CriterionStatus.UNKNOWN)],
        }
        return _state(patient, trials, filtered_trials=trials[:2], evaluations=evals)

    def test_total_criteria(self, sample_patient, sample_trials):
        assert compute_metrics(self._s(sample_patient, sample_trials))["total_criteria_evaluated"] == 4.0

    def test_supported_count(self, sample_patient, sample_trials):
        assert compute_metrics(self._s(sample_patient, sample_trials))["supported_count"] == 2.0

    def test_unknown_count(self, sample_patient, sample_trials):
        assert compute_metrics(self._s(sample_patient, sample_trials))["unknown_count"] == 1.0

    def test_rates_zero_when_empty(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials))
        assert m["supported_rate"] == 0.0 and m["not_supported_rate"] == 0.0


class TestRankingMetrics:
    def test_counts(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials, ranked_trials=[
            make_ranking("t1", "A", CriterionStatus.SUPPORTED, 1.0, sup=2),
            make_ranking("t2", "B", CriterionStatus.NOT_SUPPORTED, -1.0, nsup=1),
        ]))
        assert m["ranked_supported_trials"] == 1.0
        assert m["ranked_not_supported_trials"] == 1.0

    def test_human_review_count(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials, ranked_trials=[
            make_ranking("t1", "A", CriterionStatus.UNKNOWN, 0.0, unk=1),
        ]))
        assert m["ranked_human_review_trials"] == 1.0


class TestEvaluatorDist:
    def test_counts(self, sample_patient, sample_trials):
        evals = {"t1": [
            make_eval("c0", CriterionStatus.SUPPORTED,        "rule_engine"),
            make_eval("c1", CriterionStatus.SUPPORTED,        "rule_engine"),
            make_eval("c2", CriterionStatus.UNKNOWN,          "llm_engine"),
            make_eval("c3", CriterionStatus.REQUIRES_CLINICAL_REVIEW, "clinical_review"),
            make_eval("c4", CriterionStatus.REQUIRES_CLINICAL_REVIEW, "llm_engine_error"),
        ]}
        m = compute_metrics(_state(sample_patient, sample_trials, evaluations=evals))
        assert m["rule_engine_evaluations"] == 2.0
        assert m["llm_engine_evaluations"]  == 1.0
        assert m["clinical_review_evaluations"] == 1.0
        assert m["llm_error_evaluations"]   == 1.0


class TestEvidence:
    def test_patient_evidence_count(self, sample_patient, sample_trials):
        ev = [Evidence(text=f"e{i}", source="t") for i in range(5)]
        m  = compute_metrics(_state(sample_patient, sample_trials, patient_evidence=ev))
        assert m["avg_patient_evidence_count"] == 5.0


class TestCustomMetrics:
    def test_preserved(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials,
                                   custom_metrics={"run_duration_ms": 1234.5}))
        assert m["run_duration_ms"] == 1234.5


class TestSave:
    def test_creates_json(self, tmp_path, sample_patient, sample_trials):
        state = _state(sample_patient, sample_trials)
        pm    = PipelineMetrics()
        path  = pm.save(pm.compute(state), state.patient.id, state.trace_id, output_dir=tmp_path)
        assert path.exists() and path.suffix == ".json"

    def test_valid_json(self, tmp_path, sample_patient, sample_trials):
        state   = _state(sample_patient, sample_trials)
        pm      = PipelineMetrics()
        path    = pm.save(pm.compute(state), state.patient.id, state.trace_id, output_dir=tmp_path)
        payload = json.loads(path.read_text())
        assert payload["patient_id"] == state.patient.id and "metrics" in payload

    def test_all_float(self, sample_patient, sample_trials):
        m = compute_metrics(_state(sample_patient, sample_trials))
        for k, v in m.items():
            assert isinstance(v, float), f"{k}: {type(v)}"
