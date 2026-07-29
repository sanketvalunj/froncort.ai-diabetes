"""Tests for the services layer (Tasks 6–7) and LangGraph workflow."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models.evaluation import CriterionEvaluation, CriterionStatus, Evidence, TrialRanking
from app.models.patient import Patient, LabResult
from app.models.state import AgentState
from app.models.trial import Criterion, CriterionType, Trial
from app.services.filtering_service import FilteringService
from app.services.ranking_service import RankingService, TOP_K
from app.services.report_service import ReportService
from app.graph.workflow import build_workflow, run_workflow
from app.graph.nodes import make_nodes
from tests.conftest import make_eval, make_ranking


# ── FilteringService ──────────────────────────────────────────────────────────

class TestFilteringService:
    def test_removes_non_recruiting(self, sample_state):
        r = FilteringService().run(sample_state)
        assert all(t.status.upper() == "RECRUITING" for t in r.filtered_trials)

    def test_filter_reasons_cover_excluded(self, sample_state):
        r = FilteringService().run(sample_state)
        excluded = set(r.filter_reasons.keys())
        filtered = {t.id for t in r.filtered_trials}
        assert excluded | filtered == {t.id for t in sample_state.all_trials}

    def test_age_excluded(self, sample_patient):
        young = Trial(id="y", title="Young", status="Recruiting",
                      inclusion_criteria=[Criterion(id="c", type=CriterionType.AGE,
                          description="Aged 18-45 years", is_inclusion=True)])
        r = FilteringService().run(AgentState(patient=sample_patient, all_trials=[young]))
        assert len(r.filtered_trials) == 0 and "y" in r.filter_reasons

    def test_returns_agent_state(self, sample_state):
        assert isinstance(FilteringService().run(sample_state), AgentState)

    def test_original_unchanged(self, sample_state):
        before = len(sample_state.filtered_trials)
        FilteringService().run(sample_state)
        assert len(sample_state.filtered_trials) == before


# ── RankingService ────────────────────────────────────────────────────────────

class TestRankingService:
    def _make_trial(self, tid):
        return Trial(id=tid, title=f"Trial {tid}", status="Recruiting")

    def _state(self, patient, trials_evals):
        trials = [te[0] for te in trials_evals]
        state  = AgentState(patient=patient, all_trials=trials)
        evals  = {t.id: [make_eval(f"c{i}", s) for i, s in enumerate(statuses)]
                  for t, statuses in trials_evals}
        return state.model_copy(update={"filtered_trials": trials, "evaluations": evals})

    def test_all_supported_ranked_first(self, sample_patient):
        t_sup = self._make_trial("t_sup")
        t_mix = self._make_trial("t_mix")
        t_not = self._make_trial("t_not")
        state = self._state(sample_patient, [
            (t_sup, [CriterionStatus.SUPPORTED] * 3),
            (t_mix, [CriterionStatus.SUPPORTED, CriterionStatus.NOT_SUPPORTED]),
            (t_not, [CriterionStatus.NOT_SUPPORTED] * 3),
        ])
        ranked = RankingService().run(state).ranked_trials
        assert ranked[0].trial_id == "t_sup" and ranked[-1].trial_id == "t_not"

    def test_not_supported_clinical_fit(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient,
            [(t, [CriterionStatus.SUPPORTED, CriterionStatus.NOT_SUPPORTED])])
        assert RankingService().run(state).ranked_trials[0].clinical_fit == CriterionStatus.NOT_SUPPORTED

    def test_supported_clinical_fit(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient,
            [(t, [CriterionStatus.SUPPORTED, CriterionStatus.SUPPORTED])])
        assert RankingService().run(state).ranked_trials[0].clinical_fit == CriterionStatus.SUPPORTED

    def test_unknown_does_not_become_negative(self, sample_patient):
        """UNKNOWN criteria must not lower the clinical_fit to NOT_SUPPORTED."""
        t = self._make_trial("t")
        state = self._state(sample_patient, [(t, [CriterionStatus.UNKNOWN])])
        r = RankingService().run(state).ranked_trials[0]
        assert r.clinical_fit != CriterionStatus.NOT_SUPPORTED

    def test_human_review_flag_when_unknown(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient,
            [(t, [CriterionStatus.SUPPORTED, CriterionStatus.UNKNOWN])])
        assert RankingService().run(state).ranked_trials[0].requires_human_review is True

    def test_no_human_review_when_all_supported(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient, [(t, [CriterionStatus.SUPPORTED] * 3)])
        assert RankingService().run(state).ranked_trials[0].requires_human_review is False

    def test_capped_at_top_k(self, sample_patient):
        """RankingService must return at most TOP_K (3) trials."""
        trials_evals = [
            (self._make_trial(f"t{i}"), [CriterionStatus.SUPPORTED])
            for i in range(6)
        ]
        state  = self._state(sample_patient, trials_evals)
        result = RankingService().run(state)
        assert len(result.ranked_trials) <= TOP_K

    def test_counts_correct(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient, [(t, [
            CriterionStatus.SUPPORTED, CriterionStatus.SUPPORTED,
            CriterionStatus.NOT_SUPPORTED, CriterionStatus.UNKNOWN,
            CriterionStatus.REQUIRES_CLINICAL_REVIEW,
        ])])
        r = RankingService().run(state).ranked_trials[0]
        assert r.supported_count == 2
        assert r.not_supported_count == 1
        assert r.unknown_count == 1
        assert r.review_count == 1

    def test_is_recruiting_flag(self, sample_patient):
        t = self._make_trial("t")
        state = self._state(sample_patient, [(t, [CriterionStatus.SUPPORTED])])
        assert RankingService().run(state).ranked_trials[0].is_recruiting is True


# ── ReportService ─────────────────────────────────────────────────────────────

class TestReportService:
    def _settings(self, tmp_path):
        s = MagicMock()
        s.paths.reports = str(tmp_path / "reports")
        return s

    def _state(self, sample_patient):
        ranking = make_ranking("t1", "Test Trial", CriterionStatus.SUPPORTED,
                               score=1.0, sup=2)
        return AgentState(
            patient=sample_patient, all_trials=[],
            ranked_trials=[ranking],
            evaluations={"t1": [make_eval("c1", CriterionStatus.SUPPORTED)]},
        )

    def test_creates_md_file(self, sample_patient, tmp_path):
        ReportService(self._settings(tmp_path)).run(self._state(sample_patient))
        assert len(list((tmp_path / "reports").glob("*.md"))) == 1

    def test_filename_has_patient_id(self, sample_patient, tmp_path):
        ReportService(self._settings(tmp_path)).run(self._state(sample_patient))
        files = list((tmp_path / "reports").glob("*.md"))
        assert any(sample_patient.id in f.name for f in files)

    def test_report_markdown_in_state(self, sample_patient, tmp_path):
        r = ReportService(self._settings(tmp_path)).run(self._state(sample_patient))
        assert r.report_markdown and len(r.report_markdown) > 0

    def test_report_data_in_state(self, sample_patient, tmp_path):
        r = ReportService(self._settings(tmp_path)).run(self._state(sample_patient))
        assert isinstance(r.report_data, dict) and "patient_id" in r.report_data

    def test_creates_dir_if_missing(self, sample_patient, tmp_path):
        s = MagicMock()
        s.paths.reports = str(tmp_path / "deep" / "nested" / "reports")
        ReportService(s).run(self._state(sample_patient))
        assert (tmp_path / "deep" / "nested" / "reports").exists()


# ── LangGraph workflow ────────────────────────────────────────────────────────

def _ws(tmp_path):
    s = MagicMock()
    s.paths.vector_store        = "data/vector_store"
    s.embeddings.model          = "sentence-transformers/all-MiniLM-L6-v2"
    s.retrieval.top_k           = 5
    s.retrieval.score_threshold = 0.3
    s.paths.reports             = str(tmp_path / "reports")
    return s


class TestMakeNodes:
    def test_returns_five_nodes(self, tmp_path):
        nodes = make_nodes(_ws(tmp_path), MagicMock())
        assert set(nodes.keys()) == {"filter_node", "retrieval_node",
                                     "evaluation_node", "ranking_node", "report_node"}

    def test_all_callable(self, tmp_path):
        for name, fn in make_nodes(_ws(tmp_path), MagicMock()).items():
            assert callable(fn), f"{name} not callable"


class TestBuildWorkflow:
    def test_compiles(self, tmp_path):
        wf = build_workflow(_ws(tmp_path), MagicMock())
        assert wf is not None and hasattr(wf, "invoke")


class TestRunWorkflow:
    def _run(self, sample_patient, sample_trials, tmp_path):
        def mock_retrieval(state):
            return state.model_copy(update={"patient_evidence": [], "trial_evidence": {}})

        def mock_eval(state):
            evals = {t.id: [CriterionEvaluation(
                                criterion_id="c0",
                                status=CriterionStatus.SUPPORTED,
                                reasoning="mocked", confidence=1.0,
                                evaluator_type="mock")]
                     for t in state.filtered_trials}
            return state.model_copy(update={"evaluations": evals})

        with (
            patch("app.services.retrieval_service.RetrievalService.run",
                  side_effect=mock_retrieval),
            patch("app.services.evaluation_service.EvaluationService.run",
                  side_effect=mock_eval),
        ):
            initial = AgentState(patient=sample_patient, all_trials=sample_trials)
            return run_workflow(_ws(tmp_path), MagicMock(), initial)

    def test_returns_agent_state(self, sample_patient, sample_trials, tmp_path):
        final = self._run(sample_patient, sample_trials, tmp_path)
        assert isinstance(final, AgentState) and final.report_markdown

    def test_creates_report_file(self, sample_patient, sample_trials, tmp_path):
        self._run(sample_patient, sample_trials, tmp_path)
        assert len(list((tmp_path / "reports").glob("*.md"))) >= 1

    def test_completed_trial_not_ranked(self, sample_patient, sample_trials, tmp_path):
        final = self._run(sample_patient, sample_trials, tmp_path)
        assert "trial_003" not in {r.trial_id for r in final.ranked_trials}

    def test_ranked_at_most_top_k(self, sample_patient, sample_trials, tmp_path):
        final = self._run(sample_patient, sample_trials, tmp_path)
        assert len(final.ranked_trials) <= TOP_K
