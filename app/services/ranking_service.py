# Ranks evaluated clinical trials based on patient fit.
"""
RankingService — Stage 4 of the pipeline.

Scores and sorts trials. Caps output at TOP_K (3) per the assignment.
Clinical fit is kept separate from recruiting status.
"""

from typing import List

from app.models.evaluation import CriterionStatus, TrialRanking
from app.models.state import AgentState
from app.utils.logger import get_logger

log = get_logger(__name__)

TOP_K = 3  # assignment: return no more than three potential matches

# How each status contributes to the clinical fit score
_SCORE_WEIGHT = {
    CriterionStatus.SUPPORTED:                 1.0,
    CriterionStatus.NOT_SUPPORTED:            -2.0,   # hard negative
    CriterionStatus.UNKNOWN:                   0.0,   # missing data — no penalty
    CriterionStatus.CONFLICTING_EVIDENCE:      0.0,   # neutral, needs review
    CriterionStatus.REQUIRES_CLINICAL_REVIEW:  0.0,   # human to decide
}


def _clinical_fit(supported: int, not_supported: int, total: int) -> CriterionStatus:
    """Derive overall clinical fit from per-criterion counts."""
    if not_supported > 0:
        return CriterionStatus.NOT_SUPPORTED
    if supported > 0 and supported == total:
        return CriterionStatus.SUPPORTED
    if supported > 0:
        return CriterionStatus.UNKNOWN          # some supported, none explicitly not
    return CriterionStatus.UNKNOWN


def _score(evals) -> float:
    total = len(evals)
    if total == 0:
        return 0.0
    return sum(_SCORE_WEIGHT.get(e.status, 0.0) for e in evals) / total


def _reason_surfaced(trial, evals) -> str:
    """One-sentence explanation of why this trial was surfaced."""
    supported = sum(1 for e in evals if e.status == CriterionStatus.SUPPORTED)
    total     = len(evals)
    return (
        f"{supported}/{total} evaluated criteria supported; "
        f"trial is actively recruiting."
    )


class RankingService:
    def __init__(self, settings=None):
        self._settings = settings

    def run(self, state: AgentState) -> AgentState:
        log.info("ranking_start", trials=len(state.filtered_trials),
                 trace_id=state.trace_id)

        rankings: List[TrialRanking] = []

        for trial in state.filtered_trials:
            evals = state.evaluations.get(trial.id, [])
            if not evals:
                log.warning("ranking_no_evaluations", trial_id=trial.id,
                            trace_id=state.trace_id)
                continue

            supported    = sum(1 for e in evals if e.status == CriterionStatus.SUPPORTED)
            not_sup      = sum(1 for e in evals if e.status == CriterionStatus.NOT_SUPPORTED)
            unknown      = sum(1 for e in evals if e.status == CriterionStatus.UNKNOWN)
            conflicting  = sum(1 for e in evals if e.status == CriterionStatus.CONFLICTING_EVIDENCE)
            review       = sum(1 for e in evals if e.status == CriterionStatus.REQUIRES_CLINICAL_REVIEW)
            total        = len(evals)

            # Human-review flag: any UNKNOWN or REQUIRES_CLINICAL_REVIEW triggers it
            human_review = (unknown + conflicting + review) > 0

            rankings.append(TrialRanking(
                trial_id=trial.id,
                title=trial.title,
                clinical_fit=_clinical_fit(supported, not_sup, total),
                is_recruiting=(trial.status.upper() == "RECRUITING"),
                score=_score(evals),
                supported_count=supported,
                not_supported_count=not_sup,
                unknown_count=unknown,
                conflicting_count=conflicting,
                review_count=review,
                total_criteria=total,
                requires_human_review=human_review,
                reason_surfaced=_reason_surfaced(trial, evals),
            ))

        # Sort by score descending; cap at TOP_K
        rankings.sort(key=lambda r: r.score, reverse=True)
        top_rankings = rankings[:TOP_K]

        log.info("ranking_complete", total=len(rankings),
                 returned=len(top_rankings), trace_id=state.trace_id)

        return state.model_copy(update={"ranked_trials": top_rankings})
