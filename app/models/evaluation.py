# Pydantic models for evaluation results and trial rankings.
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class CriterionStatus(str, Enum):
    """
    Assignment-defined criterion states.

    SUPPORTED             — available evidence supports the criterion.
    NOT_SUPPORTED         — available evidence does not support the criterion.
    UNKNOWN               — the data needed to evaluate the criterion is not available.
    CONFLICTING_EVIDENCE  — available evidence does not point to one clear conclusion.
    REQUIRES_CLINICAL_REVIEW — cannot be resolved without clinical judgement.
    """
    SUPPORTED                = "SUPPORTED"
    NOT_SUPPORTED            = "NOT_SUPPORTED"
    UNKNOWN                  = "UNKNOWN"
    CONFLICTING_EVIDENCE     = "CONFLICTING_EVIDENCE"
    REQUIRES_CLINICAL_REVIEW = "REQUIRES_CLINICAL_REVIEW"


class Evidence(BaseModel):
    text: str
    source: str
    relevance_score: float = 1.0
    retrieved_from: Literal["patient", "trial"] = "patient"
    # Optional evidence identifier that traces back to the source JSON
    evidence_id: str = ""
    date: str = ""


class CriterionEvaluation(BaseModel):
    criterion_id: str
    status: CriterionStatus
    reasoning: str
    evidence_used: List[Evidence] = []
    confidence: float = Field(ge=0.0, le=1.0)
    evaluator_type: str
    # Explicit questions raised when status is UNKNOWN or REQUIRES_CLINICAL_REVIEW
    unanswered_questions: List[str] = []


class TrialRanking(BaseModel):
    trial_id: str
    title: str
    # Overall clinical fit (independent of recruiting status)
    clinical_fit: CriterionStatus
    # Recruiting status as a separate field
    is_recruiting: bool
    score: float
    supported_count: int
    not_supported_count: int
    unknown_count: int
    conflicting_count: int
    review_count: int
    total_criteria: int
    # Human-review flag required by the assignment
    requires_human_review: bool
    # Short reason this trial was surfaced
    reason_surfaced: str = ""
