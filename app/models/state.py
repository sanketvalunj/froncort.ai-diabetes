# Pydantic model representing the state of the LangGraph workflow.
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timezone
from uuid import uuid4

from .patient import Patient
from .trial import Trial
from .evaluation import Evidence, CriterionEvaluation, TrialRanking


class AgentState(BaseModel):
    # Input
    patient: Patient
    all_trials: List[Trial]

    # Stage 1 — Filtering
    filtered_trials: List[Trial] = Field(default_factory=list)
    filter_reasons: Dict[str, str] = Field(default_factory=dict)

    # Stage 2 — Retrieval
    patient_evidence: List[Evidence] = Field(default_factory=list)
    trial_evidence: Dict[str, List[Evidence]] = Field(default_factory=dict)

    # Stage 3 — Evaluation
    evaluations: Dict[str, List[CriterionEvaluation]] = Field(default_factory=dict)

    # Stage 4 — Ranking
    ranked_trials: List[TrialRanking] = Field(default_factory=list)

    # Stage 5 — Reporting
    report_markdown: Optional[str] = None
    report_data: Optional[Dict] = None

    # Metadata
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    custom_metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
