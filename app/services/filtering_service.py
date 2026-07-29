import re
from typing import Dict, List, Tuple

from app.models.state import AgentState
from app.models.trial import CriterionType, Trial
from app.utils.logger import get_logger

log = get_logger(__name__)


def _parse_age_bounds(description: str):
    m = re.search(r"(\d+)\s*(?:[-–]|to|and)\s*(\d+)", description)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r">=?\s*(\d+)", description)
    if m:
        return int(m.group(1)), None
    m = re.search(r"<=?\s*(\d+)", description)
    if m:
        return None, int(m.group(1))
    return None, None


def _age_clearly_outside(age: int, criteria) -> bool:
    for c in criteria:
        if c.type != CriterionType.AGE:
            continue
        lo, hi = _parse_age_bounds(c.description)
        if lo is not None and age < lo:
            return True
        if hi is not None and age > hi:
            return True
    return False


class FilteringService:
    def __init__(self, settings=None):
        self._settings = settings

    def run(self, state: AgentState) -> AgentState:
        patient  = state.patient
        filtered: List[Trial] = []
        reasons:  Dict[str, str] = {}
        log.info("filtering_start", total_trials=len(state.all_trials),
                 trace_id=state.trace_id)
        for trial in state.all_trials:
            if trial.status.upper() != "RECRUITING":
                reasons[trial.id] = f"Status: {trial.status}"
                continue
            all_criteria = trial.inclusion_criteria + trial.exclusion_criteria
            if _age_clearly_outside(patient.age, all_criteria):
                reasons[trial.id] = f"Age {patient.age} outside criterion range"
                continue
            filtered.append(trial)
        log.info("filtering_complete", total=len(state.all_trials),
                 passed=len(filtered), excluded=len(reasons),
                 trace_id=state.trace_id)
        return state.model_copy(update={"filtered_trials": filtered,
                                        "filter_reasons": reasons})
