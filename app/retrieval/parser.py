"""
Parse the real JSON dataset into typed domain models.

Dataset structure (Type2-Diabetes-Trial-Agent-Dataset.json):
  - patients[]  — synthetic FHIR-normalised records
      patient_id, demographics.age_at_reference_date, demographics.sex
      observations[]  — {type, value, unit, effective_date, source_id}
      medications[]   — {name, status, start_date, source_id}
      conditions[]    — {display, source_id}
  - trials[]    — derived from public ClinicalTrials.gov records
      nct_id, brief_title, overall_status
      minimum_age_years, maximum_age_years
      eligibility_text  — raw free-text inclusion/exclusion block

Criterion type detection uses a priority-ordered keyword map (same as spec).
"""

import re
from datetime import date as DateType
from typing import List

from app.models.patient import Patient, LabResult
from app.models.trial import Trial, Criterion, CriterionType

# ── Keyword map ───────────────────────────────────────────────────────────────

_CRITERION_TYPE_MAP: List[tuple] = [
    (CriterionType.AGE,        ["age", "years old", "year old"]),
    (CriterionType.HBA1C,      ["hba1c", "hemoglobin a1c", "a1c"]),
    (CriterionType.EGFR,       ["egfr", "renal", "kidney", "creatinine", "gfr"]),
    (CriterionType.RECRUITING, ["recruiting", "enrollment"]),
    (CriterionType.MEDICATION, ["medication", "drug", "insulin", "metformin", "treatment",
                                "therapy", "agent"]),
    (CriterionType.CONDITION,  ["diagnosis", "condition", "disease", "diabetes", "type 2"]),
]


def _detect_criterion_type(description: str) -> CriterionType:
    lower = description.lower()
    for ctype, keywords in _CRITERION_TYPE_MAP:
        if any(kw in lower for kw in keywords):
            return ctype
    return CriterionType.OTHER


# ── Eligibility text parser ───────────────────────────────────────────────────

def _parse_eligibility_text(text: str) -> tuple[List[Criterion], List[Criterion]]:
    """
    Split the raw eligibility_text blob into inclusion and exclusion Criterion objects.

    The text follows the standard ClinicalTrials.gov format:
        Inclusion Criteria:
        1. ...
        2. ...
        Exclusion Criteria:
        1. ...
    """
    if not text:
        return [], []

    inclusion_raw: List[str] = []
    exclusion_raw: List[str] = []
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if "inclusion criteria" in low:
            current = "inclusion"
            continue
        if "exclusion criteria" in low:
            current = "exclusion"
            continue
        # Numbered or lettered list item
        item = re.sub(r"^[\d]+\.\s*", "", stripped).strip()
        if not item:
            continue
        if current == "inclusion":
            inclusion_raw.append(item)
        elif current == "exclusion":
            exclusion_raw.append(item)

    inclusion_criteria = [
        Criterion(
            id=f"inc_{i}",
            type=_detect_criterion_type(text),
            description=text,
            is_inclusion=True,
        )
        for i, text in enumerate(inclusion_raw)
    ]
    exclusion_criteria = [
        Criterion(
            id=f"exc_{i}",
            type=_detect_criterion_type(text),
            description=text,
            is_inclusion=False,
        )
        for i, text in enumerate(exclusion_raw)
    ]
    return inclusion_criteria, exclusion_criteria


def parse_criterion(raw: dict, idx: int, is_inclusion: bool) -> Criterion:
    """Parse a single criterion dict (used when criteria are pre-split in the JSON)."""
    description = (
        raw.get("description") or raw.get("criterion")
        or raw.get("text") or raw.get("value") or ""
    ).strip()
    criterion_id = str(
        raw.get("id") or raw.get("criterion_id")
        or f"{'inc' if is_inclusion else 'exc'}_{idx}"
    )
    return Criterion(
        id=criterion_id,
        type=_detect_criterion_type(description),
        description=description,
        is_inclusion=is_inclusion,
    )


# ── Trial parser ──────────────────────────────────────────────────────────────

def parse_trials(raw: list) -> List[Trial]:
    """Parse a list of raw trial dicts (real dataset format or generic)."""
    trials: List[Trial] = []
    for item in raw:
        trial_id = str(
            item.get("nct_id") or item.get("id")
            or item.get("trial_id") or ""
        )
        title = (
            item.get("brief_title") or item.get("official_title")
            or item.get("title") or ""
        )
        phase       = item.get("phase") or item.get("study_phase") or ""
        status      = (
            item.get("overall_status") or item.get("status")
            or item.get("recruitment_status") or ""
        )
        description = (
            item.get("brief_summary") or item.get("description")
            or item.get("eligibility_text") or ""
        )

        # --- Criteria: try pre-split first, fall back to eligibility_text ---
        inclusion_raw = item.get("inclusion_criteria") or []
        exclusion_raw = item.get("exclusion_criteria") or []

        if not inclusion_raw and not exclusion_raw:
            # Try nested eligibility block
            eligibility = item.get("eligibility") or {}
            inclusion_raw = eligibility.get("inclusion_criteria") or []
            exclusion_raw = eligibility.get("exclusion_criteria") or []

        if not inclusion_raw and not exclusion_raw:
            # Try flat criteria list with is_inclusion flag
            for c in item.get("criteria") or []:
                flag = c.get("is_inclusion", c.get("inclusion", True))
                (inclusion_raw if flag else exclusion_raw).append(c)

        if not inclusion_raw and not exclusion_raw:
            # Parse from raw eligibility_text blob (real dataset format)
            eligibility_text = item.get("eligibility_text", "")
            inc_parsed, exc_parsed = _parse_eligibility_text(eligibility_text)
            trials.append(Trial(
                id=trial_id, title=title, phase=phase, status=status,
                inclusion_criteria=inc_parsed, exclusion_criteria=exc_parsed,
                description=description,
            ))
            continue

        def _normalise(lst):
            return [{"description": e} if isinstance(e, str) else e for e in lst]

        trials.append(Trial(
            id=trial_id, title=title, phase=phase, status=status,
            inclusion_criteria=[
                parse_criterion(c, i, True)
                for i, c in enumerate(_normalise(inclusion_raw))
            ],
            exclusion_criteria=[
                parse_criterion(c, i, False)
                for i, c in enumerate(_normalise(exclusion_raw))
            ],
            description=description,
        ))

    return trials


# ── Patient parser ────────────────────────────────────────────────────────────

def parse_patient(raw: dict) -> Patient:
    """
    Parse a patient dict.  Handles both:
    - Real dataset format  (patient_id, demographics{}, observations[], medications[])
    - Generic format       (id/age/gender/lab_results[]/conditions[]/medications[])
    """
    # --- Real dataset format ---
    if "demographics" in raw or "observations" in raw:
        demo   = raw.get("demographics") or {}
        pid    = str(raw.get("patient_id") or demo.get("patient_id") or "")
        age    = int(
            demo.get("age_at_reference_date")
            or raw.get("age") or 0
        )
        gender = str(demo.get("sex") or demo.get("gender") or demo.get("administrative_gender") or raw.get("gender") or "")

        # Conditions from conditions[]
        conditions: List[str] = [
            c.get("display") or c.get("name") or ""
            for c in (raw.get("conditions") or [])
            if c.get("display") or c.get("name")
        ]

        # Medications: active ones only
        medications: List[str] = [
            m.get("name") or ""
            for m in (raw.get("medications") or [])
            if (m.get("status", "").lower() in ("active", "") or m.get("status") is None)
            and m.get("name")
        ]

        # Lab results from observations[]
        lab_results: List[LabResult] = []
        for obs in raw.get("observations") or []:
            obs_type = obs.get("type") or obs.get("name") or ""
            value    = obs.get("value")
            if value is None:
                continue
            raw_date = obs.get("effective_date")
            parsed_date = None
            if raw_date:
                try:
                    parsed_date = DateType.fromisoformat(raw_date)
                except (ValueError, TypeError):
                    pass
            lab_results.append(LabResult(
                test=obs_type,
                value=float(value),
                unit=obs.get("unit") or "",
                date=parsed_date,
                source_id=obs.get("source_id") or "",
            ))

        medical_history = raw.get("medical_history") or ""
        return Patient(
            id=pid, age=age, gender=gender,
            conditions=conditions, medications=medications,
            lab_results=lab_results, medical_history=medical_history,
        )

    # --- Generic / test format ---
    lab_results = []
    for lr in (raw.get("lab_results") or raw.get("labs") or []):
        if isinstance(lr, dict):
            lab_results.append(LabResult(
                test=lr.get("test") or lr.get("name") or "",
                value=float(lr.get("value") or lr.get("result") or 0.0),
                unit=lr.get("unit") or lr.get("units") or "",
                date=lr.get("date"),
            ))
    return Patient(
        id=str(raw.get("id") or raw.get("patient_id") or ""),
        age=int(raw.get("age") or 0),
        gender=str(raw.get("gender") or raw.get("sex") or ""),
        conditions=raw.get("conditions") or raw.get("diagnoses") or [],
        medications=raw.get("medications") or raw.get("drugs") or [],
        lab_results=lab_results,
        medical_history=raw.get("medical_history") or raw.get("history") or "",
    )
