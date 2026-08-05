# Pydantic models representing patient profiles and medical data.
from pydantic import BaseModel
from datetime import date as DateType
from typing import List, Optional


class LabResult(BaseModel):
    test: str
    value: float
    unit: str
    # 'date' is aliased to 'DateType' to avoid Python 3.14 PEP 649 lazy
    # annotation resolution shadowing datetime.date with the field's own
    # default value (None), which caused Pydantic to infer NoneType only.
    date: Optional[DateType] = None
    # Dataset source_id (UUID from the real JSON) — preserved for evidence
    # traceability so the report can point back to the originating record.
    source_id: str = ""


class Patient(BaseModel):
    id: str
    age: int
    gender: str
    conditions: List[str] = []
    medications: List[str] = []
    lab_results: List[LabResult] = []
    medical_history: str = ""
