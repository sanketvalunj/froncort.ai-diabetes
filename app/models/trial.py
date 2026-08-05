# Pydantic models representing clinical trial data structures.
from pydantic import BaseModel
from typing import List
from enum import Enum


class CriterionType(str, Enum):
    AGE = "age"
    HBA1C = "hba1c"
    EGFR = "egfr"
    RECRUITING = "recruiting"
    MEDICATION = "medication"
    CONDITION = "condition"
    OTHER = "other"


class Criterion(BaseModel):
    id: str
    type: CriterionType
    description: str
    is_inclusion: bool


class Trial(BaseModel):
    id: str
    title: str
    phase: str = ""
    status: str = ""
    inclusion_criteria: List[Criterion] = []
    exclusion_criteria: List[Criterion] = []
    description: str = ""
