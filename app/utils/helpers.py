import re
from datetime import datetime
from typing import Optional


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_numeric(text: str) -> Optional[float]:
    match = re.search(r"\d+\.?\d*", text)
    if match:
        return float(match.group())
    return None


def compare_values(value: float, operator: str, threshold: float) -> bool:
    ops = {
        ">":  value > threshold,
        ">=": value >= threshold,
        "<":  value < threshold,
        "<=": value <= threshold,
        "==": value == threshold,
    }
    if operator not in ops:
        raise ValueError(f"Unsupported operator: {operator!r}")
    return ops[operator]


def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")
