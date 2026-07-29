CRITERION_EVAL_PROMPT = """You are a clinical trial eligibility evaluator.

Criterion: {criterion_description}
Type: {criterion_type} criterion

Patient summary:
{patient_summary}

Supporting evidence:
{evidence_text}

Evaluate whether this patient meets this criterion.
Use exactly one of these states:
  SUPPORTED                — the available evidence supports the criterion
  NOT_SUPPORTED            — the available evidence does not support the criterion
  UNKNOWN                  — the data needed to evaluate the criterion is not available
  CONFLICTING_EVIDENCE     — the evidence does not point to one clear conclusion
  REQUIRES_CLINICAL_REVIEW — the criterion cannot be resolved without clinical judgement

Respond with valid JSON only:
{{
  "status": "<one of the five states above>",
  "reasoning": "one sentence plain-language explanation",
  "confidence": 0.0 to 1.0,
  "unanswered_questions": ["question if UNKNOWN or REQUIRES_CLINICAL_REVIEW, else empty list"]
}}
"""
