# Contains prompt templates for LLM interactions.
# Compact criterion-evaluation prompt.
# Target: 300–500 input tokens total (template ~80 tokens + criterion + patient fields + evidence).
# Response shape: state / reason (≤20 words) / evidence_ids only — no verbose fields.
CRITERION_EVAL_PROMPT = """Evaluate clinical trial eligibility for one criterion.

Criterion ({criterion_type}): {criterion_description}
Patient: {patient_fields}
Evidence: {evidence_json}

States: SUPPORTED | NOT_SUPPORTED | UNKNOWN | CONFLICTING_EVIDENCE | REQUIRES_CLINICAL_REVIEW
- UNKNOWN = needed data absent; REQUIRES_CLINICAL_REVIEW = needs clinician.

Reply JSON only:
{{"state":"<STATE>","reason":"<≤20 words>","evidence_ids":[<ids or []>]}}"""
