"""
ReportGenerator — produces the coordinator-facing Markdown eligibility report.

Per-trial output includes (as required by the assignment):
  - trial identifier and title
  - short reason the trial was surfaced
  - each evaluated criterion with its CriterionStatus
  - patient evidence identifiers and dates used
  - plain-language explanation of each conclusion
  - unanswered questions / items requiring clinical review
  - clinical fit shown separately from recruiting status
  - human-review flag
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.models.evaluation import CriterionEvaluation, CriterionStatus, TrialRanking
from app.models.patient import Patient

_STATUS_BADGE = {
    CriterionStatus.SUPPORTED:                "✅ SUPPORTED",
    CriterionStatus.NOT_SUPPORTED:            "❌ NOT SUPPORTED",
    CriterionStatus.UNKNOWN:                  "❓ UNKNOWN",
    CriterionStatus.CONFLICTING_EVIDENCE:     "⚡ CONFLICTING EVIDENCE",
    CriterionStatus.REQUIRES_CLINICAL_REVIEW: "🔍 REQUIRES CLINICAL REVIEW",
}

_STATUS_ICON = {
    CriterionStatus.SUPPORTED:                "✅",
    CriterionStatus.NOT_SUPPORTED:            "❌",
    CriterionStatus.UNKNOWN:                  "❓",
    CriterionStatus.CONFLICTING_EVIDENCE:     "⚡",
    CriterionStatus.REQUIRES_CLINICAL_REVIEW: "🔍",
}

_FIT_BADGE = {
    CriterionStatus.SUPPORTED:                "✅ Likely eligible",
    CriterionStatus.NOT_SUPPORTED:            "❌ Likely ineligible",
    CriterionStatus.UNKNOWN:                  "❓ Insufficient data",
    CriterionStatus.CONFLICTING_EVIDENCE:     "⚡ Conflicting evidence",
    CriterionStatus.REQUIRES_CLINICAL_REVIEW: "🔍 Requires clinical review",
}


def _format_evidence(evidence_list) -> str:
    """
    Convert a list of Evidence objects into a concise, human-readable string
    suitable for a Markdown table cell.

    Rules:
    - Show the test/observation name capitalised, not the raw source field.
    - Append the date in parentheses when present.
    - For medication evidence, prefix with "Medication:".
    - Never show raw UUIDs or internal source_id values.
    - Deduplicate identical labels.
    - Return "—" when the list is empty.
    """
    if not evidence_list:
        return "—"

    seen: set = set()
    parts = []
    for e in evidence_list:
        src = e.source.lower()
        text_lower = e.text.lower()

        # Derive a readable label from the evidence text or source
        if "medication" in src or "medication" in text_lower:
            # e.g. "Medication: Empagliflozin"
            name_part = e.text.split(":", 1)[-1].strip() if ":" in e.text else e.text.strip()
            label = f"Medication: {name_part}" if name_part else "Medication"
        elif "lab" in src or any(kw in text_lower for kw in ("hba1c", "a1c", "egfr", "gfr", "bmi", "glucose", "creatinine")):
            name_part = e.text.split(":", 1)[0].strip() if ":" in e.text else e.source
            label = f"{name_part} (lab_results)"
            if e.date:
                label += f" ({e.date})"
        else:
            # Generic: capitalise source, add date
            label = e.source.replace("_", " ").title()
            if e.date:
                label += f" ({e.date})"

        if label not in seen:
            seen.add(label)
            parts.append(label)

    return "; ".join(parts) if parts else "—"


class ReportGenerator:
    def generate(
        self,
        patient: Patient,
        ranked_trials: List[TrialRanking],
        evaluations: Dict[str, List[CriterionEvaluation]],
        filter_reasons: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict]:
        now = datetime.now(timezone.utc)
        sections = [
            self._header(patient, now),
            self._executive_summary(ranked_trials),
            self._trial_details(ranked_trials, evaluations),
        ]
        if filter_reasons:
            sections.append(self._filtered_appendix(filter_reasons))
        sections.append(self._footer())
        markdown    = "\n\n".join(sections)
        report_data = self._build_data(patient, ranked_trials, evaluations, filter_reasons, now)
        return markdown, report_data

    # ── Header ────────────────────────────────────────────────────────────────

    @staticmethod
    def _header(patient: Patient, now: datetime) -> str:
        labs_md = ""
        if patient.lab_results:
            lab_lines = "\n".join(
                f"  - {lr.test}: **{lr.value} {lr.unit}**"
                + (f" _(recorded {lr.date})_" if lr.date else "")
                for lr in patient.lab_results
            )
            labs_md = f"\n\n**Lab Results:**\n{lab_lines}"
        conditions  = ", ".join(patient.conditions)  if patient.conditions  else "—"
        medications = ", ".join(patient.medications) if patient.medications else "—"
        return (
            f"# Clinical Trial Pre-Screening Report\n\n"
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n---\n\n"
            f"## Patient Profile\n\n"
            f"| Field        | Value |\n|:-------------|:------|\n"
            f"| Patient ID   | `{patient.id}` |\n"
            f"| Age          | {patient.age} |\n"
            f"| Gender       | {patient.gender} |\n"
            f"| Conditions   | {conditions} |\n"
            f"| Medications  | {medications} |"
            f"{labs_md}\n\n"
            f"> **Safety note:** This report is a pre-screening aid only. "
            f"It does not constitute a final eligibility decision, diagnosis, or clinical recommendation."
        )

    # ── Executive summary ─────────────────────────────────────────────────────

    @staticmethod
    def _executive_summary(ranked_trials: List[TrialRanking]) -> str:
        if not ranked_trials:
            return "## Executive Summary\n\nNo trials passed initial filtering for this patient."

        sup_count = sum(1 for r in ranked_trials if r.clinical_fit == CriterionStatus.SUPPORTED)
        nsup_count = sum(1 for r in ranked_trials if r.clinical_fit == CriterionStatus.NOT_SUPPORTED)
        hrv_count  = sum(1 for r in ranked_trials if r.requires_human_review)

        lines = [
            "## Executive Summary",
            "",
            f"Showing **{len(ranked_trials)}** top-ranked trial(s) (capped at 3).\n",
            f"| Clinical Fit   | Count |",
            f"|:---------------|------:|",
            f"| ✅ Supported    | {sup_count} |",
            f"| ❌ Not Supported| {nsup_count} |",
            f"| ❓ Other        | {len(ranked_trials) - sup_count - nsup_count} |",
            f"| 🔍 Human Review Required | {hrv_count} |",
        ]
        return "\n".join(lines)

    # ── Per-trial detail ──────────────────────────────────────────────────────

    @staticmethod
    def _trial_details(
        ranked_trials: List[TrialRanking],
        evaluations: Dict[str, List[CriterionEvaluation]],
    ) -> str:
        if not ranked_trials:
            return ""

        lines = ["## Trial Results\n"]
        for rank_idx, ranking in enumerate(ranked_trials, start=1):
            # ── Trial header ──────────────────────────────────────────────
            lines.append(f"---\n\n### {rank_idx}. {ranking.title}")
            lines.append(f"**Trial ID:** `{ranking.trial_id}`")
            lines.append(f"**Reason surfaced:** {ranking.reason_surfaced}\n")

            # ── Clinical fit vs recruiting status (kept separate) ─────────
            fit_badge = _FIT_BADGE.get(ranking.clinical_fit, ranking.clinical_fit.value)
            rec_badge = "🟢 Actively recruiting" if ranking.is_recruiting else "🔴 Not recruiting"
            lines.append(f"| Field              | Status |")
            lines.append(f"|:-------------------|:-------|")
            lines.append(f"| **Clinical Fit**   | {fit_badge} |")
            lines.append(f"| **Recruiting**     | {rec_badge} |")
            lines.append(f"| **Score**          | `{ranking.score:.3f}` |")
            lines.append(f"| **Human Review**   | {'⚠️  Required' if ranking.requires_human_review else '✅ Not required'} |\n")

            # ── Criterion table ───────────────────────────────────────────
            trial_evals = evaluations.get(ranking.trial_id, [])
            if trial_evals:
                # Detect whether any criteria failed LLM evaluation so we can
                # show a single deduplicated notice instead of repeating it per row.
                _llm_error_msg = "LLM evaluation unavailable — criterion requires manual clinical review."
                llm_unavailable_ids = [
                    ev.criterion_id for ev in trial_evals
                    if ev.evaluator_type == "llm_engine_error"
                ]

                lines.append("#### Criterion Evaluation\n")
                lines.append("| Criterion | Status | Evaluator | Reasoning | Evidence |")
                lines.append("|:----------|:------:|:---------:|:----------|:---------|")
                for ev in trial_evals:
                    icon = _STATUS_ICON.get(ev.status, "?")
                    # Clean reasoning: strip pipe chars that break Markdown tables,
                    # cap length so cells don't overflow the page.
                    reasoning = ev.reasoning.replace("|", "/")
                    if ev.evaluator_type == "llm_engine_error":
                        reasoning = "LLM service unavailable — manual review required."
                    elif len(reasoning) > 200:
                        reasoning = reasoning[:197] + "..."
                    # Build human-readable evidence refs (no raw UUIDs)
                    ev_refs = _format_evidence(ev.evidence_used)
                    lines.append(
                        f"| `{ev.criterion_id}` | {icon} {ev.status.value} "
                        f"| {ev.evaluator_type} | {reasoning} | {ev_refs} |"
                    )

                # Single deduplicated LLM-unavailable notice
                if llm_unavailable_ids:
                    lines.append(
                        "\n> **Note:** Some criteria requiring LLM evaluation "
                        f"({', '.join(f'`{c}`' for c in llm_unavailable_ids)}) "
                        "could not be automatically assessed because the LLM service was unavailable. "
                        "Please review these criteria manually."
                    )

                # ── Unanswered questions ──────────────────────────────────
                all_questions: List[str] = []
                for ev in trial_evals:
                    for q in ev.unanswered_questions:
                        if q not in all_questions:
                            all_questions.append(q)

                if all_questions:
                    lines.append("\n#### Unanswered Questions / Items for Clinical Review\n")
                    for q in all_questions:
                        lines.append(f"- {q}")
                else:
                    lines.append("\n_No unanswered questions — all criteria resolved automatically._")
            else:
                lines.append("_No criterion evaluations available._")

            lines.append("")  # spacer

        return "\n".join(lines)

    # ── Filtered-out appendix ─────────────────────────────────────────────────

    @staticmethod
    def _filtered_appendix(filter_reasons: Dict[str, str]) -> str:
        if not filter_reasons:
            return ""
        lines = [
            "## Appendix — Trials Excluded Before Evaluation\n",
            "These trials were removed by deterministic filters:\n",
            "| Trial ID | Reason |",
            "|:---------|:-------|",
        ]
        for tid, reason in filter_reasons.items():
            lines.append(f"| `{tid}` | {reason} |")
        return "\n".join(lines)

    # ── Footer ────────────────────────────────────────────────────────────────

    @staticmethod
    def _footer() -> str:
        return (
            "---\n\n"
            "_Pre-screening report generated by the Clinical Trial Pre-Screening Assistant. "
            "This report is intended to support — not replace — clinical judgement. "
            "All recommendations must be verified by a qualified healthcare professional "
            "before any action is taken._"
        )

    # ── Structured data dict ──────────────────────────────────────────────────

    @staticmethod
    def _build_data(
        patient: Patient,
        ranked_trials: List[TrialRanking],
        evaluations: Dict[str, List[CriterionEvaluation]],
        filter_reasons: Optional[Dict[str, str]],
        now: datetime,
    ) -> Dict:
        return {
            "generated_at":   now.isoformat(),
            "patient_id":     patient.id,
            "patient_age":    patient.age,
            "patient_gender": patient.gender,
            "summary": {
                "total_ranked":          len(ranked_trials),
                "supported_count":       sum(1 for r in ranked_trials
                                             if r.clinical_fit == CriterionStatus.SUPPORTED),
                "not_supported_count":   sum(1 for r in ranked_trials
                                             if r.clinical_fit == CriterionStatus.NOT_SUPPORTED),
                "human_review_required": sum(1 for r in ranked_trials if r.requires_human_review),
            },
            "trials": [
                {
                    **r.model_dump(),
                    "criteria": [e.model_dump() for e in evaluations.get(r.trial_id, [])],
                }
                for r in ranked_trials
            ],
            "filtered_out": filter_reasons or {},
        }
