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
                lines.append("#### Criterion Evaluation\n")
                lines.append("| Criterion ID | Status | Evaluator | Reasoning | Evidence Used |")
                lines.append("|:-------------|:------:|:---------:|:----------|:--------------|")
                for ev in trial_evals:
                    icon      = _STATUS_ICON.get(ev.status, "?")
                    reasoning = ev.reasoning.replace("|", "\\|")
                    # Collect evidence source labels, IDs, and dates
                    ev_refs = "; ".join(
                        f"`{e.source}`"
                        + (f" [{e.evidence_id}]" if e.evidence_id else "")
                        + (f" ({e.date})" if e.date else "")
                        for e in ev.evidence_used
                    ) or "—"
                    lines.append(
                        f"| `{ev.criterion_id}` | {icon} {ev.status.value} "
                        f"| {ev.evaluator_type} | {reasoning} | {ev_refs} |"
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
