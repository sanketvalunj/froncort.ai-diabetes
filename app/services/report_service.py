# Generates markdown and PDF reports for trial screening results.
from datetime import datetime, timezone
from pathlib import Path

from app.models.state import AgentState
from app.reports.report_generator import ReportGenerator
from app.reports.pdf_generator import md_file_to_pdf
from app.utils.helpers import format_timestamp
from app.utils.logger import get_logger

log = get_logger(__name__)


class ReportService:
    def __init__(self, settings):
        self._settings  = settings
        self._generator = ReportGenerator()

    def run(self, state: AgentState) -> AgentState:
        log.info("report_start", patient_id=state.patient.id,
                 ranked_trials=len(state.ranked_trials), trace_id=state.trace_id)
        report_markdown, report_data = self._generator.generate(
            patient=state.patient,
            ranked_trials=state.ranked_trials,
            evaluations=state.evaluations,
            filter_reasons=state.filter_reasons or None,
        )
        reports_dir = Path(self._settings.paths.reports)
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp   = format_timestamp(datetime.now(timezone.utc))
        report_path = reports_dir / f"{state.patient.id}_{timestamp}.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        log.info("report_written", path=str(report_path), trace_id=state.trace_id)

        # ── PDF artifact (additive — does not affect state or Markdown output) ─
        try:
            pdf_dir  = reports_dir.parent / "report_pdfs"
            pdf_path = md_file_to_pdf(report_path, pdf_dir)
            log.info("pdf_written", path=str(pdf_path), trace_id=state.trace_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("pdf_generation_failed", error=str(exc),
                        trace_id=state.trace_id)

        return state.model_copy(update={"report_markdown": report_markdown,
                                        "report_data": report_data})
