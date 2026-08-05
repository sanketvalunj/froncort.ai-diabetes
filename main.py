# CLI entrypoint for running the clinical trial agent.
"""
Clinical Trial Pre-Screening Assistant — CLI entrypoint.

Commands:
    run    Evaluate a patient against the trial dataset.
    index  Build (or rebuild) the FAISS vector index.
    info   Print current configuration.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from config.settings import settings as _default_settings
from app.llm.client import LLMClient
from app.models.patient import Patient
from app.models.state import AgentState
from app.graph.workflow import run_workflow
from app.retrieval.loader import load_dataset
from app.retrieval.parser import parse_patient, parse_trials
from app.utils.helpers import format_timestamp
from app.utils.logger import get_logger
from metrics.metrics import PipelineMetrics

app = typer.Typer(name="clinical-trial-agent",
                  help="Agentic Clinical Trial Pre-Screening Assistant",
                  add_completion=False)
log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_patient(patient_file, patient_json) -> Patient:
    if patient_file and patient_json:
        typer.echo("Error: provide either --patient-file or --patient-json, not both.", err=True)
        raise typer.Exit(1)
    if patient_file:
        if not patient_file.exists():
            typer.echo(f"Error: patient file not found: {patient_file}", err=True)
            raise typer.Exit(1)
        raw = json.loads(patient_file.read_text(encoding="utf-8"))
    elif patient_json:
        try:
            raw = json.loads(patient_json)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: invalid JSON: {exc}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("Error: one of --patient-file or --patient-json is required.", err=True)
        raise typer.Exit(1)
    return parse_patient(raw)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    patient_file: Optional[Path] = typer.Option(None, "--patient-file", "-p",
                                                help="Path to patient JSON file."),
    patient_json: Optional[str]  = typer.Option(None, "--patient-json", "-j",
                                                help="Patient data as inline JSON string."),
    data_path:    Path           = typer.Option(_default_settings.paths.data,
                                                "--data-path", "-d",
                                                help="Path to trials JSON dataset."),
    output_dir:   Optional[Path] = typer.Option(None, "--output-dir", "-o",
                                                help="Report output directory."),
    no_metrics:   bool           = typer.Option(False, "--no-metrics",
                                                help="Skip metrics computation."),
    verbose:      bool           = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a patient against the clinical trial dataset."""
    effective_settings = _default_settings
    if output_dir:
        effective_settings = effective_settings.model_copy(
            update={"paths": effective_settings.paths.model_copy(
                update={"reports": output_dir})})

    typer.echo("Loading patient data...")
    patient = _load_patient(patient_file, patient_json)
    if verbose:
        typer.echo(f"  Patient: {patient.id}, age {patient.age}, {patient.gender}")

    typer.echo(f"Loading trials from {data_path}...")
    try:
        raw = load_dataset(data_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}\nTip: run `python main.py index` first.", err=True)
        raise typer.Exit(1)
    trials_raw = raw if isinstance(raw, list) else raw.get("trials", raw)
    trials     = parse_trials(trials_raw)
    typer.echo(f"  Loaded {len(trials)} trials.")

    initial_state = AgentState(patient=patient, all_trials=trials)
    llm_client    = LLMClient(effective_settings.llm)

    typer.echo("Running pre-screening pipeline...")
    t0 = time.perf_counter()
    try:
        final_state = run_workflow(effective_settings, llm_client, initial_state)
    except Exception as exc:
        log.error("pipeline_error", error=str(exc), trace_id=initial_state.trace_id)
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not no_metrics:
        try:
            pm      = PipelineMetrics(effective_settings)
            metrics = pm.compute(final_state)
            metrics["run_duration_ms"] = elapsed_ms
            pm.save(metrics, patient.id, final_state.trace_id)
        except Exception as exc:
            log.warning("metrics_error", error=str(exc))

    typer.echo(f"\nCompleted in {elapsed_ms:.0f} ms — {len(final_state.ranked_trials)} trial(s) ranked.")
    for idx, r in enumerate(final_state.ranked_trials, start=1):
        typer.echo(f"  {idx}. [{r.clinical_fit.value}] {r.title} (score: {r.score:.3f})")


@app.command()
def index(
    data_path:   Path = typer.Option(_default_settings.paths.data,
                                     "--data-path", "-d"),
    output_path: Path = typer.Option(_default_settings.paths.vector_store,
                                     "--output-path", "-o"),
) -> None:
    """Build (or rebuild) the FAISS vector index from the trial dataset."""
    from scripts.build_index import app as build_app
    from typer.testing import CliRunner
    result = CliRunner().invoke(build_app, ["--data-path", str(data_path),
                                            "--output-path", str(output_path)])
    typer.echo(result.output)
    if result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command()
def info() -> None:
    """Print the resolved configuration."""
    typer.echo(json.dumps(_default_settings.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    app()
