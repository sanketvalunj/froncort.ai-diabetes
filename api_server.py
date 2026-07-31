"""
FastAPI wrapper around the existing LangGraph clinical-trial pipeline.

Endpoints
---------
GET  /patients
    Returns the list of all patients from the dataset (id + demographics + labs).

POST /screen
    Body: {"patient_id": "P-1842"}
    Runs the full LangGraph pipeline and returns the structured result.

GET  /report/{patient_id}/markdown
    Returns the most recent Markdown report for the given patient.

GET  /report/{patient_id}/pdf
    Streams the most recent PDF report for the given patient.

GET  /health
    Returns {"status": "ok"} — used by Render health checks.

This file contains NO business logic — every heavy-lifting call goes straight
into the existing services and workflow already implemented in app/.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from config.settings import settings
from app.llm.client import LLMClient
from app.models.state import AgentState
from app.graph.workflow import run_workflow
from app.retrieval.loader import load_dataset
from app.retrieval.parser import parse_patient, parse_trials

# ── CORS origins ──────────────────────────────────────────────────────────────
# In production set ALLOWED_ORIGINS to your Vercel URL, e.g.:
#   ALLOWED_ORIGINS=https://clinical-trial-ui.vercel.app
# Leave unset (or set to "*") for development / open access.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins: List[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins != "*"
    else ["*"]
)

app = FastAPI(
    title="Clinical Trial Pre-Screening API",
    description="Thin REST wrapper around the LangGraph pre-screening pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # permit all Vercel preview URLs
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ── Module-level singletons ───────────────────────────────────────────────────
# Load the dataset and instantiate the LLM client once at startup so the
# heavy sentence-transformer model and FAISS index are not reloaded per request.

_llm_client: Optional[LLMClient] = None
_raw_patients: list = []
_trials_raw: list = []


def _ensure_loaded():
    global _llm_client, _raw_patients, _trials_raw
    if _llm_client is not None:
        return
    _llm_client = LLMClient(settings.llm)
    raw = load_dataset(settings.paths.data)
    dataset = raw if isinstance(raw, dict) else {}
    _raw_patients = dataset.get("patients", raw if isinstance(raw, list) else [])
    trials_root = dataset.get("trials", [])
    _trials_raw = trials_root if isinstance(trials_root, list) else []


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    patient_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_raw_patient(patient_id: str) -> dict:
    """Return the raw patient dict from the dataset or raise 404."""
    for p in _raw_patients:
        pid = p.get("patient_id") or p.get("id") or ""
        if str(pid) == patient_id:
            return p
    raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")


def _latest_report(patient_id: str, extension: str) -> Path:
    """Return the most-recently written report file for a patient, or raise 404."""
    reports_dir = Path(settings.paths.reports)
    pdf_dir = reports_dir.parent / "report_pdfs"

    if extension == ".md":
        search_dir = reports_dir
    else:
        search_dir = pdf_dir

    candidates = sorted(
        search_dir.glob(f"{patient_id}_*{extension}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No {extension} report found for patient '{patient_id}'. "
                   "Run /screen first.",
        )
    return candidates[0]


def _patient_summary(raw: dict) -> dict:
    """
    Convert a raw dataset patient into a JSON-serialisable summary dict.
    Reuses parse_patient() so no duplication of parsing logic.
    """
    p = parse_patient(raw)
    demo = raw.get("demographics", {})
    return {
        "patient_id":   p.id,
        "name":         demo.get("name", ""),
        "age":          p.age,
        "gender":       p.gender,
        "conditions":   p.conditions,
        "medications":  p.medications,
        "lab_results":  [
            {
                "test":      lr.test,
                "value":     lr.value,
                "unit":      lr.unit,
                "date":      str(lr.date) if lr.date else None,
                "source_id": lr.source_id,
            }
            for lr in p.lab_results
        ],
        "as_of_date":   raw.get("as_of_date", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    _ensure_loaded()


@app.get("/health", summary="Health check for Render")
def health_check():
    return JSONResponse({"status": "ok"})


@app.get("/patients", summary="List all patients from the dataset")
def list_patients():
    _ensure_loaded()
    return [_patient_summary(p) for p in _raw_patients]


@app.post("/screen", summary="Run the LangGraph pre-screening pipeline")
def screen_patient(body: ScreenRequest):
    _ensure_loaded()

    # 1. Parse patient from the real dataset
    raw_patient = _find_raw_patient(body.patient_id)
    patient = parse_patient(raw_patient)

    # 2. Parse all trials
    trials = parse_trials(_trials_raw)
    if not trials:
        raise HTTPException(status_code=500, detail="Trial dataset is empty or failed to parse.")

    # 3. Run the full LangGraph pipeline — no business logic here, just wiring
    initial_state = AgentState(patient=patient, all_trials=trials)
    try:
        final_state = run_workflow(settings, _llm_client, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # 4. Serialise AgentState → JSON-friendly dict
    #    Use model_dump() on the Pydantic models so enums, dates, etc. are
    #    converted to plain Python types automatically.
    result = {
        "patient":         patient.model_dump(mode="json"),
        "ranked_trials":   [r.model_dump(mode="json") for r in final_state.ranked_trials],
        "evaluations": {
            tid: [e.model_dump(mode="json") for e in evals]
            for tid, evals in final_state.evaluations.items()
        },
        "filter_reasons":  final_state.filter_reasons,
        "report_markdown": final_state.report_markdown,
        "report_data":     final_state.report_data,
        "trace_id":        final_state.trace_id,
        "run_timestamp":   final_state.run_timestamp.isoformat(),
    }
    return result


@app.get(
    "/report/{patient_id}/markdown",
    response_class=PlainTextResponse,
    summary="Return the latest Markdown report for a patient",
)
def get_report_markdown(patient_id: str):
    path = _latest_report(patient_id, ".md")
    return path.read_text(encoding="utf-8")


@app.get(
    "/report/{patient_id}/pdf",
    summary="Stream the latest PDF report for a patient",
)
def get_report_pdf(patient_id: str):
    path = _latest_report(patient_id, ".pdf")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=f"{patient_id}_report.pdf",
    )


# ── Entrypoint (used by Render: python api_server.py) ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
