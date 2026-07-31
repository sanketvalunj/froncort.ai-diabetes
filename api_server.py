"""
FastAPI wrapper around the LangGraph clinical-trial pipeline.

Startup strategy — zero heavy work at import time
-------------------------------------------------
When uvicorn does `import api_server`, Python executes every top-level
statement in this file.  We keep those statements to an absolute minimum:

  ALLOWED at module level   : FastAPI app creation, CORS middleware, Pydantic
                              request/response models, route *decorators*.
  NOT ALLOWED at module level: anything that imports sentence_transformers,
                              faiss, langgraph, langchain, or reads files.

All heavy resources are initialised inside _ensure_loaded(), which is called
by the first real API request — never during import.

Memory budget (Render free tier: 512 MB)
-----------------------------------------
  Import time           : ~60 MB  (FastAPI + Pydantic + structlog only)
  After first /screen   : ~350 MB (+ LangChain client + SentenceTransformer
                                     + FAISS index + compiled LangGraph)
  Subsequent requests   : no additional allocation (everything cached)
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# ── CORS ──────────────────────────────────────────────────────────────────────
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
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
# All None until _ensure_loaded() is called by the first real request.
_llm_client = None
_raw_patients: list = []
_trials_raw: list = []
_settings = None
_loaded: bool = False


def _get_settings():
    """Return the settings singleton, importing config only when first needed."""
    global _settings
    if _settings is None:
        from config.settings import settings as _s
        _settings = _s
    return _settings


def _timed(label: str):
    """Simple context manager that prints elapsed time for each init step."""
    class _Timer:
        def __enter__(self):
            print(f"[startup] {label}...", flush=True)
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._t0
            print(f"[startup] {label} done ({elapsed:.2f}s)", flush=True)

    return _Timer()


def _ensure_loaded() -> None:
    """
    Initialise all singletons on first call; no-op thereafter.

    This function only does the cheap work:
      - import config.settings (pydantic-settings, no model weights)
      - construct LLMClient (just a Python object, no network call)
      - read + parse the JSON dataset (~5 MB)

    The truly heavy objects (SentenceTransformer, FAISS index, LangGraph
    compiled graph) are initialised lazily inside the service layer on the
    first /screen request.
    """
    global _llm_client, _raw_patients, _trials_raw, _loaded

    if _loaded:
        return

    print("[startup] First request — initialising singletons", flush=True)
    t_total = time.perf_counter()

    with _timed("Loading config"):
        from config.settings import settings
        _settings_ref = settings  # local alias used below

    with _timed("Constructing LLMClient"):
        from app.llm.client import LLMClient
        _llm_client = LLMClient(_settings_ref.llm)

    with _timed("Loading dataset JSON"):
        from app.retrieval.loader import load_dataset
        raw = load_dataset(_settings_ref.paths.data)
        dataset = raw if isinstance(raw, dict) else {}
        _raw_patients = dataset.get("patients", raw if isinstance(raw, list) else [])
        trials_root = dataset.get("trials", [])
        _trials_raw = trials_root if isinstance(trials_root, list) else []

    # Store settings for use by route handlers
    global _settings
    _settings = _settings_ref

    elapsed_total = time.perf_counter() - t_total
    print(
        f"[startup] Init complete ({elapsed_total:.2f}s) — "
        f"{len(_raw_patients)} patients, {len(_trials_raw)} trials.",
        flush=True,
    )
    print(
        "[startup] NOTE: SentenceTransformer + FAISS + LangGraph will load "
        "on the first /screen request (lazy).",
        flush=True,
    )

    _loaded = True


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    patient_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_raw_patient(patient_id: str) -> dict:
    for p in _raw_patients:
        pid = p.get("patient_id") or p.get("id") or ""
        if str(pid) == patient_id:
            return p
    raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")


def _latest_report(patient_id: str, extension: str) -> Path:
    s = _get_settings()
    reports_dir = Path(s.paths.reports)
    search_dir = reports_dir if extension == ".md" else reports_dir.parent / "report_pdfs"
    candidates = sorted(
        search_dir.glob(f"{patient_id}_*{extension}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No {extension} report found for patient '{patient_id}'. Run /screen first.",
        )
    return candidates[0]


def _patient_summary(raw: dict) -> dict:
    from app.retrieval.parser import parse_patient
    p = parse_patient(raw)
    demo = raw.get("demographics", {})
    return {
        "patient_id":  p.id,
        "name":        demo.get("name", ""),
        "age":         p.age,
        "gender":      p.gender,
        "conditions":  p.conditions,
        "medications": p.medications,
        "lab_results": [
            {
                "test":      lr.test,
                "value":     lr.value,
                "unit":      lr.unit,
                "date":      str(lr.date) if lr.date else None,
                "source_id": lr.source_id,
            }
            for lr in p.lab_results
        ],
        "as_of_date": raw.get("as_of_date", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Startup event — intentionally empty
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Do NOT load any models, read any files, or import any heavy packages here.

    Render hits /health immediately after the process starts.  Any blocking
    work here delays the port bind and causes Render to report
    "No open ports detected".  All heavy work is deferred to first request.
    """
    print("[startup] FastAPI startup complete — ready to accept connections.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check for Render")
@app.get("/", include_in_schema=False)
def health_check():
    """Returns instantly — no heavy resources touched.
    Handles both / and /health so Render's default health probe succeeds."""
    return JSONResponse({"status": "ok"})


@app.get("/patients", summary="List all patients from the dataset")
def list_patients():
    _ensure_loaded()
    return [_patient_summary(p) for p in _raw_patients]


@app.post("/screen", summary="Run the LangGraph pre-screening pipeline")
def screen_patient(body: ScreenRequest):
    _ensure_loaded()

    t0 = time.perf_counter()
    s = _get_settings()

    # 1. Parse patient
    from app.retrieval.parser import parse_patient, parse_trials
    raw_patient = _find_raw_patient(body.patient_id)
    patient = parse_patient(raw_patient)

    # 2. Parse all trials
    with _timed(f"Parsing trials for {body.patient_id}"):
        trials = parse_trials(_trials_raw)
    if not trials:
        raise HTTPException(status_code=500, detail="Trial dataset is empty or failed to parse.")

    # 3. Build initial state
    from app.models.state import AgentState
    initial_state = AgentState(patient=patient, all_trials=trials)

    # 4. Run LangGraph pipeline
    # On the FIRST call this triggers (once only, then cached):
    #   - LangChain client construction
    #   - SentenceTransformer load (~200 MB, ~5–15 s on Render free tier)
    #   - FAISS index read from disk (~1 MB)
    #   - LangGraph StateGraph compilation
    from app.graph.workflow import run_workflow
    try:
        print(f"[screen] Running pipeline for patient {body.patient_id}", flush=True)
        final_state = run_workflow(s, _llm_client, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    elapsed = time.perf_counter() - t0
    print(f"[screen] Pipeline complete for {body.patient_id} ({elapsed:.2f}s)", flush=True)

    return {
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


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
