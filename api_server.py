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

Startup strategy
----------------
Nothing heavy is loaded at process start.  The FastAPI startup event is
intentionally empty so Render's health check passes immediately with a tiny
RSS footprint (~60 MB).

Heavy resources are initialised lazily on the first real request:
  - LLMClient          → on first /screen  (~negligible, API client only)
  - Dataset JSON        → on first /patients or /screen  (~5 MB)
  - SentenceTransformer → on first /screen retrieval step  (~200 MB)
  - FAISS index         → on first /screen retrieval step  (~1 MB)
  - Compiled LangGraph  → on first /screen  (~negligible)

Each step is timed and printed so future startup bottlenecks are obvious in
the Render log stream.
"""

import os
import time
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
# All module-level variables start as None / empty.  Nothing is loaded until
# _ensure_loaded() is called by the first real request.  The startup event
# does NOT call _ensure_loaded() so the process stays lightweight at boot.

_llm_client: Optional[LLMClient] = None
_raw_patients: list = []
_trials_raw: list = []
_loaded: bool = False          # guard so we only run init once


def _timed(label: str):
    """Context manager that prints a START / DONE line with elapsed seconds."""
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
    Initialise all lightweight singletons on first call; no-op thereafter.

    Heavy resources (SentenceTransformer, FAISS index) are NOT loaded here —
    they are loaded lazily inside RetrievalService on the first /screen call.
    This function only handles the truly cheap operations:
      - constructing LLMClient (no network call, no model weights)
      - reading and parsing the ~5 MB JSON dataset
    """
    global _llm_client, _raw_patients, _trials_raw, _loaded
    if _loaded:
        return

    print("[startup] First request — initialising lightweight singletons", flush=True)
    t_total = time.perf_counter()

    with _timed("Constructing LLMClient"):
        _llm_client = LLMClient(settings.llm)
        # Note: the underlying LangChain client is NOT created here.
        # It is created lazily inside LLMClient.client on first generate() call.

    with _timed("Loading dataset JSON"):
        raw = load_dataset(settings.paths.data)
        dataset = raw if isinstance(raw, dict) else {}
        _raw_patients = dataset.get("patients", raw if isinstance(raw, list) else [])
        trials_root = dataset.get("trials", [])
        _trials_raw = trials_root if isinstance(trials_root, list) else []

    elapsed_total = time.perf_counter() - t_total
    print(
        f"[startup] Lightweight init complete ({elapsed_total:.2f}s) — "
        f"{len(_raw_patients)} patients, {len(_trials_raw)} trials loaded.",
        flush=True,
    )
    print(
        "[startup] NOTE: SentenceTransformer + FAISS index will load on the "
        "first /screen request (lazy). That request will be slower than subsequent ones.",
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
    search_dir = reports_dir if extension == ".md" else pdf_dir
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
    """Convert a raw dataset patient into a JSON-serialisable summary dict."""
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
# Startup event
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Intentionally lightweight — do NOT load models or datasets here.

    Render hits /health immediately after the process starts.  If we block
    the event loop loading a 200 MB model, the health check times out and
    Render kills the instance before it ever serves a request.

    All heavy resources are deferred to the first real API call via
    _ensure_loaded().
    """
    print("[startup] FastAPI startup complete — process is ready (lazy mode).", flush=True)
    print(f"[startup] LLM provider: {settings.llm.provider}, model: {settings.llm.model}", flush=True)
    print(f"[startup] Embedding model: {settings.embeddings.model}", flush=True)
    print(f"[startup] Data path: {settings.paths.data}", flush=True)
    print(f"[startup] Vector store: {settings.paths.vector_store}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check for Render")
def health_check():
    """Returns immediately without touching any heavy resource."""
    return JSONResponse({"status": "ok"})


@app.get("/patients", summary="List all patients from the dataset")
def list_patients():
    _ensure_loaded()
    return [_patient_summary(p) for p in _raw_patients]


@app.post("/screen", summary="Run the LangGraph pre-screening pipeline")
def screen_patient(body: ScreenRequest):
    _ensure_loaded()

    t0 = time.perf_counter()

    # 1. Parse patient
    raw_patient = _find_raw_patient(body.patient_id)
    patient = parse_patient(raw_patient)

    # 2. Parse all trials
    with _timed(f"Parsing trials for patient {body.patient_id}"):
        trials = parse_trials(_trials_raw)
    if not trials:
        raise HTTPException(status_code=500, detail="Trial dataset is empty or failed to parse.")

    # 3. Run the full LangGraph pipeline
    # Note: on the FIRST request this also triggers:
    #   - LangChain client construction (LLMClient.client property)
    #   - SentenceTransformer load (~200 MB, ~5–15 s on Render free tier)
    #   - FAISS index load from disk (~1 MB, fast)
    #   - LangGraph workflow compilation (once only, then cached)
    initial_state = AgentState(patient=patient, all_trials=trials)
    try:
        print(f"[screen] Running pipeline for patient {body.patient_id}", flush=True)
        final_state = run_workflow(settings, _llm_client, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    elapsed = time.perf_counter() - t0
    print(f"[screen] Pipeline complete for {body.patient_id} ({elapsed:.2f}s)", flush=True)

    # 4. Serialise AgentState → JSON-friendly dict
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


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
