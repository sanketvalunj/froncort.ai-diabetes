# Clinical Trial Pre-Screening Assistant

An agentic pipeline that evaluates a patient record against a set of clinical trials and produces a coordinator-facing eligibility report. Built with LangGraph, Pydantic v2, FAISS, and LangChain.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Design Decisions](#design-decisions)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Running Tests](#running-tests)

---

## Overview

Given a patient record and a dataset of clinical trials, the pipeline:

1. **Filters** trials to those currently recruiting and broadly age-appropriate.
2. **Retrieves** relevant evidence from the trial criteria using semantic search (FAISS + sentence-transformers).
3. **Evaluates** each criterion using a hybrid strategy: deterministic rules for numeric criteria (age, HbA1c, eGFR) and an LLM for semantic criteria (medications, conditions).
4. **Ranks** trials by clinical fit score and caps output at three candidates.
5. **Generates** a structured Markdown report listing per-criterion status, evidence sources, unanswered questions, and a human-review flag.

---

## Architecture

```
Patient JSON
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│ filter_node │────▶│retrieval_node│────▶│evaluation_node│────▶│ ranking_node │────▶│ report_node │
└─────────────┘     └──────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
      │                    │                     │                     │                    │
 FilteringService    RetrievalService      EvaluationService     RankingService       ReportService
 • status filter     • FAISS vector        • RuleEngine          • score per          • Markdown
 • age pre-check       search              • LLMEvaluator          criterion            report
                     • patient evidence    • EvaluatorRouter     • cap at TOP_K=3     • JSON data
                       extraction          • hybrid dispatch                           • file write
```

The graph is compiled with **LangGraph** (`StateGraph`) and runs as a single `invoke()` call. Each node receives and returns the full `AgentState`, keeping the pipeline stateless and easy to test in isolation.

### Evaluator routing

| Criterion type | Evaluator      | Rationale |
|:---------------|:---------------|:----------|
| AGE            | Rule engine    | Numeric range / bound — deterministic |
| HBA1C          | Rule engine    | Numeric threshold — deterministic |
| EGFR           | Rule engine    | Numeric threshold — deterministic |
| RECRUITING     | Rule engine    | Boolean status already known at filter time |
| MEDICATION     | LLM            | Requires semantic matching of drug names / classes |
| CONDITION      | LLM            | Requires clinical concept understanding |
| OTHER          | Clinical review flag | Cannot be automated without domain knowledge |

### Criterion status vocabulary

| Status | Meaning |
|:-------|:--------|
| `SUPPORTED` | Evidence supports the patient meeting this criterion |
| `NOT_SUPPORTED` | Evidence shows the patient does not meet this criterion |
| `UNKNOWN` | Required data is absent from the patient record |
| `CONFLICTING_EVIDENCE` | Evidence points in both directions |
| `REQUIRES_CLINICAL_REVIEW` | Cannot be resolved automatically |

`UNKNOWN` is never treated as a negative — it raises an explicit question for the coordinator instead.

---

## Project Structure

```
clinical-trial-agent/
├── main.py                        # CLI entrypoint (run / index / info)
├── requirements.txt
├── .env.example
│
├── config/
│   ├── settings.py                # Pydantic-settings config (LLM, embeddings, paths)
│   └── prompts.py                 # LLM prompt templates
│
├── app/
│   ├── graph/
│   │   ├── workflow.py            # LangGraph graph definition + run_workflow()
│   │   └── nodes.py               # Node factory wiring services into graph callables
│   │
│   ├── services/                  # One service per pipeline stage
│   │   ├── filtering_service.py
│   │   ├── retrieval_service.py
│   │   ├── evaluation_service.py
│   │   ├── ranking_service.py
│   │   └── report_service.py
│   │
│   ├── evaluation/
│   │   ├── rule_engine.py         # Deterministic evaluator (AGE, HBA1C, EGFR, RECRUITING)
│   │   ├── llm_engine.py          # LLM evaluator (MEDICATION, CONDITION)
│   │   └── router.py              # Dispatches criteria to the right evaluator
│   │
│   ├── retrieval/
│   │   ├── parser.py              # Parses both real-dataset and generic JSON formats
│   │   ├── loader.py              # Loads the dataset JSON from disk
│   │   ├── embeddings.py          # Wraps sentence-transformers
│   │   ├── vectorstore.py         # FAISS index (build / save / load / search)
│   │   └── retriever.py           # Builds patient queries + retrieves trial evidence
│   │
│   ├── reports/
│   │   └── report_generator.py    # Produces the coordinator-facing Markdown report
│   │
│   ├── models/                    # Pydantic v2 domain models
│   │   ├── patient.py             # Patient, LabResult
│   │   ├── trial.py               # Trial, Criterion, CriterionType
│   │   ├── evaluation.py          # CriterionEvaluation, TrialRanking, Evidence
│   │   └── state.py               # AgentState (the LangGraph state object)
│   │
│   └── utils/
│       ├── helpers.py
│       ├── logger.py              # structlog-based structured logging
│       └── constants.py
│
├── metrics/
│   └── metrics.py                 # PipelineMetrics — compute + save pipeline stats
│
├── scripts/
│   └── build_index.py             # Standalone CLI to build the FAISS vector index
│
├── data/
│   ├── Type2-Diabetes-Trial-Agent-Dataset.json   # 15 patients, 36 trials
│   ├── example_patient.json                      # Quick-start demo patient
│   └── vector_store/              # FAISS index written here after `index` command
│
├── artifacts/
│   ├── reports/                   # Generated Markdown reports
│   ├── metrics/                   # Pipeline metric JSON files
│   └── logs/
│
└── tests/
    ├── conftest.py                # Shared fixtures and helper factories
    ├── test_retrieval.py          # Parser, criterion-type detection, FAISS store
    ├── test_evaluation.py         # RuleEngine, LLMEvaluator, EvaluatorRouter
    ├── test_graph.py              # Services layer + LangGraph end-to-end
    ├── test_report.py             # ReportGenerator — all coordinator-facing fields
    └── test_metrics.py            # PipelineMetrics — compute + save
```

---

## Setup

### Prerequisites

- Python 3.11+ (tested on 3.14.6)
- An OpenAI or Anthropic API key for LLM-based criterion evaluation

### Install

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your API key:
#   OPENAI_API_KEY=sk-...
#   LLM__MODEL=gpt-4o-mini          # or any other supported model
```

All settings can also be overridden via environment variables using the `LLM__`, `EMBEDDINGS__`, `RETRIEVAL__`, and `PATHS__` prefixes (see `config/settings.py`).

### Build the vector index

The retrieval stage requires a FAISS index built from the trial dataset. Run this once before evaluating patients:

```bash
python main.py index
```

The index is written to `data/vector_store/` by default.

---

## Usage

### Evaluate a patient

Using the provided example patient:

```bash
python main.py run --patient-file data/example_patient.json
```

Using inline JSON:

```bash
python main.py run --patient-json '{"id":"p1","age":58,"gender":"Female","conditions":["Type 2 Diabetes"],"medications":["metformin"],"lab_results":[{"test":"HbA1c","value":7.9,"unit":"%"}]}'
```

Additional options:

```
--data-path PATH      Path to the trials dataset (default: data/Type2-Diabetes-Trial-Agent-Dataset.json)
--output-dir PATH     Directory for the generated report (default: artifacts/reports/)
--no-metrics          Skip writing pipeline metrics to disk
--verbose, -v         Print patient summary and per-trial details to stdout
```

The command prints a ranked list of up to three trials and writes a Markdown report to `artifacts/reports/`.

### Print resolved configuration

```bash
python main.py info
```

### Rebuild the index

```bash
python main.py index --data-path data/Type2-Diabetes-Trial-Agent-Dataset.json \
                     --output-path data/vector_store
```

---

## Design Decisions

### Hybrid evaluation strategy

Pure rule-based evaluation handles numeric criteria (age, HbA1c, eGFR) with full determinism and no latency cost. The LLM is only invoked for criteria that require semantic understanding of drug names, disease concepts, or free-text conditions. This keeps the majority of evaluations fast and auditable while still handling the long tail of complex criteria.

### UNKNOWN is not a negative

When a required data point is absent from the patient record (e.g. no eGFR on file), the rule engine returns `UNKNOWN` and surfaces an explicit question for the coordinator. This prevents false `NOT_SUPPORTED` verdicts due to missing data — a silent negative would be clinically misleading.

### Clinical fit and recruiting status are kept separate

Recruiting status is a logistical property of the trial, not a property of the patient. The report and ranking model carry both fields independently so a coordinator can see "clinically eligible but trial closed" as a distinct outcome.

### TOP_K cap of 3

The assignment specifies returning no more than three candidates. The ranking service sorts by normalised score and slices at `TOP_K = 3`. Trials with any `NOT_SUPPORTED` criterion receive a hard negative weight (−2.0 per criterion) so they naturally sort below trials with only missing data.

### Parser annotation alias (Python 3.14 + Pydantic v2)

In Python 3.14, annotation evaluation behaviour changed in a way that caused the `LabResult.date` field annotation to resolve incorrectly at runtime when the field name (`date`) matched the imported type name (`datetime.date`). The fix is to import the type under an alias (`from datetime import date as DateType`) so the annotation resolves unambiguously. This does not affect the public API of the model.

### Structured logging with structlog

All services and nodes emit structured JSON-compatible log events via `structlog`. Each event carries a `trace_id` (UUID per pipeline run) so logs from a single patient evaluation can be correlated across all five stages.

---

## Evaluation Metrics

After each `run`, a JSON metrics file is written to `artifacts/metrics/`. It includes:

| Metric | Description |
|:-------|:------------|
| `total_trials_input` | Trials in the dataset before filtering |
| `trials_after_filtering` | Trials that passed the filter stage |
| `filter_rate` | Fraction of trials filtered out |
| `total_criteria_evaluated` | Total criterion evaluations performed |
| `criterion_resolution_rate` | Fraction of criteria resolved to a definite answer (SUPPORTED or NOT_SUPPORTED). Measures how much work the pipeline offloads from the coordinator. |
| `criterion_unresolved_rate` | Fraction requiring human review (complement of resolution rate) |
| `supported_rate` | Fraction of criteria resolved as SUPPORTED |
| `unknown_rate` | Fraction where required patient data was absent |
| `rule_engine_evaluations` | Count of deterministic rule-engine evaluations |
| `llm_engine_evaluations` | Count of LLM-based evaluations |
| `clinical_review_evaluations` | Count of criteria flagged for human review |
| `ranked_supported_trials` | Trials in the final TOP_K with SUPPORTED clinical fit |
| `ranked_human_review_trials` | Trials in the final TOP_K requiring human review |
| `top_trial_score` | Score of the highest-ranked trial |
| `run_duration_ms` | Wall-clock time for the full pipeline run |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_evaluation.py -v
pytest tests/test_graph.py -v
pytest tests/test_report.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_metrics.py -v
```

The test suite (131 tests) runs without any API keys. LLM calls are mocked at the `LLMClient` level.

---

## Example Run

The following is a real pipeline run on patient **P-1842** from the dataset, executed without an OpenAI API key. LLM-evaluated criteria (MEDICATION, CONDITION) fall back to `REQUIRES_CLINICAL_REVIEW` — this is the expected safe-failure behaviour and represents the honest baseline for a no-key environment.

### Command

```bash
python main.py run \
  --patient-json '{"id":"P-1842","age":60,"gender":"male","conditions":["Diabetes","Hypertension","Coronary Heart Disease"],"medications":[],"lab_results":[{"test":"HbA1c","value":7.0,"unit":"%"},{"test":"eGFR","value":91.0,"unit":"mL/min/1.73m2"}]}' \
  --verbose
```

Or using the real dataset entry directly:

```bash
# Extract patient P-1842 and run
python -c "
import json
data = json.load(open('data/Type2-Diabetes-Trial-Agent-Dataset.json'))
p = next(p for p in data['patients'] if p['patient_id'] == 'P-1842')
open('/tmp/p1842.json', 'w').write(json.dumps(p))
"
python main.py run --patient-file /tmp/p1842.json --verbose
```

### Pipeline output (stdout)

```
Loading patient data...
  Patient: P-1842, age 60, male
Loading trials from data/Type2-Diabetes-Trial-Agent-Dataset.json...
  Loaded 36 trials.
Running pre-screening pipeline...

Completed in 9500 ms — 3 trial(s) ranked.
  1. [UNKNOWN] GLUCOSE-MGH: Genetic Links Understood Through Challenge With Oral Semaglutide Exposure at MGH (score: 0.154)
  2. [UNKNOWN] Primary Care Pragmatic, Real World Experience for Automated Insulin Delivery (score: 0.104)
  3. [UNKNOWN] The Effects of the GOLO for Life® Plan With Release Supplement on Weight Loss (score: 0.095)
```

All three ranked trials show `UNKNOWN` clinical fit because the LLM criteria (MEDICATION, CONDITION) could not be evaluated without an API key. The rule engine successfully evaluated AGE and EGFR criteria — the ranking reflects those partial results.

### Report excerpt (`artifacts/reports/P-1842_*.md`)

```markdown
# Clinical Trial Pre-Screening Report

**Generated:** 2026-07-29 15:22 UTC

---

## Patient Profile

| Field        | Value |
|:-------------|:------|
| Patient ID   | `P-1842` |
| Age          | 60 |
| Gender       | male |
| Conditions   | Chronic sinusitis (disorder), Hypertension, Anemia (disorder), Diabetes, ... |
| Medications  | — |

**Lab Results:**
  - hba1c: **7.0 %** _(recorded 2026-04-30)_
  - egfr: **91.0 mL/min/1.73m2** _(recorded 2026-04-25)_

> **Safety note:** This report is a pre-screening aid only. It does not constitute
> a final eligibility decision, diagnosis, or clinical recommendation.

## Executive Summary

Showing **3** top-ranked trial(s) (capped at 3).

| Clinical Fit    | Count |
|:----------------|------:|
| ✅ Supported     | 0 |
| ❌ Not Supported | 0 |
| ❓ Other         | 3 |
| 🔍 Human Review Required | 3 |

## Trial Results

---

### 1. GLUCOSE-MGH: Genetic Links Understood Through Challenge With Oral Semaglutide...
**Trial ID:** `NCT06003153`
**Reason surfaced:** 2/13 evaluated criteria supported; trial is actively recruiting.

| Field            | Status |
|:-----------------|:-------|
| **Clinical Fit** | ❓ Insufficient data |
| **Recruiting**   | 🟢 Actively recruiting |
| **Score**        | `0.154` |
| **Human Review** | ⚠️  Required |

#### Criterion Evaluation

| Criterion ID | Status | Evaluator | Reasoning |
|:-------------|:------:|:---------:|:----------|
| `inc_1` | ✅ SUPPORTED | rule_engine | Patient age 60 is within the required range 18–65. |
| `exc_4` | ✅ SUPPORTED | rule_engine | Patient EGFR 91.0 does not meet criterion < 60.0. |
| `inc_3` | 🔍 REQUIRES_CLINICAL_REVIEW | llm_engine_error | LLM evaluation failed: Missing credentials. |
| `exc_0` | 🔍 REQUIRES_CLINICAL_REVIEW | llm_engine_error | LLM evaluation failed: Missing credentials. |
| ...    | ...    | ...       | ... |

#### Unanswered Questions / Items for Clinical Review

- Please review this criterion manually with the study team.
- Automated evaluation unavailable — please review manually.
```

### Pipeline metrics (`artifacts/metrics/P-1842_*.json`)

```json
{
  "patient_id": "P-1842",
  "timestamp": "2026-07-29T15:22:06Z",
  "metrics": {
    "total_trials_input": 36.0,
    "trials_after_filtering": 13.0,
    "filter_rate": 0.6389,
    "total_criteria_evaluated": 220.0,
    "criterion_resolution_rate": 0.0773,
    "criterion_unresolved_rate": 0.9227,
    "supported_rate": 0.0636,
    "not_supported_rate": 0.0136,
    "unknown_rate": 0.0,
    "rule_engine_evaluations": 41.0,
    "llm_engine_evaluations": 0.0,
    "clinical_review_evaluations": 130.0,
    "llm_error_evaluations": 49.0,
    "ranked_human_review_trials": 3.0,
    "top_trial_score": 0.1538,
    "run_duration_ms": 9500.21
  }
}
```

### What the numbers mean

| Observation | Explanation |
|:------------|:------------|
| Filter rate 63.9% | 23 of 36 trials eliminated before evaluation — non-recruiting or age mismatch |
| Resolution rate 7.7% | Without an API key, only rule-engine criteria (AGE, EGFR) resolve automatically |
| 0 LLM evaluations | No API key present; all MEDICATION/CONDITION criteria fell back to error handler |
| All 3 ranked trials need human review | Expected — no LLM criteria resolved, so every trial has unresolved criteria |
| Run time ~9.5 s | Dominated by sentence-transformer embedding (retrieval stage); evaluation itself is sub-second |

With a configured API key (`OPENAI_API_KEY=sk-...` in `.env`), `llm_engine_evaluations` would replace `llm_error_evaluations`, the resolution rate would rise to an estimated 30–40%, and some trials would reach `SUPPORTED` clinical fit.
# froncort.ai-diabetes
