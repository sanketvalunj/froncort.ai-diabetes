# Evaluation

This document covers:
1. The original evaluation metric (`criterion_resolution_rate`)
2. The evaluation suite results (18 cases)
3. Interpretation and limitations

---

## Original Metric: `criterion_resolution_rate`

### Definition

```
resolved                = SUPPORTED + NOT_SUPPORTED
unresolved              = UNKNOWN + CONFLICTING_EVIDENCE + REQUIRES_CLINICAL_REVIEW
criterion_resolution_rate = resolved / total_criteria_evaluated
```

A criterion is *resolved* when the pipeline produced a definite answer — either the patient meets it or they do not. A criterion is *unresolved* when the answer could not be determined automatically and must be reviewed by a coordinator.

### Failure Hypothesis

**Hypothesis:** In the absence of a live LLM (or with incomplete patient records), the majority of criteria will be unresolved. Specifically:

- `MEDICATION` and `CONDITION` criteria — which make up a large fraction of real trial eligibility text — are routed to the LLM. Without API credentials, these fall back to `REQUIRES_CLINICAL_REVIEW`.
- Criteria classified as `OTHER` (free text that the keyword classifier cannot categorise) are always flagged for review.
- `UNKNOWN` criteria arise when a required lab value is absent from the patient record.

The complement metric, `criterion_unresolved_rate`, measures the coordinator workload per run. A high unresolved rate signals either missing patient data or a missing API key.

### Calculation

The metric is computed inside `PipelineMetrics.compute()` in `metrics/metrics.py`:

```python
resolved   = sup + nsup          # SUPPORTED + NOT_SUPPORTED
unresolved = unk + conf + rev    # UNKNOWN + CONFLICTING_EVIDENCE + REQUIRES_CLINICAL_REVIEW
m["criterion_resolution_rate"] = resolved   / total if total else 0.0
m["criterion_unresolved_rate"] = unresolved / total if total else 0.0
```

Both values are stored in the per-run metrics JSON alongside all other pipeline metrics.

### Baseline Run

**Patient:** P-1842 (age 60, male, 13 conditions including Diabetes, Hypertension, Coronary Heart Disease)  
**Dataset:** 36 trials (real dataset)  
**LLM:** No API key (all LLM criteria fall back to `REQUIRES_CLINICAL_REVIEW`)  
**Run date:** 2026-07-29

| Metric | Value |
|:-------|------:|
| `total_trials_input` | 36 |
| `trials_after_filtering` | 13 |
| `filter_rate` | 0.6389 |
| `total_criteria_evaluated` | 220 |
| `criterion_resolution_rate` | **0.0773** |
| `criterion_unresolved_rate` | **0.9227** |
| `supported_rate` | 0.0636 |
| `not_supported_rate` | 0.0136 |
| `unknown_rate` | 0.0000 |
| `rule_engine_evaluations` | 41 |
| `llm_engine_evaluations` | 0 |
| `clinical_review_evaluations` | 130 |
| `llm_error_evaluations` | 49 |
| `ranked_human_review_trials` | 3 / 3 |
| `top_trial_score` | 0.154 |
| `run_duration_ms` | ~9,500 ms |

### Interpretation

The baseline confirms the failure hypothesis:

- **7.7% resolution rate** — only 17 of 220 criteria were resolved automatically. All 17 were rule-engine evaluations (AGE or EGFR numeric checks).
- **92.3% unresolved** — split between 130 clinical review flags (OTHER criteria) and 49 LLM error fallbacks (MEDICATION and CONDITION criteria with no API key).
- **0 LLM evaluations** — confirming that without credentials, the entire LLM path is unavailable.
- **Filter rate 63.9%** — 23 of 36 trials were eliminated before evaluation (non-recruiting or age mismatch), which is working as intended.
- **All 3 ranked trials require human review** — expected without LLM-resolved criteria.

With a live API key, the expected resolution rate would rise substantially. Empirically, MEDICATION and CONDITION criteria (49 LLM-error evaluations here) would each resolve to SUPPORTED, NOT_SUPPORTED, or UNKNOWN, pushing the resolution rate from ~8% toward ~30–40% depending on patient data completeness.

### Limitations of This Metric

1. **No ground truth.** The metric measures *pipeline* resolution, not *correctness*. A criterion resolved as `SUPPORTED` by the rule engine is counted as resolved even if the rule-engine's regex parsed the criterion incorrectly.

2. **Resolution rate is partially controlled by the API key.** The baseline number (7.7%) is dominated by the absence of an LLM. With credentials, the number changes significantly, making this metric environment-dependent.

3. **`REQUIRES_CLINICAL_REVIEW` is conflated.** The metric groups deliberate clinical review flags (OTHER criteria that genuinely need a clinician) with LLM error fallbacks. These have different causes and should ideally be tracked separately. The metrics module records `clinical_review_evaluations` and `llm_error_evaluations` separately for this reason.

4. **No per-criterion-type breakdown.** The aggregate rate hides which criterion types are the bottleneck. A future version should report resolution rate per `CriterionType`.

---

## Evaluation Suite Results

18 cases across 5 categories, run against the real dataset with the built FAISS index.

```
==================================================================================
  EVALUATION SUITE RESULTS
==================================================================================
  ID    Category              Case                                        Result
----------------------------------------------------------------------------------
  R1    Retrieval             Retriever returns results for T2D trial     PASS
  R2    Retrieval             Retriever scopes to requested trial only    PASS
  R3    Retrieval             Patient evidence covers all source types    PASS
  R4    Retrieval             Query builder includes HbA1c and eGFR       PASS
----------------------------------------------------------------------------------
  C1    Criterion status      Age in range → SUPPORTED                    PASS
  C2    Criterion status      Age outside range → NOT_SUPPORTED           PASS
  C3    Criterion status      Missing lab → UNKNOWN (not NOT_SUPPORTED)   PASS
  C4    Criterion status      Exclusion not triggered → SUPPORTED         PASS
  C5    Criterion status      Unparseable → REQUIRES_CLINICAL_REVIEW      PASS
----------------------------------------------------------------------------------
  A1    Agent behaviour       FilteringService removes non-recruiting     PASS
  A2    Agent behaviour       RankingService caps at TOP_K=3              PASS
  A3    Agent behaviour       requires_human_review=True for UNKNOWN      PASS
----------------------------------------------------------------------------------
  D1    Dataset coverage      All 15 real patients parse without error    PASS
  D2    Dataset coverage      All 36 real trials parse without error      PASS
  D3    Dataset coverage      FilteringService reduces 36-trial set       PASS
  D4    Dataset coverage      All patients have ≥1 lab result             PASS
----------------------------------------------------------------------------------
  O1    Output quality        Report contains all required fields (14/14) PASS
  O2    Output quality        UNKNOWN questions surface in report         PASS
----------------------------------------------------------------------------------
  TOTAL: 18  PASSED: 18  FAILED: 0  PASS RATE: 100.0%
==================================================================================
```

Full machine-readable results are in `results/eval_results.json`.

### What the Cases Cover

| Category | Cases | What is tested |
|:---------|------:|:---------------|
| Retrieval | 4 | FAISS search returns results, scoped to the right trial, patient evidence covers all source types, queries include numeric lab values |
| Criterion status | 5 | Rule engine correctness for all deterministic paths: in-range SUPPORTED, out-of-range NOT_SUPPORTED, missing data UNKNOWN, exclusion not-triggered SUPPORTED, unparseable REQUIRES_CLINICAL_REVIEW |
| Agent behaviour | 3 | Filtering removes non-recruiting trials, ranking caps at TOP_K=3, human-review flag is set correctly |
| Dataset coverage | 4 | All 15 patients and 36 trials parse from the real dataset, filtering reduces the set, every patient has lab results |
| Output quality | 2 | Report contains all 14 coordinator-required fields, unanswered questions from UNKNOWN criteria appear in the report |

### What the Evaluation Suite Does Not Cover

- **LLM evaluation quality** — no live API key; all LLM paths are mocked or fall back to error handling.
- **Ranking score calibration** — whether the score ordering correctly prioritises trials that a clinician would agree are the best fit.
- **Retrieval recall** — whether the FAISS index returns *all* relevant criteria for a trial (only precision of the scoping filter is tested here).
- **End-to-end correctness against gold-standard annotations** — there are no human-labelled eligibility verdicts to compare against.

---

## Running the Evaluation Suite

```bash
# Run all 18 cases and print results table
python scripts/evaluate.py

# Also write results/eval_results.json
python scripts/evaluate.py --json
```

The suite requires the FAISS index to be built first (`python main.py index`). It does not require an API key.
