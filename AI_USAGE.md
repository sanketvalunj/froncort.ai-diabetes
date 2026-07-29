# AI Usage

This document describes how AI tools were used during development of the Clinical Trial Pre-Screening Assistant, including specific examples of suggestions accepted, suggestions rejected or modified, and how the final behaviour was verified.

---

## Tools Used

| Tool | Purpose |
|:-----|:--------|
| **Kiro (Claude-based)** | Primary development assistant — architecture design, code generation, debugging, test writing, documentation |
| **GitHub Copilot** | Inline completions for boilerplate (Pydantic model fields, pytest fixtures, regex patterns) |

---

## How AI Was Used

AI assistance was used throughout the project, but always in a review-and-verify loop rather than as an autonomous code generator. The workflow was:

1. Describe the design intent and constraints to the AI.
2. Review the generated code against the requirements before accepting it.
3. Run the test suite after every non-trivial change.
4. Interrogate any generated code that was not immediately obvious (ask the AI to explain its reasoning).

The AI was treated as a fast-drafting tool, not an authority. Every design decision in this project was made by a human after reviewing the AI's suggestion and the alternatives.

---

## Example: Suggestion Accepted

**Context:** Designing the `CriterionStatus` vocabulary.

The AI suggested using five status values — `SUPPORTED`, `NOT_SUPPORTED`, `UNKNOWN`, `CONFLICTING_EVIDENCE`, and `REQUIRES_CLINICAL_REVIEW` — rather than a simpler `ELIGIBLE / INELIGIBLE / UNKNOWN` three-value scheme.

**Why it was accepted:**

The five-value scheme is more useful to a clinical coordinator. `CONFLICTING_EVIDENCE` and `REQUIRES_CLINICAL_REVIEW` are meaningfully different outcomes: the first means "the data says contradictory things", the second means "the criterion cannot be resolved by looking at the data at all". Collapsing both into a generic "uncertain" bucket would lose information that a coordinator needs to decide what to do next.

The suggestion was accepted verbatim and became the `CriterionStatus` enum in `app/models/evaluation.py`.

---

## Example: Suggestion Rejected / Modified

**Context:** Implementing the criterion evaluator router.

The AI initially suggested routing all criteria through the LLM by default, with the rule engine as an opt-in path for criteria explicitly marked as numeric. The default routing looked like:

```python
# AI's initial suggestion
def evaluate(self, criterion, patient, ...):
    if criterion.type in _RULE_TYPES:
        return self._rule_engine.evaluate(criterion, patient)
    return self._llm_evaluator.evaluate(...)   # default for everything else
```

This was **rejected** for two reasons:

1. **Cost and latency.** Routing `RECRUITING` and `AGE` criteria to the LLM on every evaluation adds unnecessary API calls. These are deterministic boolean/numeric checks that should never require language understanding.
2. **Auditability.** LLM evaluations are probabilistic. Using the rule engine for criteria where a deterministic answer is available gives coordinators a higher-confidence, fully traceable verdict rather than an LLM opinion.

The modified version inverts the default: the rule engine is the primary path for all four deterministic criterion types; the LLM is called only for `MEDICATION` and `CONDITION`; `OTHER` is flagged for clinical review rather than sent to the LLM (which would likely produce an unreliable verdict on criteria that require direct clinician judgement).

```python
# Modified version — rule engine is the primary path
_RULE_TYPES = {CriterionType.AGE, CriterionType.HBA1C,
               CriterionType.EGFR, CriterionType.RECRUITING}
_LLM_TYPES  = {CriterionType.MEDICATION, CriterionType.CONDITION}

def evaluate(self, criterion, patient, ...):
    if criterion.type in _RULE_TYPES:
        return self._rule_engine.evaluate(criterion, patient)
    if criterion.type in _LLM_TYPES:
        return self._llm_evaluator.evaluate(...)
    # OTHER → clinical review flag, not LLM
    return CriterionEvaluation(status=REQUIRES_CLINICAL_REVIEW, ...)
```

---

## How Final Behaviour Was Verified

### Automated tests

The complete test suite (131 tests) was run after every significant change. Tests cover:

- **Rule engine** — each criterion type (AGE, HBA1C, EGFR, RECRUITING), boundary conditions (in-range, below-range, above-range, single bound), missing data (`UNKNOWN`), and unparseable text (`REQUIRES_CLINICAL_REVIEW`).
- **LLM evaluator** — happy path, `ELIGIBLE` alias mapping, API error fallback, JSON parse error fallback.
- **Router** — dispatch to correct evaluator for each criterion type.
- **Filtering service** — recruiting status filter, age pre-filter, filter reason coverage.
- **Ranking service** — score ordering, `TOP_K` cap, clinical fit derivation, human-review flag.
- **Report generator** — all coordinator-facing fields, structured data dict, edge cases.
- **Metrics** — all computed metrics, save/load roundtrip.
- **LangGraph workflow** — end-to-end run with mocked retrieval and evaluation.

### Parser fix verification

A specific regression was introduced during development when Python 3.14's annotation evaluation behaviour caused `LabResult.date` to resolve as `NoneType` rather than `datetime.date`. This was diagnosed by:

1. Running the parser against all 15 real patients and capturing the exact Pydantic validation error.
2. Inspecting `LabResult.model_fields['date'].annotation` at runtime and confirming it showed `<class 'NoneType'>`.
3. Applying the alias fix (`from datetime import date as DateType`) and re-verifying that the annotation resolved to `datetime.date | None`.
4. Re-running the full test suite (131 passed) to confirm no regressions.

### Manual spot-checks

After the full pipeline was integrated, several manual spot-checks were performed:

- Parsing all 15 patients and 36 trials from the real dataset and confirming zero parse errors.
- Running the `info` CLI command and verifying the resolved configuration matched the `.env` file.
- Running the `index` command and confirming the FAISS index was written to `data/vector_store/`.
- Running the `run` command on `data/example_patient.json` and reviewing the generated Markdown report for correctness of criterion status labels, evidence sources, and human-review flags.
