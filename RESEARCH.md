# Research Notes

This document describes the research process, design alternatives considered, and the rationale behind key technical decisions in the Clinical Trial Pre-Screening Assistant.

---

## Research Process

### Problem framing

The core task is eligibility pre-screening: given a patient record and a set of clinical trials, determine which trials are worth forwarding to a clinical coordinator. This is not a binary classification problem — the output must explain *why* each trial was surfaced and flag *what is missing*, so a coordinator can make an informed judgement rather than trust a black-box verdict.

That framing led to three requirements that shaped every design decision:

1. **Auditability** — every conclusion must be traceable to a specific piece of evidence.
2. **Conservative uncertainty** — missing data must never silently become a negative verdict.
3. **Structured output** — the result must be machine-readable (for downstream tools) and human-readable (for coordinators).

### Literature and tooling survey

Before settling on an architecture, the following were reviewed:

- **LangGraph** (LangChain ecosystem) — stateful agentic graphs with explicit node/edge topology.
- **LlamaIndex** — document-centric pipelines with built-in retrieval abstractions.
- **Haystack** — production-oriented NLP pipelines with a component registry.
- **Raw LangChain LCEL** — composable chains without an explicit state machine.
- **Custom orchestration** — hand-written Python with no framework.

For the vector store:

- **FAISS** (Facebook AI Similarity Search) — in-process, no server required.
- **ChromaDB** — embedded or server-hosted, richer metadata filtering.
- **Weaviate** — full-featured vector database, requires a running service.
- **Pinecone** — managed cloud vector database.

For embeddings:

- **sentence-transformers/all-MiniLM-L6-v2** — fast, lightweight, runs locally.
- **OpenAI text-embedding-ada-002 / text-embedding-3-small** — higher quality, requires API call per query.

---

## Design Alternatives Considered

### Orchestration framework

| Option | Pros | Cons | Decision |
|:-------|:-----|:-----|:---------|
| **LangGraph** | Explicit state machine, easy to test each node in isolation, first-class support for typed state | Slightly more setup than plain LCEL | **Selected** |
| LlamaIndex | Good retrieval abstractions out of the box | Pipeline topology less explicit; harder to audit state transitions | Not selected |
| Haystack | Production-hardened | Large dependency surface; component registry adds indirection | Not selected |
| Raw LCEL | Minimal boilerplate | No built-in state; managing inter-stage data becomes fragile at scale | Not selected |
| Custom Python | Full control, zero dependencies | All orchestration logic is bespoke; harder to extend | Not selected |

### Vector store

| Option | Pros | Cons | Decision |
|:-------|:-----|:-----|:---------|
| **FAISS** | In-process, no server, fast for small corpora, deterministic | No built-in persistence API (managed manually); no native metadata filtering | **Selected** |
| ChromaDB | Metadata filtering, persistent by default, simple Python API | Slightly heavier than FAISS for small corpora | Close alternative |
| Weaviate / Pinecone | Rich features, cloud-hosted option | Requires external service or network access — too heavy for a self-contained assignment | Not selected |

### Embedding model

| Option | Pros | Cons | Decision |
|:-------|:-----|:-----|:---------|
| **all-MiniLM-L6-v2** | Runs locally, no API key required, 384-dimensional output, fast | Lower semantic resolution than OpenAI embeddings for specialist text | **Selected** |
| OpenAI text-embedding-3-small | Higher quality, especially for medical terminology | Requires API call + key; adds latency and cost to the index-build step | Not selected |

The trade-off is acceptable for a 36-trial corpus where retrieval is a soft signal (evidence retrieval does not gate evaluation — the rule engine evaluates criteria even if retrieval returns nothing).

### Evaluation strategy

Three strategies were considered:

1. **LLM-only** — pass every criterion and all patient data to the LLM and ask for a verdict.
2. **Rule-only** — write deterministic parsers for every criterion type.
3. **Hybrid** — rules for numeric/boolean criteria, LLM for semantic criteria.

LLM-only has the highest coverage but the lowest auditability and the highest latency and cost. Rule-only is fully auditable but breaks on free-text criteria that require understanding drug classes, disease synonyms, or nuanced clinical language. The hybrid approach retains determinism where it is achievable and falls back to the LLM only where natural language understanding is genuinely needed.

The boundary between rule and LLM was drawn at criterion type:
- `AGE`, `HBA1C`, `EGFR`, `RECRUITING` → rule engine (numeric or boolean, fully parseable)
- `MEDICATION`, `CONDITION` → LLM (require semantic matching)
- `OTHER` → flagged for human review (cannot be automated without clinical domain knowledge)

---

## Why LangGraph

LangGraph was chosen over alternatives for four specific reasons:

**1. Typed shared state.** `AgentState` is a Pydantic model that every node receives and returns. This makes the data contract between stages explicit and testable — any node can be tested in isolation by constructing a partial `AgentState` and asserting on the returned state.

**2. Explicit topology.** The five-node linear graph (`filter → retrieval → evaluation → ranking → report`) is declared once in `workflow.py`. There is no implicit call chain hidden inside a class hierarchy. Adding a new stage or inserting a conditional branch is a one-line graph change.

**3. Compile-time graph validation.** `graph.compile()` validates the graph structure before any patient is processed, catching missing edges or unreachable nodes early.

**4. Future extensibility.** LangGraph supports conditional edges and cycles, which would be needed if a future version requires iterative clarification (e.g. asking the LLM to re-evaluate after a coordinator provides missing data). Adding that behaviour requires adding an edge and a condition function — not refactoring the entire pipeline.

---

## Why FAISS

FAISS was chosen for the vector store for the following reasons:

**1. Zero infrastructure.** The index lives in a local directory and is loaded in-process. There is no server to start, no network call during query time, and no credentials to manage.

**2. Appropriate scale.** The dataset has 36 trials. FAISS flat search (L2) over a few hundred criterion embeddings is sub-millisecond. A distributed vector database would add latency with no benefit at this scale.

**3. Deterministic results.** Given the same index and query vector, FAISS returns the same neighbours every time. This matters for reproducibility — repeated runs on the same patient should produce the same evidence retrieval, all else being equal.

**4. Transparency of failure modes.** If retrieval quality is poor, the rule engine still evaluates numeric criteria correctly (it reads directly from `patient.lab_results`). FAISS retrieval is a *supporting signal*, not a hard dependency. This means retrieval failures degrade gracefully rather than silently producing wrong verdicts.

---

## Limitations

### Current system

1. **Criterion parser coverage.** The eligibility text parser uses a keyword map for type detection. Criteria written in unusual phrasing (e.g. "participants must not have impaired kidney clearance") may be misclassified as `OTHER` rather than `EGFR`, causing them to be flagged for clinical review instead of evaluated automatically. Coverage improves with a richer keyword map or a classification LLM.

2. **LLM hallucination on MEDICATION/CONDITION criteria.** The LLM evaluator may produce `SUPPORTED` verdicts for drug classes where the patient's actual medication is not equivalent (e.g. classifying a patient on glipizide as meeting a "current sulfonylurea" criterion when the trial requires a specific agent). The prompt asks for explicit reasoning and a confidence score, but does not enforce chain-of-thought or self-consistency sampling.

3. **No de-duplication of patient evidence across trials.** `extract_patient_evidence` runs once per run and is shared, but `retrieve_for_trial` is called per trial. In a large trial corpus, repeated embedding calls could become slow. A query cache would address this.

4. **FAISS flat index does not scale.** For corpora with tens of thousands of trials, a flat L2 index becomes slow. An IVF (inverted file) or HNSW index would be needed. The current `FAISSVectorStore` abstraction encapsulates this; swapping the index type requires only a change inside `vectorstore.py`.

5. **Date-sensitive criteria not evaluated.** Some trials specify criteria like "diagnosis within the last 5 years" or "lab result within 3 months". The current rule engine does not compare criterion dates against lab result dates. This is a known gap and would require extending `LabResult.date` usage in the rule engine.

6. **Single-patient, single-run design.** The CLI evaluates one patient per invocation. A batch mode (evaluating all 15 patients against all 36 trials in one run) would be straightforward to add as a `batch` CLI command.

### Dataset

The dataset contains 15 synthetic patients and 36 trials derived from ClinicalTrials.gov. It is not a representative sample — it is designed to exercise the pipeline. Results on this dataset should not be interpreted as indicative of real-world eligibility pre-screening accuracy.

---

## Future Improvements

1. **Structured eligibility criteria.** Parse the eligibility text into a structured criterion schema (FHIR `EligibilityRequest` or similar) at index time, so the rule engine can evaluate a wider range of criteria without relying on regex against free text.

2. **Confidence-weighted ranking.** The current score is a simple weighted sum. A more principled approach would propagate LLM confidence scores into the ranking, penalising trials where the LLM was uncertain even if it returned `SUPPORTED`.

3. **Active clarification loop.** When criteria return `UNKNOWN`, the agent could generate targeted questions for the coordinator and re-evaluate once answers are provided. LangGraph's conditional edge support makes this a natural extension.

4. **FHIR-native patient input.** The current patient model is a simplified flat schema. Accepting FHIR R4 `Patient` + `Observation` + `MedicationStatement` bundles directly would allow the pipeline to be used with real EHR exports without a conversion step.

5. **Evaluation against gold-standard annotations.** The current evaluation suite measures pipeline behaviour (criterion status correctness, output completeness). A higher-quality evaluation would compare against manually annotated eligibility verdicts from clinical experts, measuring precision/recall of `SUPPORTED` and `NOT_SUPPORTED` verdicts.
