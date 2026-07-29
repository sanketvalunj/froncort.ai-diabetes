import pytest
from app.models.patient import Patient, LabResult
from app.models.trial import Trial, CriterionType
from app.retrieval.parser import parse_patient, parse_trials, _detect_criterion_type


@pytest.mark.parametrize("description,expected", [
    ("Patients aged 18-75 years",           CriterionType.AGE),
    ("Must be 18 years old or older",       CriterionType.AGE),
    ("HbA1c >= 7.5%",                       CriterionType.HBA1C),
    ("Hemoglobin A1c between 7 and 11",     CriterionType.HBA1C),
    ("A1c must be below 9.0",               CriterionType.HBA1C),
    ("eGFR >= 30 mL/min",                   CriterionType.EGFR),
    ("Renal function must be adequate",     CriterionType.EGFR),
    ("No significant kidney disease",       CriterionType.EGFR),
    ("Creatinine clearance > 60",           CriterionType.EGFR),
    ("Currently recruiting participants",   CriterionType.RECRUITING),
    ("Enrollment open",                     CriterionType.RECRUITING),
    ("Currently on metformin treatment",    CriterionType.MEDICATION),
    ("No prior insulin use",                CriterionType.MEDICATION),
    ("Diagnosis of Type 2 Diabetes",        CriterionType.CONDITION),
    ("Confirmed diabetes diagnosis",        CriterionType.CONDITION),
    ("No significant other disease",        CriterionType.CONDITION),
    ("Signed informed consent obtained",    CriterionType.OTHER),
    ("Able to comply with study visits",    CriterionType.OTHER),
])
def test_criterion_type_detection(description, expected):
    assert _detect_criterion_type(description) == expected


def test_parse_patient_from_raw_dict():
    raw = {"id": "p_test", "age": 55, "gender": "Female",
           "conditions": ["Type 2 Diabetes"], "medications": ["metformin"],
           "lab_results": [{"test": "HbA1c", "value": 7.8, "unit": "%"},
                           {"test": "eGFR",  "value": 62.0, "unit": "mL/min/1.73m2"}],
           "medical_history": "Diagnosed 5 years ago."}
    patient = parse_patient(raw)
    assert isinstance(patient, Patient)
    assert patient.age == 55
    assert patient.lab_results[0].test == "HbA1c"


def test_parse_patient_missing_optional_fields():
    patient = parse_patient({"id": "p_min", "age": 40, "gender": "Male"})
    assert patient.conditions == []
    assert patient.lab_results == []


def test_parse_trials_returns_list_of_trials():
    raw = [{"id": "T001", "title": "Test Trial", "phase": "Phase 2",
            "status": "Recruiting", "description": "A test trial.",
            "inclusion_criteria": [{"description": "Aged 18-65 years"},
                                   {"description": "HbA1c >= 7.5%"}],
            "exclusion_criteria": [{"description": "eGFR < 30"}]}]
    trials = parse_trials(raw)
    assert len(trials) == 1
    assert trials[0].inclusion_criteria[0].type == CriterionType.AGE
    assert trials[0].inclusion_criteria[1].type == CriterionType.HBA1C
    assert trials[0].exclusion_criteria[0].type == CriterionType.EGFR


def test_parse_trials_missing_fields_use_defaults():
    trials = parse_trials([{"id": "T_empty", "title": "Empty Trial"}])
    assert trials[0].inclusion_criteria == []


def test_parse_trials_plain_string_criteria():
    raw = [{"id": "T_str", "title": "X",
            "inclusion_criteria": ["Age 18 to 70 years", "HbA1c >= 7.0"]}]
    trials = parse_trials(raw)
    assert trials[0].inclusion_criteria[0].type == CriterionType.AGE


def test_parse_trials_flat_criteria_list():
    raw = [{"id": "T_flat", "title": "Y",
            "criteria": [{"description": "Age 18-65", "is_inclusion": True},
                         {"description": "No kidney disease", "is_inclusion": False}]}]
    trials = parse_trials(raw)
    assert len(trials[0].inclusion_criteria) == 1
    assert len(trials[0].exclusion_criteria) == 1


def test_sample_patient_fixture(sample_patient):
    assert sample_patient.age == 62
    assert "metformin" in sample_patient.medications
    hba1c = next(lr for lr in sample_patient.lab_results if "hba1c" in lr.test.lower())
    assert hba1c.value == 8.2


def test_sample_trials_fixture(sample_trials):
    assert len(sample_trials) == 3
    statuses = {t.status for t in sample_trials}
    assert "Recruiting" in statuses
    assert "Completed"  in statuses


def test_sample_state_fixture(sample_state):
    assert sample_state.patient.id == "patient_001"
    assert len(sample_state.all_trials) == 3


# ── Task 4 retrieval pipeline tests ──────────────────────────────────────────

import tempfile
import numpy as np
from unittest.mock import MagicMock
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vectorstore import FAISSVectorStore
from app.retrieval.retriever import EvidenceRetriever
from app.models.evaluation import Evidence


def _mock_retriever():
    r = EvidenceRetriever.__new__(EvidenceRetriever)
    r._settings        = MagicMock()
    r._index_loaded    = True
    r._embedding_service = MagicMock()
    r._vectorstore     = MagicMock()
    return r


def test_extract_patient_evidence_min_count(sample_patient):
    r = _mock_retriever()
    ev = r.extract_patient_evidence(sample_patient)
    assert len(ev) >= 4
    assert "demographics" in {e.source for e in ev}


def test_extract_patient_evidence_all_patient(sample_patient):
    r = _mock_retriever()
    ev = r.extract_patient_evidence(sample_patient)
    assert all(e.retrieved_from == "patient" for e in ev)


def test_build_queries_min_count(sample_patient):
    r = _mock_retriever()
    queries = r.build_queries(sample_patient)
    assert len(queries) >= 3


def test_build_queries_includes_demographics(sample_patient):
    r = _mock_retriever()
    queries = r.build_queries(sample_patient)
    assert any("62" in q for q in queries)
    assert any("hba1c" in q.lower() or "a1c" in q.lower() for q in queries)


@pytest.fixture
def tiny_vectorstore(tmp_path):
    embeddings = np.random.rand(3, 8).astype("float32")
    meta = [{"trial_id": "t1", "source": "inclusion", "text": "age 18-65"},
            {"trial_id": "t1", "source": "exclusion", "text": "no kidney disease"},
            {"trial_id": "t2", "source": "inclusion", "text": "hba1c > 7.5"}]
    store = FAISSVectorStore(tmp_path)
    store.build(embeddings, meta)
    store.save()
    return tmp_path, embeddings, meta


def test_vectorstore_build_and_search(tiny_vectorstore):
    tmp_path, embeddings, _ = tiny_vectorstore
    store = FAISSVectorStore(tmp_path)
    store.build(np.random.rand(3, 8).astype("float32"),
                [{"trial_id": "t1", "source": "x", "text": "age 18-65"},
                 {"trial_id": "t1", "source": "x", "text": "b"},
                 {"trial_id": "t2", "source": "x", "text": "c"}])
    results = store.search(embeddings[0], top_k=2)
    assert len(results) == 2


def test_vectorstore_save_load_roundtrip(tiny_vectorstore):
    tmp_path, embeddings, _ = tiny_vectorstore
    store2 = FAISSVectorStore(tmp_path)
    store2.load()
    assert len(store2.metadata) == 3
    results = store2.search(embeddings[2], top_k=1)
    assert results[0][0]["text"] == "hba1c > 7.5"


def test_vectorstore_search_respects_top_k(tiny_vectorstore):
    tmp_path, embeddings, _ = tiny_vectorstore
    store = FAISSVectorStore(tmp_path)
    store.load()
    assert len(store.search(embeddings[0], top_k=1)) == 1


def test_retrieve_for_trial_returns_trial_evidence(tmp_path, sample_patient):
    emb_svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
    texts   = ["Inclusion: Age 18-75 years", "Inclusion: HbA1c >= 7.5%"]
    embs    = emb_svc.embed_batch(texts)
    meta    = [{"trial_id": "trial_001", "source": "inclusion", "text": texts[0]},
               {"trial_id": "trial_001", "source": "inclusion", "text": texts[1]}]
    store   = FAISSVectorStore(tmp_path)
    store.build(embs, meta)
    store.save()

    mock_settings = MagicMock()
    mock_settings.embeddings.model   = "sentence-transformers/all-MiniLM-L6-v2"
    mock_settings.retrieval.top_k    = 5
    mock_settings.retrieval.score_threshold = 0.0
    mock_settings.paths.vector_store = tmp_path

    retriever = EvidenceRetriever(mock_settings)
    retriever._vectorstore  = store
    retriever._index_loaded = True

    queries  = retriever.build_queries(sample_patient)
    evidence = retriever.retrieve_for_trial("trial_001", queries)
    assert len(evidence) > 0
    assert all(e.retrieved_from == "trial" for e in evidence)
