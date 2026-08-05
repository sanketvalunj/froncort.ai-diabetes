# Handles retrieval of relevant trial criteria using embeddings.
from typing import List

from app.models.evaluation import Evidence
from app.models.patient import Patient
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vectorstore import FAISSVectorStore


class EvidenceRetriever:
    def __init__(self, settings):
        self._settings = settings
        self._embedding_service = EmbeddingService(settings.embeddings.model)
        self._vectorstore = FAISSVectorStore(settings.paths.vector_store)
        self._index_loaded = False

    def _ensure_index(self) -> None:
        if not self._index_loaded:
            self._vectorstore.load()
            self._index_loaded = True

    def extract_patient_evidence(self, patient: Patient) -> List[Evidence]:
        evidence: List[Evidence] = []
        evidence.append(Evidence(
            text=f"Patient: {patient.age} year old {patient.gender}",
            source="demographics", retrieved_from="patient", relevance_score=1.0,
        ))
        for condition in patient.conditions:
            evidence.append(Evidence(text=condition, source="conditions",
                                     retrieved_from="patient", relevance_score=1.0))
        for medication in patient.medications:
            evidence.append(Evidence(text=medication, source="medications",
                                     retrieved_from="patient", relevance_score=1.0))
        for lab in patient.lab_results:
            evidence.append(Evidence(text=f"{lab.test}: {lab.value} {lab.unit}",
                                     source="lab_results", retrieved_from="patient",
                                     relevance_score=1.0,
                                     evidence_id=lab.source_id,
                                     date=str(lab.date) if lab.date else ""))
        return evidence

    def build_queries(self, patient: Patient) -> List[str]:
        queries: List[str] = []
        queries.append(f"Type 2 diabetes clinical trial {patient.age} year old {patient.gender} patient")
        for condition in patient.conditions:
            queries.append(f"{condition} eligibility criteria")
        for lab in patient.lab_results:
            name_lower = lab.test.lower()
            if "hba1c" in name_lower or "a1c" in name_lower:
                queries.append(f"HbA1c {lab.value}% threshold criterion")
            elif "egfr" in name_lower or "gfr" in name_lower:
                queries.append(f"eGFR {lab.value} renal function criterion")
        if patient.medications:
            queries.append(f"patient taking {', '.join(patient.medications)} medication eligibility")
        return queries

    def retrieve_for_trial(self, trial_id: str, queries: List[str]) -> List[Evidence]:
        self._ensure_index()
        top_k     = self._settings.retrieval.top_k
        threshold = self._settings.retrieval.score_threshold
        seen_texts: set = set()
        evidence: List[Evidence] = []
        for query in queries:
            query_vec = self._embedding_service.embed(query)
            results   = self._vectorstore.search(query_vec, top_k)
            for meta, distance in results:
                if meta.get("trial_id") != trial_id:
                    continue
                score = 1.0 / (1.0 + distance)
                if score < threshold:
                    continue
                text: str = meta.get("text", "")
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                evidence.append(Evidence(text=text, source=meta.get("source", "trial"),
                                         retrieved_from="trial",
                                         relevance_score=round(score, 4)))
        return evidence
