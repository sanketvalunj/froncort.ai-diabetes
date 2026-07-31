from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from pathlib import Path


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalSettings(BaseModel):
    top_k: int = 2           # max evidence chunks retrieved per criterion query
    score_threshold: float = 0.3
    chunk_size: int = 250    # max tokens per indexed text chunk
    chunk_overlap: int = 30  # token overlap between adjacent chunks


class EvaluationSettings(BaseModel):
    confidence_threshold: float = 0.7


class PathSettings(BaseModel):
    data: Path = Path("data/Type2-Diabetes-Trial-Agent-Dataset.json")
    vector_store: Path = Path("data/vector_store")
    reports: Path = Path("artifacts/reports")
    logs: Path = Path("artifacts/logs")
    metrics: Path = Path("artifacts/metrics")


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    paths: PathSettings = PathSettings()
    google_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None

    model_config = {"env_file": ".env", "env_nested_delimiter": "__"}


settings = Settings()
