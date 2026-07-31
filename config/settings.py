from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalSettings(BaseModel):
    top_k: int = 2
    score_threshold: float = 0.3
    chunk_size: int = 250
    chunk_overlap: int = 30


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

    model_config = {
        # Only load .env when it actually exists (local dev).
        # On Render there is no .env — values come from environment variables.
        # pydantic-settings reads env vars automatically regardless of env_file.
        "env_file": ".env" if Path(".env").exists() else None,
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }


settings = Settings()
