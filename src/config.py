from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

class AppConfig(BaseModel):
    name: str
    version: str
    environment: str
    log_level: str

class DatabaseConfig(BaseModel):
    url: str
    echo_sql: bool = False

class NLPConfig(BaseModel):
    lowercase: bool = True
    strip_html: bool = True
    remove_special_chars: bool = False
    remove_stopwords: bool = False
    lemmatize: bool = False

class ClassicalConfig(BaseModel):
    model_type: str = "logistic_regression"
    max_features: int = 10000
    ngram_range: List[int] = Field(default_factory=lambda: [1,2])
    min_df: int = 1
    max_df: float = 0.98
    c_regularization: float = 1.0
    artifact_path: str

class TransformerConfig(BaseModel):
    model_name: str
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    artifact_path: str

class ModelsConfig(BaseModel):
    default_classifier: str = "classical"
    intent_classes: List[str]
    urgency_levels: List[str]
    classical: ClassicalConfig
    transformer: TransformerConfig

class SentimentConfig(BaseModel):
    model_name: str

class RAGConfig(BaseModel):
    knowledge_base_dir: str
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str
    top_k: int = 3
    similarity_threshold: float = 0.55
    index_path: str
    metadata_path: str

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return value

class LLMConfig(BaseModel):
    provider: str = "mock"
    model_name: str
    temperature: float = 0.2
    max_output_tokens: int = 512
    mock_responses_in_test: bool = True

class IntelligenceConfig(BaseModel):
    confidence_threshold: float = 0.60
    high_priority_sentiments: List[str] = Field(default_factory=lambda:["negative"])
    financial_intents: List[str] = Field(default_factory=list)
    critical_intents: List[str] = Field(default_factory=list)
    escalation_similarity_threshold: float = 0.55

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class DashboardConfig(BaseModel):
    port: int = 8501

class Settings(BaseModel):
    app: AppConfig
    database: DatabaseConfig
    nlp: NLPConfig
    models: ModelsConfig
    sentiment: SentimentConfig
    rag: RAGConfig
    llm: LLMConfig
    intelligence: IntelligenceConfig
    api: APIConfig
    dashboard: DashboardConfig

    def path(self, relative: str) -> Path:
        """Resolve a project-relative path against the repository root."""
        return ROOT_DIR / relative

    def ensure_directories(self) -> None:
        """Create directories required by the application without creating model files."""
        for rel in [
            "data/raw", "data/processed", "data/knowledge_base",
            "models/tfidf", "models/bert", "models/sentiment", "models/embeddings",
            "artifacts/evaluation_reports", "artifacts", "docs", "notebooks"
        ]:
            self.path(rel).mkdir(parents=True, exist_ok=True)

def _load_yaml() -> Dict[str, Any]:
    path = ROOT_DIR / "configs" / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _apply_env(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply only safe, documented environment overrides."""
    mapping = {
        ("app","environment"): "APP_ENV",
        ("app","log_level"): "LOG_LEVEL",
        ("database","url"): "DATABASE_URL",
        ("llm","provider"): "LLM_PROVIDER",
        ("llm","model_name"): "LLM_MODEL",
        ("api","host"): "API_HOST",
        ("api","port"): "API_PORT",
        ("dashboard","port"): "STREAMLIT_PORT",
    }
    for (section,key), env_name in mapping.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            current = data[section][key]
            if isinstance(current, bool):
                data[section][key] = value.lower() in {"1","true","yes"}
            elif isinstance(current, int):
                data[section][key] = int(value)
            else:
                data[section][key] = value
    return data

_settings: Settings | None = None

def get_settings() -> Settings:
    """Return the cached typed application settings."""
    global _settings
    if _settings is None:
        _settings = Settings.model_validate(_apply_env(_load_yaml()))
        _settings.ensure_directories()
    return _settings
