import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./docs_qa.db"
    qdrant_storage_path: str = "./qdrant_storage"
    qdrant_url: str | None = None
    gemini_api_key: str = ""
    openai_api_key: str = ""
    port: int = 8000
    host: str = "0.0.0.0"
    backend_url: str = "http://localhost:8000"
    confidence_threshold: float = 0.35
    llm_temperature: float = 0.5
    docs_seed_path: str = "starter_kit/data/docs.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
