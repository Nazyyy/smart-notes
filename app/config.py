# ### FILE: app/config.py
"""
Application Configuration and Environment Settings.
Uses Pydantic Settings for strict typing, defaults, and validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Core System Information
    PROJECT_NAME: str = "Smart Notes - Intelligent Handwritten Notes Recognition"
    PROJECT_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Current execution environment")
    DEBUG: bool = Field(default=True, description="Enable debug logging and detailed errors")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    # API Network Binding
    HOST: str = Field(default="0.0.0.0", description="API listener IP address")
    PORT: int = Field(default=8000, description="API listener port")
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # Database Configuration
    POSTGRES_USER: str = "smart_notes_user"
    POSTGRES_PASSWORD: str = "smart_notes_secret_password_2026"
    POSTGRES_DB: str = "smart_notes_db"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://smart_notes_user:smart_notes_secret_password_2026@postgres:5432/smart_notes_db",
        description="Async SQLAlchemy database connection URI"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 1800

    # Storage Subsystem Configuration
    STORAGE_ROOT: Path = Field(
        default=Path("./data/storage"),
        description="Root directory for storing images and artifacts"
    )
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=26_214_400,  # 25 MB
        description="Maximum permitted payload size for file uploads"
    )
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [
        "jpg", "jpeg", "png", "webp", "bmp", "tiff"
    ]
    MAX_IMAGE_DIMENSION: int = Field(
        default=4096,
        description="Maximum image width or height before automatic downscaling"
    )

    # Machine Learning Subsystem Configuration
    ML_MODEL_WEIGHTS_PATH: Path = Field(
        default=Path("./data/weights/crnn_htr_weights.pt"),
        description="Path to serialized PyTorch CRNN weights"
    )
    ML_USE_TRANSFORMER: bool = Field(
        default=True,
        description="Enable state-of-the-art TrOCR Vision Transformer for handwriting recognition"
    )
    ML_TRANSFORMER_PATH: Path = Field(
        default=Path("./data/weights/trocr_ru_lines") if Path("./data/weights/trocr_ru_lines").exists() else Path("./data/weights/trocr_ru"),
        description="Path to local directory with serialized TrOCR model"
    )
    ML_DEVICE: str = Field(
        default="cuda",
        description="Inference execution device ('cpu' or 'cuda')"
    )
    ML_BATCH_SIZE: int = Field(
        default=16,
        description="Batch size for parallel text line tensor inference"
    )
    ML_BEAM_WIDTH: int = Field(
        default=5,
        description="Beam search decoder width for CTC sequence inference"
    )
    ML_IMAGE_HEIGHT: int = 32
    ML_IMAGE_WIDTH: int = 256
    ML_BLANK_INDEX: int = 0

    @field_validator("STORAGE_ROOT", "ML_MODEL_WEIGHTS_PATH", "ML_TRANSFORMER_PATH", mode="after")
    @classmethod
    def resolve_paths(cls, path_value: Path) -> Path:
        """Resolve path to absolute location and ensure parent directory exists."""
        resolved = path_value.resolve()
        return resolved


@lru_cache()
def get_settings() -> Settings:
    """Cached accessor for singleton application settings."""
    return Settings()
