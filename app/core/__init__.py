# ### FILE: app/core/__init__.py
"""
Core utilities: logging, exception hierarchy, security, and telemetry.
"""

from app.core.exceptions import (
    SmartNotesBaseException,
    DocumentNotFoundException,
    PageNotFoundException,
    InvalidFilePayloadException,
    ImageProcessingException,
    ModelInferenceException,
    StorageException,
    DatabaseOperationException,
)
from app.core.logging import setup_logging, get_logger

__all__ = [
    "SmartNotesBaseException",
    "DocumentNotFoundException",
    "PageNotFoundException",
    "InvalidFilePayloadException",
    "ImageProcessingException",
    "ModelInferenceException",
    "StorageException",
    "DatabaseOperationException",
    "setup_logging",
    "get_logger",
]
