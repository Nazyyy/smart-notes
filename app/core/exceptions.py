# ### FILE: app/core/exceptions.py
"""
Domain Exceptions Hierarchy for Smart Notes Platform.
Ensures fail-fast semantics and uniform error response mapping.
"""

from typing import Any, Dict, Optional


class SmartNotesBaseException(Exception):
    """Base exception for all domain-specific errors in Smart Notes."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured error response representation."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class DocumentNotFoundException(SmartNotesBaseException):
    """Raised when a requested document ID does not exist in the database."""

    def __init__(self, document_id: Any) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' was not found.",
            status_code=404,
            details={"document_id": str(document_id)},
        )


class PageNotFoundException(SmartNotesBaseException):
    """Raised when a requested page ID does not exist."""

    def __init__(self, page_id: Any) -> None:
        super().__init__(
            message=f"Page with ID '{page_id}' was not found.",
            status_code=404,
            details={"page_id": str(page_id)},
        )


class InvalidFilePayloadException(SmartNotesBaseException):
    """Raised when an uploaded file fails validation checks (extension, size, format)."""

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Invalid file payload: {reason}",
            status_code=400,
            details=details or {},
        )


class ImageProcessingException(SmartNotesBaseException):
    """Raised when OpenCV fails to load, process, deskew, or binarize an image."""

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(
            message=f"Computer Vision processing failed at step '{step}': {reason}",
            status_code=422,
            details={"step": step, "reason": reason},
        )


class ModelInferenceException(SmartNotesBaseException):
    """Raised when PyTorch neural network inference or CTC decoding encounters a fault."""

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"ML OCR inference error: {reason}",
            status_code=500,
            details=details or {},
        )


class StorageException(SmartNotesBaseException):
    """Raised when file system or media storage I/O fails."""

    def __init__(self, operation: str, path: str, reason: str) -> None:
        super().__init__(
            message=f"Storage I/O failure during '{operation}' on path '{path}': {reason}",
            status_code=500,
            details={"operation": operation, "path": path, "reason": reason},
        )


class DatabaseOperationException(SmartNotesBaseException):
    """Raised when an unexpected database transaction or integrity violation occurs."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Database failure during '{operation}': {reason}",
            status_code=500,
            details={"operation": operation, "reason": reason},
        )
