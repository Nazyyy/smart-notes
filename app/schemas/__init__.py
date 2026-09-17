# ### FILE: app/schemas/__init__.py
"""
Pydantic Data Transfer Objects (DTO) and Schema Contracts.
"""

from app.schemas.line import (
    BoundingBox,
    TextLineBase,
    TextLineCreate,
    TextLineUpdate,
    TextLineRead,
)
from app.schemas.page import (
    PageBase,
    PageCreate,
    PageRead,
    PageDebugArtifacts,
)
from app.schemas.document import (
    DocumentBase,
    DocumentCreate,
    DocumentUpdate,
    DocumentRead,
    DocumentListResponse,
)
from app.schemas.recognition import (
    RecognitionTriggerRequest,
    RecognitionProgressResponse,
    PageRecognitionResult,
)
from app.schemas.export import (
    ExportRequest,
    ExportDocumentRead,
    StructuredNoteBlock,
)

__all__ = [
    "BoundingBox",
    "TextLineBase",
    "TextLineCreate",
    "TextLineUpdate",
    "TextLineRead",
    "PageBase",
    "PageCreate",
    "PageRead",
    "PageDebugArtifacts",
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentRead",
    "DocumentListResponse",
    "RecognitionTriggerRequest",
    "RecognitionProgressResponse",
    "PageRecognitionResult",
    "ExportRequest",
    "ExportDocumentRead",
    "StructuredNoteBlock",
]
