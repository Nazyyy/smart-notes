# ### FILE: app/schemas/export.py
"""
Pydantic Schemas for Document Structuring and Multi-Format Exports.
"""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.models.entities import ExportFormat


class ExportRequest(BaseModel):
    """Payload specifying export format parameters."""
    export_format: ExportFormat = ExportFormat.MARKDOWN
    include_math_delimiters: bool = True
    synthesize_headers: bool = True


class ExportDocumentRead(BaseModel):
    """Schema returned after generating a structured export."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    export_format: ExportFormat
    file_path: str
    content: str
    created_at: datetime


class StructuredNoteBlock(BaseModel):
    """Semantic block parsed from recognized lines (heading, list, math, paragraph)."""
    block_type: str = Field(description="heading_1, heading_2, list_item, math_block, paragraph")
    content: str
    indent_level: int = 0
