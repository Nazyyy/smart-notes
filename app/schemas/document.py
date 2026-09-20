# ### FILE: app/schemas/document.py
"""
Pydantic Schemas for Root Document Management and Lifecycle Tracking.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.models.entities import DocumentStatus
from app.schemas.page import PageRead


class DocumentBase(BaseModel):
    """Base fields for documents."""
    title: str = Field(min_length=1, max_length=255, description="Human-readable title")
    description: Optional[str] = Field(default=None, description="Optional notes or context")
    author: Optional[str] = Field(default="default", description="Author identifier or username")


class DocumentCreate(DocumentBase):
    """Payload for creating a document record."""
    user_id: Optional[UUID] = None
    original_filename: str
    file_path: str
    file_size_bytes: int = Field(ge=0)
    mime_type: str = "image/jpeg"


class DocumentUpdate(BaseModel):
    """Fields allowed when updating document metadata or status."""
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[DocumentStatus] = None
    error_message: Optional[str] = None
    author: Optional[str] = None


class DocumentRead(DocumentBase):
    """Document response representation including page summary."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: Optional[UUID] = None
    author: Optional[str] = "default"
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    pages: List[PageRead] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """Paginated or listed documents response."""
    items: List[DocumentRead]
    total_count: int
