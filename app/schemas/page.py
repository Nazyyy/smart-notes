# ### FILE: app/schemas/page.py
"""
Pydantic Schemas for Document Page Records and Processing Artifacts.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.models.entities import PageStatus
from app.schemas.line import TextLineRead


class PageBase(BaseModel):
    """Base fields for page entities."""
    page_number: int = Field(ge=1, default=1, description="1-indexed sequence number")
    width: int = Field(ge=0, default=0)
    height: int = Field(ge=0, default=0)
    skew_angle: float = Field(default=0.0, description="Calculated skew angle in degrees")
    status: PageStatus = PageStatus.PENDING


class PageCreate(PageBase):
    """Payload for creating a page record."""
    document_id: UUID
    raw_image_path: str


class PageRead(PageBase):
    """Full representation of a page returned from API."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    raw_image_path: str
    processed_image_path: Optional[str] = None
    deskewed_image_path: Optional[str] = None
    debug_dir_path: Optional[str] = None
    created_at: datetime
    lines: List[TextLineRead] = Field(default_factory=list)


class PageDebugArtifacts(BaseModel):
    """Urls or relative paths to debug images produced during CV pipeline."""
    raw_image_url: str
    deskewed_image_url: Optional[str] = None
    shadow_removed_url: Optional[str] = None
    binarized_url: Optional[str] = None
    projection_profile_url: Optional[str] = None
    segmented_lines_overlay_url: Optional[str] = None
