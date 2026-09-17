# ### FILE: app/schemas/line.py
"""
Pydantic Schemas for Bounding Box Coordinates and Text Lines.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoundingBox(BaseModel):
    """Geometric coordinates and dimensions of a detected text line."""
    x: int = Field(ge=0, description="Top-left X pixel coordinate")
    y: int = Field(ge=0, description="Top-left Y pixel coordinate")
    w: int = Field(gt=0, description="Bounding box width in pixels")
    h: int = Field(gt=0, description="Bounding box height in pixels")


class TextLineBase(BaseModel):
    """Base fields shared by text line representations."""
    line_index: int = Field(ge=0, description="Index of line in page order")
    bbox_x: int = Field(ge=0)
    bbox_y: int = Field(ge=0)
    bbox_w: int = Field(gt=0)
    bbox_h: int = Field(gt=0)
    recognized_text: str = Field(default="", description="Decoded transcription text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="OCR confidence score")
    is_header: bool = Field(default=False, description="Header classification flag")
    is_math: bool = Field(default=False, description="Formula/math notation flag")
    indent_level: int = Field(default=0, ge=0, description="Structural indentation level")


class TextLineCreate(TextLineBase):
    """Schema for persisting a newly segmented line."""
    page_id: UUID
    cropped_image_path: Optional[str] = None


class TextLineUpdate(BaseModel):
    """Schema for updating a recognized line (user manual edits)."""
    recognized_text: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_header: Optional[bool] = None
    is_math: Optional[bool] = None
    indent_level: Optional[int] = Field(default=None, ge=0)


class TextLineRead(TextLineBase):
    """Schema returned by API representing a text line."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    page_id: UUID
    cropped_image_path: Optional[str] = None
    created_at: datetime
