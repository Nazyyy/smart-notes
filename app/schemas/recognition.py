# ### FILE: app/schemas/recognition.py
"""
Pydantic Schemas for Recognition Tasks and Inference Results.
"""

from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from app.schemas.line import TextLineRead


class RecognitionTriggerRequest(BaseModel):
    """Parameters for triggering or re-triggering ML OCR pipeline on a document or page."""
    beam_width: Optional[int] = Field(default=5, ge=1, le=20)
    apply_deskew: bool = True
    apply_shadow_removal: bool = True


class RecognitionProgressResponse(BaseModel):
    """Current processing state of a recognition pipeline."""
    document_id: UUID
    page_id: Optional[UUID] = None
    status: str
    processed_lines: int
    total_lines: int
    estimated_progress_pct: float = Field(ge=0.0, le=100.0)


class PageRecognitionResult(BaseModel):
    """Complete page OCR result with lines and raw structured text."""
    page_id: UUID
    page_number: int
    skew_angle: float
    total_lines: int
    lines: List[TextLineRead]
    combined_raw_text: str
