# ### FILE: app/api/v1/endpoints_pages.py
"""
Page Endpoints: Inspection, Debug Visualizations, and Line Transcription Edits.
"""

from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, Depends, Body, status
from app.api.dependencies import get_page_repository, get_storage_service
from app.core.exceptions import PageNotFoundException
from app.repositories.page_repository import PageRepository
from app.schemas.page import PageRead, PageDebugArtifacts
from app.schemas.line import TextLineRead, TextLineUpdate
from app.services.storage_service import StorageService

router = APIRouter(prefix="/pages", tags=["Pages"])


@router.get(
    "/{page_id}",
    response_model=PageRead,
    summary="Get page details and recognized lines",
)
async def get_page_details(
    page_id: UUID,
    page_repo: PageRepository = Depends(get_page_repository),
) -> PageRead:
    """Retrieve full details of a specific page including its segmented lines."""
    page = await page_repo.get_page_with_lines(page_id)
    if not page:
        raise PageNotFoundException(page_id)
    return PageRead.model_validate(page)


@router.get(
    "/{page_id}/debug-artifacts",
    response_model=PageDebugArtifacts,
    summary="Get paths/URLs to intermediate CV debug artifacts",
)
async def get_page_debug_artifacts(
    page_id: UUID,
    page_repo: PageRepository = Depends(get_page_repository),
    storage: StorageService = Depends(get_storage_service),
) -> PageDebugArtifacts:
    """Retrieve relative URLs to stage images (raw, rectified, binarized, HPP, overlay)."""
    page = await page_repo.get_by_id(page_id)
    if not page:
        raise PageNotFoundException(page_id)

    debug_dir = storage.get_page_debug_directory(page.document_id, page.id)
    base_prefix = f"/static/documents/{page.document_id}/pages/{page.id}"

    return PageDebugArtifacts(
        raw_image_url=f"/static/documents/{page.document_id}/raw_source.jpg",
        deskewed_image_url=f"{base_prefix}/02_rectified.jpg",
        shadow_removed_url=f"{base_prefix}/03_shadow_suppressed.jpg",
        binarized_url=f"{base_prefix}/04_binarized.png",
        projection_profile_url=f"{base_prefix}/05_projection_profile.png",
        segmented_lines_overlay_url=f"{base_prefix}/06_segmented_overlay.jpg",
    )


@router.put(
    "/{page_id}/lines/{line_id}",
    response_model=TextLineRead,
    summary="Update line transcription (Manual Correction)",
)
async def update_line_transcription(
    page_id: UUID,
    line_id: UUID,
    payload: TextLineUpdate = Body(...),
    page_repo: PageRepository = Depends(get_page_repository),
) -> TextLineRead:
    """Update recognized text transcription or confidence for an individual line."""
    if payload.recognized_text is None:
        raise ValueError("recognized_text is required for updating a text line.")

    updated_line = await page_repo.update_line_text(
        line_id=line_id,
        new_text=payload.recognized_text,
        confidence=payload.confidence,
    )
    if not updated_line:
        raise PageNotFoundException(line_id)

    return TextLineRead.model_validate(updated_line)
