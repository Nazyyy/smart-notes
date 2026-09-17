# ### FILE: app/api/v1/endpoints_recognition.py
"""
Recognition Pipeline Trigger and Status Monitoring Endpoints.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, BackgroundTasks, Body
from app.api.dependencies import (
    get_document_service,
    get_pipeline,
    get_document_repository,
    get_page_repository,
)
from app.core.exceptions import PageNotFoundException
from app.models.entities import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.schemas.recognition import RecognitionProgressResponse, RecognitionTriggerRequest
from app.services.document_service import DocumentService
from app.services.processing_pipeline import DocumentProcessingPipeline

router = APIRouter(prefix="/recognition", tags=["Recognition"])


@router.post(
    "/pages/{page_id}/reprocess",
    summary="Trigger reprocessing of a specific page",
)
async def reprocess_page(
    page_id: UUID,
    background_tasks: BackgroundTasks,
    request_params: RecognitionTriggerRequest = Body(default_factory=RecognitionTriggerRequest),
    pipeline: DocumentProcessingPipeline = Depends(get_pipeline),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    page_repo: PageRepository = Depends(get_page_repository),
) -> dict:
    """Manually re-run the optical rectification, segmentation, and recognition pipeline."""
    page = await page_repo.get_by_id(page_id)
    if not page:
        raise PageNotFoundException(page_id)

    background_tasks.add_task(
        pipeline.execute_page_pipeline,
        document_id=page.document_id,
        page_id=page.id,
        doc_repo=doc_repo,
        page_repo=page_repo,
        beam_width=request_params.beam_width or 5,
    )

    return {
        "status": "QUEUED",
        "message": f"Reprocessing scheduled for page {page_id}.",
        "document_id": str(page.document_id),
        "page_id": str(page.id),
    }


@router.get(
    "/documents/{document_id}/status",
    response_model=RecognitionProgressResponse,
    summary="Get processing status and line metrics for document",
)
async def get_recognition_status(
    document_id: UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> RecognitionProgressResponse:
    """Check processing lifecycle state and line metrics for progress polling."""
    doc = await doc_service.get_document(document_id)

    total_lines = 0
    recognized_lines = 0

    for page in doc.pages:
        total_lines += len(page.lines)
        recognized_lines += sum(1 for line in page.lines if line.recognized_text)

    if doc.status == DocumentStatus.COMPLETED.value:
        progress = 100.0
    elif doc.status == DocumentStatus.PROCESSING.value:
        progress = 50.0 if total_lines == 0 else (recognized_lines / float(total_lines)) * 100.0
    elif doc.status == DocumentStatus.FAILED.value:
        progress = 0.0
    else:
        progress = 10.0

    return RecognitionProgressResponse(
        document_id=doc.id,
        status=doc.status,
        processed_lines=recognized_lines,
        total_lines=total_lines,
        estimated_progress_pct=min(100.0, max(0.0, progress)),
    )
