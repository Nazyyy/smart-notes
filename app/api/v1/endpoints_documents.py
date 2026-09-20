# ### FILE: app/api/v1/endpoints_documents.py
"""
Document Endpoints: Upload, Listing, Details, and Deletion.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, File, Form, UploadFile, Query, BackgroundTasks, status
from app.api.dependencies import get_document_service
from app.schemas.document import DocumentRead, DocumentListResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload note image and trigger recognition",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Photograph or scan of handwritten note page"),
    title: str = Form(..., min_length=1, max_length=255, description="Document title"),
    author: Optional[str] = Form("default", description="Author or profile ID for this note"),
    description: Optional[str] = Form(None, description="Optional document description"),
    process_immediately: bool = Form(True, description="Immediately trigger CV and ML pipeline"),
    async_background: bool = Form(False, description="Run processing in background task"),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentRead:
    """
    Ingest a handwritten document photo, validate format and size limits,
    persist records, and trigger recognition pipeline.
    """
    document = await doc_service.create_document_from_upload(
        title=title,
        description=description,
        upload_file=file,
        author=author,
        background_tasks=background_tasks if async_background else None,
        process_immediately=process_immediately,
    )
    return DocumentRead.model_validate(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents(
    limit: int = Query(20, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    author: Optional[str] = Query(None, description="Filter documents by author"),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List documents ordered by creation date descending, optionally filtered by author."""
    items, total = await doc_service.list_documents(limit=limit, offset=offset, author=author)
    return DocumentListResponse(
        items=[DocumentRead.model_validate(item) for item in items],
        total_count=total,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get document details by ID",
)
async def get_document_details(
    document_id: UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentRead:
    """Fetch complete document metadata including pages and recognized text lines."""
    document = await doc_service.get_document(document_id)
    return DocumentRead.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document and files",
)
async def delete_document(
    document_id: UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> None:
    """Permanently delete a document, its database entries, and media storage files."""
    await doc_service.delete_document(document_id)
