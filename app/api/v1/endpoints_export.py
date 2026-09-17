# ### FILE: app/api/v1/endpoints_export.py
"""
Document Export Endpoints: Markdown, Plaintext, LaTeX, and JSON formatting.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, Body, Query, Response, status
from fastapi.responses import PlainTextResponse
from app.api.dependencies import get_document_service, get_document_repository
from app.models.entities import ExportFormat
from app.repositories.document_repository import DocumentRepository
from app.schemas.export import ExportRequest, ExportDocumentRead
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Export"])


@router.post(
    "/{document_id}/export",
    response_model=ExportDocumentRead,
    summary="Generate structured note export",
)
async def generate_document_export(
    document_id: UUID,
    payload: ExportRequest = Body(...),
    doc_service: DocumentService = Depends(get_document_service),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> ExportDocumentRead:
    """Generate and persist structured note export according to format specified."""
    content = await doc_service.export_document(
        document_id=document_id,
        export_format=payload.export_format,
    )
    latest_export = await doc_repo.get_export(
        document_id=document_id,
        export_format=payload.export_format.value,
    )
    if not latest_export:
        raise ValueError("Failed to retrieve generated export record.")

    return ExportDocumentRead.model_validate(latest_export)


@router.get(
    "/{document_id}/export/{export_format}/download",
    summary="Download exported document file",
)
async def download_document_export(
    document_id: UUID,
    export_format: ExportFormat,
    doc_service: DocumentService = Depends(get_document_service),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> PlainTextResponse:
    """Download the raw structured Markdown, TXT, LaTeX, or JSON export file."""
    export_record = await doc_repo.get_export(
        document_id=document_id,
        export_format=export_format.value,
    )

    if not export_record:
        # Generate on the fly
        content = await doc_service.export_document(document_id, export_format)
    else:
        content = export_record.content

    media_types = {
        ExportFormat.MARKDOWN: "text/markdown; charset=utf-8",
        ExportFormat.TXT: "text/plain; charset=utf-8",
        ExportFormat.LATEX: "application/x-latex; charset=utf-8",
        ExportFormat.JSON: "application/json; charset=utf-8",
    }

    ext_map = {
        ExportFormat.MARKDOWN: "md",
        ExportFormat.TXT: "txt",
        ExportFormat.LATEX: "tex",
        ExportFormat.JSON: "json",
    }

    filename = f"smart_notes_{document_id}.{ext_map.get(export_format, 'txt')}"

    return PlainTextResponse(
        content=content,
        media_type=media_types.get(export_format, "text/plain"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
