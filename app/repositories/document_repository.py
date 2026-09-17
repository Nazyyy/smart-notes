# ### FILE: app/repositories/document_repository.py
"""
Document Repository handling aggregate document queries, pages, and exports.
"""

from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.models.entities import Document, Page, TextLine, ExportDocument, DocumentStatus
from app.core.exceptions import DatabaseOperationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentRepository(BaseRepository[Document]):
    """Specialized repository for Document entity aggregate queries."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Document, session)

    async def get_document_with_details(self, document_id: UUID) -> Optional[Document]:
        """Fetch document with all its pages and segmented text lines eagerly loaded."""
        try:
            self.session.expire_all()
            stmt = (
                select(Document)
                .where(Document.id == document_id)
                .options(
                    selectinload(Document.pages).selectinload(Page.lines),
                    selectinload(Document.exports),
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Error loading document details for %s: %s", document_id, exc)
            raise DatabaseOperationException(
                operation="get_document_with_details",
                reason=str(exc),
            ) from exc

    async def list_recent(self, limit: int = 50, offset: int = 0) -> List[Document]:
        """List documents sorted by creation date descending."""
        try:
            stmt = (
                select(Document)
                .options(selectinload(Document.pages))
                .order_by(desc(Document.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Error listing recent documents: %s", exc)
            raise DatabaseOperationException(
                operation="list_recent",
                reason=str(exc),
            ) from exc

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        """Update document processing status and optional failure diagnostics."""
        values: dict = {"status": status.value}
        if error_message is not None:
            values["error_message"] = error_message
        return await self.update(document_id, values)

    async def save_export(
        self,
        document_id: UUID,
        export_format: str,
        file_path: str,
        content: str,
    ) -> ExportDocument:
        """Persist structured export result for a document."""
        try:
            export_entity = ExportDocument(
                document_id=document_id,
                export_format=export_format,
                file_path=file_path,
                content=content,
            )
            self.session.add(export_entity)
            await self.session.flush()
            await self.session.refresh(export_entity)
            return export_entity
        except Exception as exc:
            logger.error("Failed saving export for document %s: %s", document_id, exc)
            raise DatabaseOperationException(
                operation="save_export",
                reason=str(exc),
            ) from exc

    async def get_export(
        self,
        document_id: UUID,
        export_format: str,
    ) -> Optional[ExportDocument]:
        """Retrieve most recent export for document and format."""
        try:
            stmt = (
                select(ExportDocument)
                .where(
                    ExportDocument.document_id == document_id,
                    ExportDocument.export_format == export_format,
                )
                .order_by(desc(ExportDocument.created_at))
                .limit(1)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed fetching export for %s: %s", document_id, exc)
            raise DatabaseOperationException(
                operation="get_export",
                reason=str(exc),
            ) from exc
