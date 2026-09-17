# ### FILE: app/repositories/page_repository.py
"""
Page and TextLine Repository with transaction-isolated bulk line persistence.
"""

from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, delete, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.models.entities import Page, TextLine, PageStatus
from app.core.exceptions import DatabaseOperationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class PageRepository(BaseRepository[Page]):
    """Repository managing Page records and associated TextLines."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Page, session)

    async def get_page_with_lines(self, page_id: UUID) -> Optional[Page]:
        """Fetch a page with its text lines ordered by line_index."""
        try:
            stmt = (
                select(Page)
                .where(Page.id == page_id)
                .options(selectinload(Page.lines))
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Error loading page %s with lines: %s", page_id, exc)
            raise DatabaseOperationException(
                operation="get_page_with_lines",
                reason=str(exc),
            ) from exc

    async def get_pages_by_document(self, document_id: UUID) -> List[Page]:
        """List all pages belonging to a document ordered by page_number."""
        try:
            stmt = (
                select(Page)
                .where(Page.document_id == document_id)
                .options(selectinload(Page.lines))
                .order_by(Page.page_number)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Error listing pages for document %s: %s", document_id, exc)
            raise DatabaseOperationException(
                operation="get_pages_by_document",
                reason=str(exc),
            ) from exc

    async def replace_page_lines(
        self,
        page_id: UUID,
        lines: List[TextLine],
    ) -> List[TextLine]:
        """
        Atomically delete old segmented lines for a page and insert updated lines.
        Ensures idempotency during pipeline re-runs.
        """
        try:
            # Delete existing lines for this page
            del_stmt = delete(TextLine).where(TextLine.page_id == page_id)
            await self.session.execute(del_stmt)

            # Insert new lines
            for line in lines:
                line.page_id = page_id
                self.session.add(line)

            await self.session.commit()


            # Return newly inserted lines in order
            stmt = (
                select(TextLine)
                .where(TextLine.page_id == page_id)
                .order_by(TextLine.line_index)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Error replacing lines for page %s: %s", page_id, exc)
            raise DatabaseOperationException(
                operation="replace_page_lines",
                reason=str(exc),
            ) from exc

    async def update_line_text(
        self,
        line_id: UUID,
        new_text: str,
        confidence: Optional[float] = None,
    ) -> Optional[TextLine]:
        """Update the transcription of an individual text line."""
        try:
            stmt = select(TextLine).where(TextLine.id == line_id)
            result = await self.session.execute(stmt)
            line = result.scalar_one_or_none()
            if not line:
                return None

            line.recognized_text = new_text
            if confidence is not None:
                line.confidence = confidence

            await self.session.flush()
            await self.session.refresh(line)
            return line
        except Exception as exc:
            logger.error("Error updating text for line %s: %s", line_id, exc)
            raise DatabaseOperationException(
                operation="update_line_text",
                reason=str(exc),
            ) from exc
