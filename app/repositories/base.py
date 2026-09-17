# ### FILE: app/repositories/base.py
"""
Generic Asynchronous Base Repository for SQLAlchemy 2.0 Entities.
Implements standard CRUD operations with connection isolation and error handling.
"""

from typing import Generic, TypeVar, Type, Optional, List, Any
from uuid import UUID
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import Base
from app.core.exceptions import DatabaseOperationException
from app.core.logging import get_logger

logger = get_logger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic repository providing thread-safe async CRUD access."""

    def __init__(self, model: Type[ModelType], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get_by_id(self, entity_id: UUID) -> Optional[ModelType]:
        """Fetch an entity by its primary key UUID."""
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == entity_id)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed to fetch %s by id %s: %s", self.model.__name__, entity_id, exc)
            raise DatabaseOperationException(
                operation=f"get_by_id on {self.model.__name__}",
                reason=str(exc),
            ) from exc

    async def list_all(self, offset: int = 0, limit: int = 100) -> List[ModelType]:
        """List entities with pagination."""
        try:
            result = await self.session.execute(
                select(self.model).offset(offset).limit(limit)
            )
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Failed to list %s: %s", self.model.__name__, exc)
            raise DatabaseOperationException(
                operation=f"list_all on {self.model.__name__}",
                reason=str(exc),
            ) from exc

    async def count(self) -> int:
        """Count total entities in table."""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id))
            )
            return int(result.scalar_one() or 0)
        except Exception as exc:
            logger.error("Failed to count %s: %s", self.model.__name__, exc)
            raise DatabaseOperationException(
                operation=f"count on {self.model.__name__}",
                reason=str(exc),
            ) from exc

    async def create(self, entity: ModelType) -> ModelType:
        """Add and persist a new entity."""
        try:
            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)
            return entity
        except Exception as exc:
            logger.error("Failed to create %s: %s", self.model.__name__, exc)
            raise DatabaseOperationException(
                operation=f"create on {self.model.__name__}",
                reason=str(exc),
            ) from exc

    async def update(self, entity_id: UUID, values: dict[str, Any]) -> Optional[ModelType]:
        """Update fields on an entity by ID."""
        try:
            stmt = (
                update(self.model)
                .where(self.model.id == entity_id)
                .values(**values)
                .execution_options(synchronize_session="fetch")
            )
            await self.session.execute(stmt)
            await self.session.flush()
            return await self.get_by_id(entity_id)
        except Exception as exc:
            logger.error("Failed to update %s id %s: %s", self.model.__name__, entity_id, exc)
            raise DatabaseOperationException(
                operation=f"update on {self.model.__name__}",
                reason=str(exc),
            ) from exc

    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by ID."""
        try:
            stmt = delete(self.model).where(self.model.id == entity_id)
            result = await self.session.execute(stmt)
            await self.session.commit()
            return (result.rowcount or 0) > 0
        except Exception as exc:
            logger.error("Failed to delete %s id %s: %s", self.model.__name__, entity_id, exc)
            raise DatabaseOperationException(
                operation=f"delete on {self.model.__name__}",
                reason=str(exc),
            ) from exc
