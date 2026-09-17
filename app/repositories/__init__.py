# ### FILE: app/repositories/__init__.py
"""
Data Access Layer (Repository Pattern).
"""

from app.repositories.base import BaseRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository

__all__ = [
    "BaseRepository",
    "DocumentRepository",
    "PageRepository",
]
