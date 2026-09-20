# ### FILE: app/models/entities.py
"""
SQLAlchemy 2.0 ORM Entity Definitions for Smart Notes.
Provides complete relational schema mapping with foreign key cascades,
enum checks, and eager/lazy relationship declarations.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from sqlalchemy import (
    String,
    Text,
    Integer,
    BigInteger,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base


class DocumentStatus(str, Enum):
    """Lifecycle status of a document."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PageStatus(str, Enum):
    """Processing stage status of an individual page."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PREPROCESSED = "PREPROCESSED"
    SEGMENTED = "SEGMENTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExportFormat(str, Enum):
    """Permitted document export target formats."""
    MARKDOWN = "MARKDOWN"
    TXT = "TXT"
    LATEX = "LATEX"
    JSON = "JSON"

    @classmethod
    def _missing_(cls, value: object):
        """Allow case-insensitive resolution of export formats."""
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return None


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """User account entity."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Document(Base):
    """Root document container aggregating uploaded pages and processed notes."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(64), default="default", nullable=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), default="image/jpeg", nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=DocumentStatus.PENDING.value, nullable=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="documents")
    pages: Mapped[List["Page"]] = relationship(
        "Page", back_populates="document", cascade="all, delete-orphan", order_by="Page.page_number", lazy="selectin"
    )
    exports: Mapped[List["ExportDocument"]] = relationship(
        "ExportDocument", back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )


class Page(Base):
    """Represents a single physical page or photograph belonging to a document."""
    __tablename__ = "pages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    raw_image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    processed_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    deskewed_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    debug_dir_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    width: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skew_angle: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=PageStatus.PENDING.value, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships & Constraints
    document: Mapped["Document"] = relationship("Document", back_populates="pages")
    lines: Mapped[List["TextLine"]] = relationship(
        "TextLine", back_populates="page", cascade="all, delete-orphan", order_by="TextLine.line_index", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page"),
    )


class TextLine(Base):
    """An individual segmented line bounding box with OCR transcription."""
    __tablename__ = "text_lines"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    line_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    cropped_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    original_raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recognized_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_header: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_math: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    indent_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships & Constraints
    page: Mapped["Page"] = relationship("Page", back_populates="lines")

    __table_args__ = (
        UniqueConstraint("page_id", "line_index", name="uq_page_line_index"),
        Index("idx_text_lines_page_index", "page_id", "line_index"),
    )


class ExportDocument(Base):
    """Structured text export generated from recognized document lines."""
    __tablename__ = "export_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    export_format: Mapped[str] = mapped_column(
        String(32), default=ExportFormat.MARKDOWN.value, nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="exports")

    __table_args__ = (
        Index("idx_export_documents_format", "document_id", "export_format"),
    )
