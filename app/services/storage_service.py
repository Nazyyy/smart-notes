# ### FILE: app/services/storage_service.py
"""
Local and Media Storage Service.
Manages disk hierarchies, streaming chunked file writes, and secure path resolution.
"""

from pathlib import Path
from typing import AsyncIterator, BinaryIO
from uuid import UUID
import aiofiles
from fastapi import UploadFile

from app.config import get_settings
from app.core.exceptions import InvalidFilePayloadException, StorageException
from app.core.security import sanitize_filename, validate_file_extension, verify_path_within_root
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class StorageService:
    """Manages disk persistence of documents, pages, crops, and export artifacts."""

    def __init__(self, root_dir: Path = settings.STORAGE_ROOT) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_document_directory(self, document_id: UUID) -> Path:
        """Return and ensure directory path for a specific document."""
        doc_dir = self.root_dir / "documents" / str(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return verify_path_within_root(doc_dir, self.root_dir)

    def get_page_debug_directory(self, document_id: UUID, page_id: UUID) -> Path:
        """Return and ensure directory path for page debug artifacts."""
        page_dir = self.get_document_directory(document_id) / "pages" / str(page_id)
        page_dir.mkdir(parents=True, exist_ok=True)
        return verify_path_within_root(page_dir, self.root_dir)

    def get_export_directory(self, document_id: UUID) -> Path:
        """Return and ensure directory path for document exports."""
        export_dir = self.get_document_directory(document_id) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        return verify_path_within_root(export_dir, self.root_dir)

    async def save_uploaded_file(
        self,
        document_id: UUID,
        upload_file: UploadFile,
    ) -> Path:
        """
        Stream an uploaded file to disk in 64KB chunks to prevent RAM exhaustion (DoS/OOM).
        Validates maximum allowed size and extension.
        """
        filename = sanitize_filename(upload_file.filename or "upload.jpg")
        ext = validate_file_extension(filename, settings.ALLOWED_IMAGE_EXTENSIONS)

        doc_dir = self.get_document_directory(document_id)
        destination_path = doc_dir / f"raw_source.{ext}"

        bytes_written = 0
        chunk_size = 64 * 1024  # 64 KB

        try:
            async with aiofiles.open(destination_path, "wb") as out_file:
                while True:
                    chunk = await upload_file.read(chunk_size)
                    if not chunk:
                        break

                    bytes_written += len(chunk)
                    if bytes_written > settings.MAX_UPLOAD_SIZE_BYTES:
                        # Clean up partial file on violation
                        if destination_path.exists():
                            destination_path.unlink()
                        raise InvalidFilePayloadException(
                            reason=(
                                f"File exceeds maximum allowed size of "
                                f"{settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024):.1f} MB."
                            ),
                            details={"bytes_received": bytes_written},
                        )

                    await out_file.write(chunk)

            logger.info("Saved upload (%d bytes) to: %s", bytes_written, destination_path)
            return destination_path

        except InvalidFilePayloadException:
            raise
        except Exception as exc:
            logger.error("Failed streaming upload to %s: %s", destination_path, exc)
            if destination_path.exists():
                destination_path.unlink()
            raise StorageException(
                operation="save_uploaded_file",
                path=str(destination_path),
                reason=str(exc),
            ) from exc

    async def write_text_file(self, target_path: Path, content: str) -> Path:
        """Write text string asynchronously to specified path."""
        verified_path = verify_path_within_root(target_path, self.root_dir)
        verified_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(verified_path, "w", encoding="utf-8") as f:
                await f.write(content)
            return verified_path
        except Exception as exc:
            logger.error("Failed writing text file %s: %s", verified_path, exc)
            raise StorageException(
                operation="write_text_file",
                path=str(verified_path),
                reason=str(exc),
            ) from exc

    async def read_text_file(self, target_path: Path) -> str:
        """Read content from text file asynchronously."""
        verified_path = verify_path_within_root(target_path, self.root_dir)
        if not verified_path.exists():
            raise StorageException(
                operation="read_text_file",
                path=str(verified_path),
                reason="File does not exist.",
            )

        try:
            async with aiofiles.open(verified_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as exc:
            logger.error("Failed reading text file %s: %s", verified_path, exc)
            raise StorageException(
                operation="read_text_file",
                path=str(verified_path),
                reason=str(exc),
            ) from exc
