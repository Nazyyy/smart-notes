# ### FILE: app/services/document_service.py
"""
Document Application Service implementing Clean Architecture domain operations.
Coordinates repositories, storage, asynchronous pipeline jobs, and exports.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID, uuid4
import cv2
from fastapi import UploadFile, BackgroundTasks

from app.core.exceptions import DocumentNotFoundException, PageNotFoundException
from app.core.logging import get_logger
from app.models.entities import Document, Page, TextLine, DocumentStatus, PageStatus, ExportFormat
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.services.storage_service import StorageService
from app.services.processing_pipeline import DocumentProcessingPipeline
from app.services.structurer_service import NoteStructurerService

logger = get_logger(__name__)


class DocumentService:
    """High-level application service for document lifecycle and recognition tasks."""

    def __init__(
        self,
        doc_repo: DocumentRepository,
        page_repo: PageRepository,
        storage_service: StorageService,
        pipeline: DocumentProcessingPipeline,
        structurer: NoteStructurerService,
    ) -> None:
        self.doc_repo = doc_repo
        self.page_repo = page_repo
        self.storage = storage_service
        self.pipeline = pipeline
        self.structurer = structurer

    async def create_document_from_upload(
        self,
        title: str,
        description: Optional[str],
        upload_file: UploadFile,
        author: Optional[str] = "default",
        background_tasks: Optional[BackgroundTasks] = None,
        process_immediately: bool = True,
    ) -> Document:
        """
        Ingest an uploaded image, persist document & initial page metadata,
        and enqueue or run the processing pipeline.
        """
        document_id = uuid4()
        page_id = uuid4()

        # Save streamed upload to disk
        saved_file_path = await self.storage.save_uploaded_file(document_id, upload_file)
        file_size = saved_file_path.stat().st_size

        # Create Document entity
        document = Document(
            id=document_id,
            title=title,
            author=author or "default",
            description=description,
            original_filename=upload_file.filename or "upload.jpg",
            file_path=str(saved_file_path),
            file_size_bytes=file_size,
            mime_type=upload_file.content_type or "image/jpeg",
            status=DocumentStatus.PENDING.value,
        )
        await self.doc_repo.create(document)

        # Create Page entity
        page = Page(
            id=page_id,
            document_id=document_id,
            page_number=1,
            raw_image_path=str(saved_file_path),
            status=PageStatus.PENDING.value,
        )
        await self.page_repo.create(page)
        await self.doc_repo.session.commit()

        # Execute or schedule pipeline
        if process_immediately:
            if background_tasks is not None:
                background_tasks.add_task(
                    self.pipeline.execute_page_pipeline,
                    document_id=document_id,
                    page_id=page_id,
                    doc_repo=self.doc_repo,
                    page_repo=self.page_repo,
                )
            else:
                await self.pipeline.execute_page_pipeline(
                    document_id=document_id,
                    page_id=page_id,
                    doc_repo=self.doc_repo,
                    page_repo=self.page_repo,
                )

        full_doc = await self.doc_repo.get_document_with_details(document_id)
        return full_doc or document

    async def get_document(self, document_id: UUID) -> Document:
        """Fetch document by ID or raise DocumentNotFoundException."""
        doc = await self.doc_repo.get_document_with_details(document_id)
        if not doc:
            raise DocumentNotFoundException(document_id)
        return doc

    async def list_documents(
        self, limit: int = 50, offset: int = 0, author: Optional[str] = None
    ) -> Tuple[List[Document], int]:
        """List documents and return total count, optionally filtered by author."""
        items = await self.doc_repo.list_recent(limit=limit, offset=offset, author=author)
        total = await self.doc_repo.count()
        return items, total

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document record and associated filesystem files."""
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise DocumentNotFoundException(document_id)

        # Remove from database (cascades to pages and text_lines)
        deleted = await self.doc_repo.delete(document_id)

        # Clean up filesystem
        try:
            doc_dir = self.storage.get_document_directory(document_id)
            import shutil
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
        except Exception as exc:
            logger.warning("Failed removing filesystem files for %s: %s", document_id, exc)

        return deleted

    async def export_document(
        self,
        document_id: UUID,
        export_format: ExportFormat,
    ) -> str:
        """Generate structured text export for a document in requested format."""
        doc = await self.get_document(document_id)

        # Aggregate lines across all pages
        all_lines: List[TextLine] = []
        for page in sorted(doc.pages, key=lambda p: p.page_number):
            all_lines.extend(sorted(page.lines, key=lambda l: l.line_index))

        content = self.structurer.export(all_lines, export_format, document_title=doc.title)

        # Save to disk
        export_dir = self.storage.get_export_directory(document_id)
        ext_map = {
            ExportFormat.MARKDOWN: "md",
            ExportFormat.TXT: "txt",
            ExportFormat.LATEX: "tex",
            ExportFormat.JSON: "json",
        }
        filename = f"export_{export_format.value.lower()}.{ext_map.get(export_format, 'txt')}"
        file_path = export_dir / filename
        await self.storage.write_text_file(file_path, content)

        # Persist export record
        await self.doc_repo.save_export(
            document_id=document_id,
            export_format=export_format.value,
            file_path=str(file_path),
            content=content,
        )

        return content

    async def export_document_with_ai(
        self,
        document_id: UUID,
        provider: str = "openrouter",
        model: str = "nex-agi/nex-n2.5-pro:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        length_mode: str = "medium",
        enrich_facts: bool = False,
        creativity_mode: str = "strict",
    ) -> str:
        """Synthesize a complete, customized academic study guide using LLM and save as primary Markdown export."""
        from app.config import get_settings
        from app.ml.llm_context_corrector import get_llm_context_corrector, LLMProviderConfig

        doc = await self.get_document(document_id)
        all_lines: List[TextLine] = []
        for page in sorted(doc.pages, key=lambda p: p.page_number):
            all_lines.extend(sorted(page.lines, key=lambda l: l.line_index))

        raw_texts = [line.recognized_text.strip() for line in all_lines if line.recognized_text and line.recognized_text.strip()]

        settings = get_settings()
        key = api_key or (settings.OPENROUTER_API_KEY if provider == "openrouter" else None)
        cfg = LLMProviderConfig(
            provider=provider,
            api_key=key,
            base_url=base_url,
            model=model or settings.OPENROUTER_DEFAULT_MODEL,
        )

        corrector = get_llm_context_corrector()
        synthesized_md = await corrector.synthesize_study_guide(
            raw_texts,
            document_title=doc.title,
            config=cfg,
            length_mode=length_mode,
            enrich_facts=enrich_facts,
            creativity_mode=creativity_mode,
        )

        # Save to disk
        export_dir = self.storage.get_export_directory(document_id)
        file_path = export_dir / "export_markdown.md"
        await self.storage.write_text_file(file_path, synthesized_md)

        # Also write plain text version
        txt_path = export_dir / "export_txt.txt"
        await self.storage.write_text_file(txt_path, synthesized_md)

        # Persist export record
        await self.doc_repo.save_export(
            document_id=document_id,
            export_format=ExportFormat.MARKDOWN.value,
            file_path=str(file_path),
            content=synthesized_md,
        )
        await self.doc_repo.save_export(
            document_id=document_id,
            export_format=ExportFormat.TXT.value,
            file_path=str(txt_path),
            content=synthesized_md,
        )

        return synthesized_md

    async def generate_interactive_kit(
        self,
        document_id: UUID,
        provider: str = "openrouter",
        model: str = "nex-agi/nex-n2.5-pro:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate or retrieve interactive study kit (flashcards, cloze tests, quiz)."""
        from app.config import get_settings
        from app.ml.llm_context_corrector import get_llm_context_corrector, LLMProviderConfig

        doc = await self.get_document(document_id)
        all_lines: List[TextLine] = []
        for page in sorted(doc.pages, key=lambda p: p.page_number):
            all_lines.extend(sorted(page.lines, key=lambda l: l.line_index))

        raw_texts = [line.recognized_text.strip() for line in all_lines if line.recognized_text and line.recognized_text.strip()]

        settings = get_settings()
        key = api_key or (settings.OPENROUTER_API_KEY if provider == "openrouter" else None)
        cfg = LLMProviderConfig(
            provider=provider,
            api_key=key,
            base_url=base_url,
            model=model or settings.OPENROUTER_DEFAULT_MODEL,
        )

        corrector = get_llm_context_corrector()
        kit = await corrector.generate_interactive_study_kit(raw_texts, document_title=doc.title, config=cfg)

        # Save kit to disk cache
        try:
            import json
            export_dir = self.storage.get_export_directory(document_id)
            kit_path = export_dir / "interactive_kit.json"
            await self.storage.write_text_file(kit_path, json.dumps(kit, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning("Failed caching interactive kit to disk: %s", exc)

        return kit


